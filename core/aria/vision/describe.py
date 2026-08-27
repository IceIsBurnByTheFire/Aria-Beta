"""Turning a screenshot into a spoken answer.

Two backends behind one interface, because neither is universally available: the
local one needs a ~6 GB model pulled, the Claude one needs an API key. Both stream,
so the first sentence reaches TTS while the rest is still being written — a vision
turn is slower than a normal one and that head start matters.
"""

from __future__ import annotations

import base64
import logging
from typing import AsyncIterator, Protocol

from ..config import VISION_PROMPT, VisionConfig
from .capture import Screenshot

log = logging.getLogger(__name__)


class VisionBackend(Protocol):
    async def preflight(self) -> str | None:
        """None if usable, else a human-readable description of what is missing."""
        ...

    async def describe(self, question: str, shot: Screenshot) -> AsyncIterator[str]:
        ...


class OllamaVision:
    """Local vision through Ollama. Free, private, offline, and weaker at small text."""

    def __init__(self, cfg: VisionConfig):
        import ollama

        self._cfg = cfg
        self._client = ollama.AsyncClient()

    async def preflight(self) -> str | None:
        try:
            listed = await self._client.list()
        except Exception:
            return (
                "Cannot reach Ollama for screen reading.\n"
                "  Start it with:  ollama serve"
            )
        if self._cfg.ollama_model not in [m.model for m in listed.models]:
            return (
                f"Screen reading needs a vision model that is not installed.\n"
                f"  Install it with:  ollama pull {self._cfg.ollama_model}\n"
                f"  (~6 GB. Or set ARIA_VISION_BACKEND=claude to use the API instead.)"
            )
        return None

    async def describe(self, question: str, shot: Screenshot) -> AsyncIterator[str]:
        stream = await self._client.chat(
            model=self._cfg.ollama_model,
            messages=[
                {"role": "system", "content": VISION_PROMPT + self._cfg.language_prompt},
                {"role": "user", "content": question, "images": [shot.data]},
            ],
            stream=True,
            think=False,  # same reason as the chat model: thinking never yields speech
            options={"num_predict": 200},
        )
        async for part in stream:
            if part.message.content:
                yield part.message.content


class ClaudeVision:
    """Claude for screen reading, where the quality gap over a local model is largest.

    Model, thinking and effort settings follow the current API guidance rather than
    being tuned by hand. Thinking is left unset, which on Sonnet 5 means adaptive —
    the only on-mode it has. That is deliberate: `thinking.display` defaults to
    omitted and `text_stream` yields text deltas only, so no reasoning can reach TTS,
    while *disabling* thinking is the setting that makes internal tags leak into the
    visible answer. `budget_tokens` and the sampling parameters are gone on this
    model; effort is the only depth control.
    """

    def __init__(self, cfg: VisionConfig):
        import anthropic

        self._cfg = cfg
        self._anthropic = anthropic
        self._client = anthropic.AsyncAnthropic()

    #: What the SDK raises when it cannot resolve *any* credential source. It is a
    #: TypeError rather than AuthenticationError — the request is never sent — so it
    #: needs catching explicitly or it surfaces as an unhelpful generic failure.
    _NO_CREDENTIALS = "Could not resolve authentication method"

    async def preflight(self) -> str | None:
        no_credentials = (
            "Screen reading via Claude needs credentials, and none were found.\n"
            "  Set ANTHROPIC_API_KEY, or install the Anthropic CLI and run:  ant auth login\n"
            "  (Or set ARIA_VISION_BACKEND=ollama to keep everything local.)"
        )
        try:
            await self._client.models.retrieve(self._cfg.claude_model)
        except TypeError as e:
            if self._NO_CREDENTIALS in str(e):
                return no_credentials
            raise
        except self._anthropic.AuthenticationError:
            return no_credentials
        except self._anthropic.NotFoundError:
            return f"Model '{self._cfg.claude_model}' not found or not available to this key."
        except self._anthropic.APIConnectionError:
            return "Cannot reach the Claude API — check the network connection."
        except Exception as e:  # noqa: BLE001 — surface anything else as-is
            return f"Claude vision unavailable: {type(e).__name__}: {e}"

        # Reachability is not usability. A key with no credit passes every check
        # above and then fails on the first screenshot — which is the failure this
        # whole preflight exists to prevent, so spend one token proving it works.
        try:
            await self._client.messages.create(
                model=self._cfg.claude_model,
                max_tokens=1,
                thinking={"type": "disabled"},  # nothing to reason about; keep it cheap
                messages=[{"role": "user", "content": "."}],
            )
        except self._anthropic.BadRequestError as e:
            if "credit balance" in str(e).lower():
                return (
                    "Screen reading via Claude is out of credit.\n"
                    "  Add credit at console.anthropic.com under Plans & Billing.\n"
                    "  (Or set ARIA_VISION_BACKEND=ollama to use the local model instead.)"
                )
            return f"Claude rejected a test request: {e}"
        except self._anthropic.PermissionDeniedError as e:
            return f"This API key isn't permitted to use {self._cfg.claude_model}: {e}"
        except self._anthropic.RateLimitError:
            pass  # rate-limited *now* says nothing about a screen question later
        except Exception as e:  # noqa: BLE001
            return f"Claude vision unavailable: {type(e).__name__}: {e}"
        return None

    async def describe(self, question: str, shot: Screenshot) -> AsyncIterator[str]:
        image = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.standard_b64encode(shot.data).decode(),
            },
        }
        try:
            async with self._client.messages.stream(
                model=self._cfg.claude_model,
                max_tokens=self._cfg.max_tokens,
                system=VISION_PROMPT + self._cfg.language_prompt,
                output_config={"effort": self._cfg.effort},
                messages=[{"role": "user", "content": [image, {"type": "text", "text": question}]}],
            ) as stream:
                async for text in stream.text_stream:
                    yield text
                final = await stream.get_final_message()

            # A screenshot can trip a safety classifier; that arrives as a normal
            # 200 with no usable content rather than an exception.
            if final.stop_reason == "refusal":
                log.warning("Claude declined to describe the screen")
                yield " I'm not able to describe what's on screen right now."
        except self._anthropic.RateLimitError:
            yield " I've hit the rate limit for looking at your screen."
        except self._anthropic.APIConnectionError:
            yield " I couldn't reach the service to look at your screen."
        except self._anthropic.APIStatusError as e:
            # Billing, permissions, an oversized image. Preflight catches these at
            # arming time, but a balance can run out mid-conversation — so say
            # something true and put the actual reason in the terminal.
            log.error("Claude vision request failed: %s", e)
            if "credit balance" in str(e).lower():
                yield " I can't look at your screen — the Claude account is out of credit."
            else:
                yield " Something went wrong looking at your screen. The terminal has the details."


def build(cfg: VisionConfig) -> VisionBackend:
    if cfg.backend == "claude":
        return ClaudeVision(cfg)
    if cfg.backend == "ollama":
        return OllamaVision(cfg)
    raise ValueError(f"unknown vision backend {cfg.backend!r} (want 'ollama' or 'claude')")
