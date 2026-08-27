"""Local LLM over Ollama."""

from __future__ import annotations

import logging
from typing import AsyncIterator, Sequence

import ollama

from ..config import LLMConfig
from .base import Message

log = logging.getLogger(__name__)


class OllamaLLM:
    def __init__(self, cfg: LLMConfig):
        self._cfg = cfg
        self._client = ollama.AsyncClient()

    async def preflight(self) -> str | None:
        """Check Ollama is usable. Returns None if fine, else what to tell the user.

        Worth doing properly at startup: without it the first failure surfaces as a
        ninety-line httpx traceback from inside the streaming call, which says
        "All connection attempts failed" and nothing about Ollama.
        """
        try:
            listed = await self._client.list()
        except Exception:
            return (
                "Cannot reach Ollama — the language model backend is not running.\n"
                "  Start it in another terminal with:  ollama serve\n"
                "  Aria will keep listening and retry on the next thing you say."
            )

        models = [m.model for m in listed.models]
        if self._cfg.model not in models:
            installed = "\n    ".join(models) if models else "(none)"
            return (
                f"Ollama is running but '{self._cfg.model}' is not installed.\n"
                f"  Install it with:  ollama pull {self._cfg.model}\n"
                f"  Currently installed:\n    {installed}"
            )
        return None

    async def warmup(self) -> None:
        """Force the weights resident so the first real turn is not a model load."""
        try:
            async for _ in await self._client.chat(
                model=self._cfg.model,
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
                think=self._cfg.think,
                options={"num_predict": 1},
                keep_alive=self._cfg.keep_alive,
            ):
                break
        except Exception as e:
            # Non-fatal: the first turn just pays the model load. Debug rather than
            # warning because preflight() has already said the useful version of this
            # to the user, and repeating it as a stack-trace-flavoured warning only
            # buries the actionable message.
            log.debug("LLM warmup failed: %s", e)

    async def stream(
        self,
        messages: Sequence[Message],
        images: Sequence[bytes] | None = None,
        num_predict: int | None = None,
    ) -> AsyncIterator[str]:
        """`num_predict` overrides the configured budget for one call.

        The default is tuned for speech, where a long reply is a long wait with no way
        to skim. A Discord message is read, not heard, so it can afford a paragraph —
        and nothing else about generation should differ between the two.
        """
        payload = list(messages)
        if images:
            payload[-1] = {**payload[-1], "images": list(images)}

        stream = await self._client.chat(
            model=self._cfg.model,
            messages=payload,
            stream=True,
            # Reasoning models route everything to `thinking` and the spoken `content`
            # never arrives inside a usable budget. See config.LLMConfig.think.
            think=self._cfg.think,
            options={
                "temperature": self._cfg.temperature,
                "num_predict": num_predict or self._cfg.num_predict,
            },
            keep_alive=self._cfg.keep_alive,
        )
        async for part in stream:
            if part.message.content:
                yield part.message.content
