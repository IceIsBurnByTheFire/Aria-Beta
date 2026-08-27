"""The conversation model in the cloud, over any OpenAI-compatible endpoint.

Why this exists: the local 9B is fast, private and free, and it is also the reason
several features in this project needed defending against it — it copies persona
examples verbatim, invents shared history, ends every reply with a question, and drops
a bracket off an emotion marker often enough to matter. A much larger model does all of
that better.

**One backend, three providers.** OpenRouter, Groq and Google all publish the same
OpenAI-compatible chat-completions surface, so what differs between them is a URL, an
environment variable and a couple of request knobs — a table in `config.CLOUD_PROVIDERS`,
not three classes. Adding a fourth is a dict entry.

Three things about the cloud path are worth knowing before reading the code.

**The daily budget is the design constraint.** OpenRouter's free tier is 50 requests a
day; Groq's is 1000 on a 70B. Aria makes two calls per turn — the reply, then the
background memory pass — so the naive version halves whatever the provider allows. That
is why `llm.background_for` pins extraction to the local model.

**Latency is a different shape.** Local generation starts in ~200 ms and streams at GPU
speed. A cloud call adds a round trip plus queueing on a tier that is explicitly
best-effort. Fine on Discord, where nobody is listening to silence. Felt immediately out
loud.

**Everything said goes to a third party**, including her memory notes, which sit in the
system prompt. On Google's free tier it is also used to train. That is a real trade
against the rest of this project, and it is why local stays the default.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator, Sequence

import httpx

from ..config import LLMConfig
from .base import Message

log = logging.getLogger(__name__)

#: Server-side capacity, not quota. Retrying is the correct response and it usually
#: clears within a second or two.
_TRANSIENT = frozenset({500, 502, 503, 504, 529})
_TRANSIENT_TRIES = 3
#: Doubles each attempt, so a turn spends at most ~2.4 s on retries before answering
#: honestly. Longer than that and she is silent for so long it reads as a hang.
_BACKOFF_S = 0.8


_THINK_OPEN, _THINK_CLOSE = "<think>", "</think>"


class _ThinkFilter:
    """Drop `<think>…</think>` blocks from a token stream.

    Reasoning models are supposed to keep reasoning out of `content`, and most do —
    OpenRouter takes `reasoning: {exclude}`, Gemini takes `reasoning_effort`. Groq's
    `qwen/qwen3.6-27b` does not: it emits a literal `<think>` block into the visible
    content, observed opening with "Here's a thinking process:". Unfiltered that reaches
    the synthesiser and gets read out loud, or posted to Discord.

    Provider knobs cannot cover this — the leaking model varies, and setting a
    reasoning parameter on a *non*-reasoning model like `llama-3.3-70b-versatile` would
    cost a rejected request and a retry on every single turn. Filtering the stream costs
    nothing and works regardless of which model is selected.

    Streaming-safe: a tag can arrive split across chunks, so the tail is held back only
    when it could still be the start of one.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._inside = False

    def feed(self, text: str) -> str:
        self._buf += text
        out: list[str] = []
        while True:
            if self._inside:
                end = self._buf.find(_THINK_CLOSE)
                if end == -1:
                    # Keep only enough to recognise a closing tag split across chunks.
                    self._buf = self._buf[-(len(_THINK_CLOSE) - 1):]
                    break
                self._buf = self._buf[end + len(_THINK_CLOSE):]
                self._inside = False
                continue

            start = self._buf.find(_THINK_OPEN)
            if start != -1:
                out.append(self._buf[:start])
                self._buf = self._buf[start + len(_THINK_OPEN):]
                self._inside = True
                continue

            # No tag. Emit everything except a trailing fragment that might become one.
            angle = self._buf.rfind("<")
            if angle == -1 or len(self._buf) - angle >= len(_THINK_OPEN):
                out.append(self._buf)
                self._buf = ""
            else:
                out.append(self._buf[:angle])
                self._buf = self._buf[angle:]
            break
        return "".join(out)

    def flush(self) -> str:
        """Whatever is left at end of stream. An unclosed block is discarded, not spoken."""
        rest = "" if self._inside else self._buf
        self._buf = ""
        return rest


class CloudLLM:
    def __init__(self, cfg: LLMConfig, transport: httpx.AsyncBaseTransport | None = None):
        """`transport` is a test seam. Swapping the whole client out instead means the
        tests stop covering the headers and timeouts set here, which is how a missing
        Authorization header passes a green suite."""
        self._cfg = cfg
        provider = cfg.provider
        if provider is None:
            raise ValueError(f"{cfg.backend!r} is not a cloud provider")
        self._p = provider
        self._client = httpx.AsyncClient(
            base_url=provider.base_url,
            timeout=httpx.Timeout(cfg.timeout_s, connect=10.0),
            transport=transport,
            headers={
                "Authorization": f"Bearer {cfg.api_key}",
                # Optional everywhere; OpenRouter uses them for public app rankings.
                "HTTP-Referer": cfg.app_url,
                "X-Title": cfg.app_name,
            },
        )

    @property
    def label(self) -> str:
        return f"{self._cfg.active_cloud_model} ({self._cfg.backend})"

    # --- lifecycle ------------------------------------------------------------
    async def preflight(self) -> str | None:
        """Check what can be checked without spending a request.

        The Claude vision backend deliberately spends one token here, because
        reachability is not usability and a key with no credit passes every cheap check
        then fails on the first screenshot. The trade inverts on a free tier: a test
        request is two percent of OpenRouter's daily allowance, and auth failures are
        already legible — `stream` turns a 401 into a sentence naming the fix.
        """
        if not self._cfg.api_key:
            return (
                f"Cloud mode needs a {self._cfg.backend} API key, and none was found.\n"
                f"  Get one free at {self._p.keys_url}, then put it in core/.env as:\n"
                f"    {self._p.key_env}=...\n"
                f"  Or start her locally instead (Start Aria.bat)."
            )
        if self._p.key_prefix and not self._cfg.api_key.startswith(self._p.key_prefix):
            return (
                f"{self._p.key_env} doesn't look like a {self._cfg.backend} key - those\n"
                f"  start with '{self._p.key_prefix}'. Check you didn't paste a different\n"
                f"  service's key into it."
            )
        return None

    async def warmup(self) -> None:
        """Nothing to warm. There are no weights here and no request to waste."""

    async def close(self) -> None:
        await self._client.aclose()

    # --- generation -----------------------------------------------------------
    async def stream(
        self,
        messages: Sequence[Message],
        images: Sequence[bytes] | None = None,
        num_predict: int | None = None,
    ) -> AsyncIterator[str]:
        """Yield text fragments, and turn every failure into something she can say.

        A backend that raises here takes the turn down and prints a traceback naming
        httpx. `_report_turn_failure` would catch it, but "she said nothing and the
        terminal mentions a status code" is a much worse answer than her saying the
        limit is gone until tomorrow — especially when that is the *expected* end of a
        free account's afternoon rather than a malfunction.
        """
        if images:
            log.warning("cloud chat backend ignores images; screen reading is vision/")

        payload = {
            "model": self._cfg.active_cloud_model,
            "messages": [dict(m) for m in messages],
            "stream": True,
            "temperature": self._cfg.temperature,
            "max_tokens": num_predict or self._cfg.num_predict,
            **self._p.extra,
        }

        # Two independent retries, for two unrelated failures.
        #
        # Provider-specific fields are the likeliest thing to rot: three services that
        # change independently share one table. A single retry without them turns "this
        # provider withdrew a parameter" from a dead assistant into a chattier one.
        #
        # Free tiers also run out of *capacity* rather than quota — Google answers 503
        # "experiencing high demand" often enough on a free key to matter, and it clears
        # in a second or two. Giving up on the first one would make her look broken
        # several times an evening.
        bare = {k: v for k, v in payload.items() if k not in self._p.extra}
        variants = [payload] + ([bare] if self._p.extra else [])

        for index, body in enumerate(variants):
            can_drop_extras = index == 0 and len(variants) > 1
            for attempt in range(_TRANSIENT_TRIES):
                try:
                    async with self._client.stream(
                        "POST", "chat/completions", json=body
                    ) as reply:
                        if reply.status_code == 400 and can_drop_extras:
                            await reply.aread()
                            log.warning("%s rejected %s; retrying without: %s",
                                        self._cfg.backend, list(self._p.extra),
                                        reply.text[:200])
                            break  # next variant
                        if reply.status_code in _TRANSIENT and attempt < _TRANSIENT_TRIES - 1:
                            await reply.aread()
                            wait = _BACKOFF_S * (2 ** attempt)
                            log.info("%s busy (%d); retrying in %.1fs",
                                     self._cfg.backend, reply.status_code, wait)
                            await asyncio.sleep(wait)
                            continue
                        if reply.status_code != 200:
                            await reply.aread()
                            yield self._explain(reply.status_code, reply.text)
                            return
                        async for text in self._parse(reply):
                            yield text
                        return
                except httpx.TimeoutException:
                    log.error("%s timed out after %.0fs",
                              self._cfg.backend, self._cfg.timeout_s)
                    yield " Sorry, the cloud model didn't answer in time."
                    return
                except httpx.HTTPError as e:
                    log.error("%s request failed: %s: %s",
                              self._cfg.backend, type(e).__name__, e)
                    yield " I couldn't reach the cloud model just now."
                    return

    async def _parse(self, reply: httpx.Response) -> AsyncIterator[str]:
        think = _ThinkFilter()
        async for line in reply.aiter_lines():
            # OpenRouter sends `: OPENROUTER PROCESSING` keep-alive comments while a
            # free-tier request queues. They are not data, and parsing them as JSON is
            # the obvious way to break on exactly the slow requests this must survive.
            if not line.startswith("data: "):
                continue
            body = line[6:].strip()
            if body == "[DONE]":
                break
            try:
                chunk = json.loads(body)
            except ValueError:
                continue
            for choice in chunk.get("choices", ()):
                if text := (choice.get("delta") or {}).get("content"):
                    if visible := think.feed(text):
                        yield visible
        if tail := think.flush():
            yield tail

    def _explain(self, status: int, body: str) -> str:
        """One sentence she can say, and the real reason in the terminal."""
        log.error("%s returned %d: %s", self._cfg.backend, status, body[:400])
        if status in (401, 403):
            self._shout(f"{self._cfg.backend} rejected the API key. Check "
                        f"{self._p.key_env} in core/.env, or make a new key at "
                        f"{self._p.keys_url}.")
            return " The cloud model rejected my key. The terminal says how to fix it."
        if status == 402:
            return " That model needs credit on the account."
        if status == 429:
            # The expected end of a free account's day, not a malfunction, so it gets
            # the plainest possible explanation and a way out.
            self._shout(f"{self._cfg.backend} rate limit reached. The free tier allows "
                        f"{self._p.limits}.\nStart Aria.bat runs the local model instead, "
                        f"with no limit.")
            return (
                " I've hit the free limit on the cloud model. Restart me on the local "
                "one and I'll keep going."
            )
        if status in _TRANSIENT:
            # Already retried and still busy. This is the free tier being oversubscribed
            # rather than anything wrong with the setup, so it must not read as an error
            # he needs to go and fix.
            return " The cloud model's busy right now. Ask me again in a second."
        if status == 404:
            self._shout(f"Model '{self._cfg.active_cloud_model}' was not found on "
                        f"{self._cfg.backend}.\nSet ARIA_CLOUD_MODEL in core/.env, or run "
                        f"--list-cloud-models to see what is available.")
            return " That cloud model doesn't exist any more. The terminal has the details."
        return f" The cloud model returned an error, {status}. It's in the terminal."

    @staticmethod
    def _shout(text: str) -> None:
        print(f"\n\033[33m{text}\033[0m\n")
