"""The LLM seam.

One interface, so the conversation backend and the screen-reading backend can differ
without anything downstream noticing. `images` is unused in M1 and exists so the M5
vision path does not force a signature change through every caller.
"""

from __future__ import annotations

from typing import AsyncIterator, Protocol, Sequence, TypedDict


class Message(TypedDict):
    role: str
    content: str


class LLMBackend(Protocol):
    async def stream(
        self, messages: Sequence[Message], images: Sequence[bytes] | None = None
    ) -> AsyncIterator[str]:
        """Yield text fragments as they arrive.

        Must be cancellable: barge-in cancels the consuming task, and the backend is
        expected to stop generating rather than run to completion in the background.
        """
        ...
