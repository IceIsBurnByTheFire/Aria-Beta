"""Working out what Aria actually said, as opposed to what she generated.

When a reply is cut off, chunks are already queued in the playback buffer that will
never reach the ear. Recording those in history is the quiet failure mode of barge-in:
Aria believes she answered, the user knows she did not, and every following turn is
built on a false premise. Two or three exchanges later the conversation is incoherent
and the cause is invisible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class WrittenChunk:
    """A chunk handed to playback, and where its audio ends on the utterance timeline."""

    text: str
    end_s: float


def spoken_prefix(chunks: Sequence[WrittenChunk], played_s: float) -> str:
    """The text corresponding to the first `played_s` seconds of audio.

    A chunk straddling the cut is truncated proportionally by word count. That is an
    approximation — words are not equal length — but it is far closer to the truth than
    either dropping the chunk entirely or keeping all of it.
    """
    if played_s <= 0 or not chunks:
        return ""

    out: list[str] = []
    prev_end = 0.0
    for chunk in chunks:
        if played_s >= chunk.end_s - 1e-6:
            out.append(chunk.text)
            prev_end = chunk.end_s
            continue

        span = chunk.end_s - prev_end
        fraction = (played_s - prev_end) / span if span > 0 else 0.0
        words = chunk.text.split()
        kept = int(len(words) * fraction)
        if kept:
            out.append(" ".join(words[:kept]))
        break

    return " ".join(out).strip()
