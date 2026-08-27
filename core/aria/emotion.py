"""Pulling `[emotion]` markers out of the LLM's output stream.

Inline markers beat a structured side-channel here for one reason: they arrive *in time
order with the speech*, so an emotion can be tied to the sentence it belongs to rather
than to the reply as a whole. A JSON blob at the end would land every expression after
the fact.

They must be removed before the text reaches TTS. `clean_for_speech` strips markdown but
not brackets, so an unhandled marker is read aloud as "bracket happy bracket".
"""

from __future__ import annotations

import functools
import re
from typing import Sequence

#: Deliberately narrow: letters and underscores, 2-24 chars. Wide enough for
#: `no_eye_highlight`, tight enough that "[1]", "[see below]" or a stray bracket in
#: normal prose is left alone rather than silently eaten from the speech.
#: Public because the Discord path needs to find markers *in place*, before they are
#: stripped — a marker's position is what says which sentence the feeling belongs to.
MARKER = re.compile(r"\[\s*([A-Za-z][A-Za-z_]{1,23})\s*\]")


@functools.lru_cache(maxsize=8)
def _with_debris(known: tuple[str, ...]) -> re.Pattern[str]:
    """Also catch a marker that lost one of its brackets on the way out of the model.

    A 9B drops a bracket every so often, and the result is not a marker that fails to
    fire — it is the word arriving as *text*: `…whatever sauce you have. shy] Tell me…`
    posted to Discord, or read aloud as "shy" mid-sentence. Rare enough to survive a
    hundred turns unnoticed and obvious enough to ruin the one it appears in.

    Safe only because it is anchored to a closed vocabulary — the emotions the loaded
    character actually has — plus at least one surviving bracket. A bare "happy" in
    ordinary prose matches nothing here, which is the entire reason the vocabulary has
    to be passed in rather than guessed.
    """
    alt = "|".join(re.escape(k) for k in sorted(known, key=len, reverse=True))
    return re.compile(
        rf"\[\s*[A-Za-z][A-Za-z_]{{1,23}}\s*\]"   # well formed, the overwhelming majority
        rf"|\[\s*(?:{alt})\b(?!\s*\])"            # opened, never closed
        rf"|(?<![\w\]])(?:{alt})\s*\]",           # closed, never opened
        re.IGNORECASE,
    )


def extract(
    text: str, known: Sequence[str] | None = None
) -> tuple[str, list[str]]:
    """Split `text` into speakable text and the emotions it carried.

    Returns (clean_text, emotions) with emotions lowercased, in order of appearance.

    Pass `known` — what the loaded character can actually show — to also recover
    markers that arrived with a bracket missing. Without it only well-formed markers
    are recognised, which is the right behaviour when there is no character attached
    and therefore no vocabulary to be sure about.
    """
    pattern = _with_debris(tuple(known)) if known else MARKER
    found = [
        m.group(0).strip("[] \t").lower() for m in pattern.finditer(text)
    ]
    if not found:
        return text, []
    cleaned = pattern.sub("", text)
    # Markers usually sit against a space on one side; collapse what they leave behind.
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    # And a marker sitting directly before punctuation leaves a single orphan space,
    # which the rule above does not catch: "[happy]." becomes " ." rather than "..".
    # Invisible in speech, and posted to Discord it reads as a typo in every message
    # where the model closes a clause with a marker — which Gemini does routinely.
    return re.sub(r"\s+([.,!?;:…])", r"\1", cleaned).strip(), found


def has_open_marker(text: str) -> bool:
    """True if `text` ends inside an unclosed `[`.

    The chunker asks this before cutting, so a marker is never split across two chunks
    — half a marker is both unrecognisable and audible.
    """
    open_at = text.rfind("[")
    return open_at != -1 and "]" not in text[open_at:]
