"""Spotting "remember that…" and "forget…" before they reach the LLM.

Same reasoning as the screen-watching commands in `vision/intent.py`: when the user
says a thing out loud about her memory, that should *happen*, not be interpreted. Left
to the 9B, "remember I take my coffee black" gets a warm agreeable reply and nothing is
written down — which is the worst outcome, because he now believes she knows.

Deliberately narrow. A false positive writes a note he didn't ask for; a false negative
just means he says it again.
"""

from __future__ import annotations

import re

#: Anchored to the start of the utterance or a clause boundary. An unanchored search
#: turns "I can't remember where I put it" into an instruction to memorise "where I
#: put it", which is both wrong and impossible to notice happening.
_REMEMBER = re.compile(
    r"(?:^|[,.]\s*)(?:please\s+|also\s+|and\s+|oh\s+)*"
    r"remember(?:\s+that|\s+this)?\b[:,]?\s*(?P<what>.+)",
    re.IGNORECASE,
)
_FORGET = re.compile(
    r"(?:^|[,.]\s*)(?:please\s+|also\s+|and\s+)*"
    r"forget(?:\s+that|\s+about)?\b[:,]?\s*(?P<what>.+)",
    re.IGNORECASE,
)
#: "I can't remember", "I don't remember" — him describing himself, not asking her.
_NEGATED = re.compile(
    r"\b(?:can'?t|cannot|don'?t|do not|didn'?t|never|won'?t|couldn'?t)\s+"
    r"(?:remember|forget)\b",
    re.IGNORECASE,
)
#: "what do you remember", "do you remember anything about me"
_RECALL = re.compile(
    r"\bwhat do you (?:remember|know)\b|\bdo you remember (?:anything|much)\b",
    re.IGNORECASE,
)

#: "remember to call mum" is a reminder, not a fact about him, and this has no timers.
_IS_REMINDER = re.compile(r"^\s*to\s+\w", re.IGNORECASE)


def memory_command(text: str) -> tuple[str, str] | None:
    """Return ("remember"|"forget"|"recall", content) or None.

    `recall` carries no content — her memories are already in the system prompt, so
    she can answer that one herself.
    """
    if _NEGATED.search(text):
        return None
    if _RECALL.search(text):
        return "recall", ""
    if m := _FORGET.search(text):
        what = m.group("what").strip(" .!?")
        return ("forget", what) if what else None
    if m := _REMEMBER.search(text):
        what = m.group("what").strip(" .!?")
        if not what or _IS_REMINDER.match(what):
            return None
        return "remember", what
    return None


def as_note(what: str) -> str:
    """Turn what he said into a note, keeping his own words inside quotes.

    The obvious approach — rewrite the pronouns into third person so it matches the
    notes the background pass writes — swaps "I" for "he" and leaves the verb behind:
    "I take my coffee black" becomes "he take his coffee black". Fixing that properly
    means conjugating, and a verb table is a queue of new bugs for no benefit.

    Quoting sidesteps the grammar entirely and is more honest anyway: a note he
    dictated is *his words*, and reading it back months later he can tell it apart
    from something she inferred about him.
    """
    text = " ".join(what.strip().split())
    text = re.sub(r"^\s*(?:that|this)\s+", "", text, flags=re.I)
    text = text.strip(" .!?,")
    if not text:
        return ""
    return f'he asked you to remember: "{text}"'
