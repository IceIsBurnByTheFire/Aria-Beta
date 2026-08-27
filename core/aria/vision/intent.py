"""Deciding whether something the user said is actually about the screen.

Attaching a screenshot to every turn would be wasteful, slow and — since capture is a
genuinely invasive capability — rude. Asking the LLM to decide costs a whole extra
round trip before the real one. So this is a keyword test: it runs in microseconds,
never adds latency, and is wrong in a way that is obvious rather than mysterious.

False negatives are the safe failure. Missing a reference means Aria answers without
looking, and the user rephrases. A false positive sends a screenshot nobody asked to
send, so the patterns are deliberately specific rather than generous.
"""

from __future__ import annotations

import re

#: Explicit references to the display itself.
_SCREEN_WORDS = r"(?:screen|display|monitor|desktop)"

#: Things on it that people ask about by name.
_ARTEFACTS = r"(?:error|exception|traceback|stack ?trace|warning|dialog|popup|window|tab|code|log|message)"

_PATTERNS = [
    # "what's on my screen", "look at my screen", "can you see my screen"
    rf"\b(?:on|at|to|see|check|read|look)\b[^.?!]{{0,20}}\bmy {_SCREEN_WORDS}\b",
    rf"\byour {_SCREEN_WORDS}\b",
    rf"\bthe {_SCREEN_WORDS}\b",
    # "what am I looking at", "what do you see"
    r"\bwhat (?:am i|are we) (?:looking at|seeing)\b",
    r"\bwhat do you see\b",
    # "what does this error say", "read this error", "this traceback"
    rf"\b(?:this|that|these|those|the) {_ARTEFACTS}\b",
    # "look at this", "check this out", "see this" — deictic, only with a look verb
    r"\b(?:look at|check out|check|read|see|explain) (?:this|that|these)\b",
    # "what does this say" — asking to be read something is always about the screen
    r"\bwhat does (?:this|that|it) say\b",
    # "help me fix this", "what's wrong here"
    r"\bwhat(?:'s| is) (?:wrong|happening|going on) (?:here|with this)\b",
    r"\bwhat(?:'s| is) this\b",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _PATTERNS]

#: Explicit control, handled before the loop ever gets to intent.
_ENABLE = re.compile(
    r"\b(?:start|begin|you can) (?:watching|looking at|seeing) (?:my )?(?:screen|display)\b"
    r"|\bwatch my (?:screen|display)\b",
    re.IGNORECASE,
)
_DISABLE = re.compile(
    r"\bstop (?:watching|looking at|seeing) (?:my )?(?:screen|display)\b"
    r"|\b(?:don't|do not) (?:watch|look at) my (?:screen|display)\b",
    re.IGNORECASE,
)


def wants_screen(text: str) -> bool:
    """True if this utterance is asking about what is on screen."""
    return any(p.search(text) for p in _COMPILED)


def capture_command(text: str) -> bool | None:
    """`True` to switch capture on, `False` off, `None` if it said no such thing."""
    if _DISABLE.search(text):
        return False
    if _ENABLE.search(text):
        return True
    return None
