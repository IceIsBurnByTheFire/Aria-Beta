"""Only answering when she's actually being spoken to.

Always-listening is fine alone in a room and awful anywhere else: a podcast, a phone
call, someone else in the flat, and she answers all of it. A wake word fixes that, and
the roadmap assumed openWakeWord would be how.

**It isn't, and the reason is that the pipeline already solved this.** openWakeWord
ships models for "alexa", "hey jarvis" and a handful of others — none of them "Aria" —
so using it means generating synthetic training data, running a notebook, and owning a
model file forever. All to recognise a word that Whisper, already running on every
utterance and vastly more accurate, hands over for free. The gate belongs after STT,
not before it.

The cost is honest: background speech is still transcribed, so the GPU does work for a
turn that gets thrown away. What it *doesn't* do is reach the LLM, the TTS, or her
mouth — which is the part that was actually annoying. On a machine with an idle 5080
that trade is obvious. On a laptop on battery it wouldn't be, and openWakeWord would
earn its keep.

Conversation is windowed rather than one-shot: saying her name opens a few seconds of
ordinary back-and-forth, because "Aria, what's the weather" / "Aria, and tomorrow?" is
not how people talk.
"""

from __future__ import annotations

import difflib
import re
import time
from dataclasses import dataclass, field

#: Whisper is good but not psychic, and "Aria" is a short word with common neighbours.
#: These are what it actually produces for her name; anything not on the list still
#: gets a fuzzy pass below.
VARIANTS = {
    "aria": ("aria", "arya", "ariya", "area", "arial", "ariah", "aaria", "haria"),
}

#: Filler that people put in front of a name and never mean as part of the request.
_LEAD = re.compile(r"^\s*(?:hey|hi|hello|ok|okay|yo|um|uh|so)\b[\s,]*", re.I)
#: Case-insensitive on purpose. Lowercase-only matched just "ria" inside "Aria" and
#: left the capital behind, so the model was asked "A, what's the weather".
_WORD = re.compile(r"[A-Za-z']+")


@dataclass
class WakeWord:
    """Decides whether an utterance is addressed to her, and strips her name off it."""

    word: str = "aria"
    #: How long an exchange keeps her awake. Long enough for a follow-up, short enough
    #: that the room's conversation doesn't get adopted after she happens to be named.
    window_s: float = 30.0
    #: How close a heard word has to be. 0.8 catches "arya" and "ariya" while leaving
    #: "maria" and "aware" alone — both of which are one edit further out than they
    #: look, and both of which turned up while tuning this.
    similarity: float = 0.8
    _awake_until: float = field(default=0.0, repr=False)

    @property
    def variants(self) -> tuple[str, ...]:
        return VARIANTS.get(self.word, (self.word,))

    def is_awake(self, now: float | None = None) -> bool:
        return (now or time.monotonic()) < self._awake_until

    def stay_awake(self, now: float | None = None) -> None:
        """Extend the window. Called after every real exchange, so a conversation
        continues without her name in front of every sentence."""
        self._awake_until = (now or time.monotonic()) + self.window_s

    def sleep(self) -> None:
        self._awake_until = 0.0

    def _is_name(self, token: str) -> bool:
        token = token.lower()
        if token in self.variants:
            return True
        # The first letter has to agree before similarity gets a vote. Without that,
        # "Maria" scores 0.89 against "aria" — it contains the whole word — and she
        # answers every time someone mentions a person by that name. Every genuine
        # mishearing Whisper produces either starts with the same letter or is on the
        # explicit list above.
        if token[:1] != self.word[:1]:
            return False
        # And it has to be roughly the same length. "aerial" scores exactly 0.80
        # against "aria" — it contains "ria" — so threshold alone lets it through, and
        # she answers the sentence about the broken aerial. A mishearing of a
        # four-letter word is not six letters long. The explicit list above carries
        # the real Whisper outputs; this fuzzy path is only a net for unlisted ones,
        # so it can afford to be strict.
        if abs(len(token) - len(self.word)) > 1:
            return False
        return difflib.SequenceMatcher(None, token, self.word).ratio() >= self.similarity

    def hears_name(self, text: str) -> bool:
        return any(self._is_name(t) for t in _WORD.findall(text))

    def strip_name(self, text: str) -> str:
        """Take her name out of the request before it reaches the model.

        "Aria, what's the weather" is addressed to her; "what's the weather" is what
        she was asked. Leaving it in means every single turn arrives with her being
        called by name, and she starts answering as if it were remarkable.
        """
        out = _WORD.sub(
            lambda m: "" if self._is_name(m.group(0)) else m.group(0), text
        )
        out = _LEAD.sub("", out)
        out = re.sub(r"\s{2,}", " ", out)
        out = re.sub(r"^[\s,.!?]+", "", out)
        return out.strip()

    def should_answer(self, text: str, now: float | None = None) -> tuple[bool, str]:
        """(answer, text_for_the_model).

        Hearing her name always wakes her, window or no window — being addressed
        directly is not something to ignore because a timer lapsed.
        """
        now = now or time.monotonic()
        if self.hears_name(text):
            self.stay_awake(now)
            stripped = self.strip_name(text)
            # "Aria?" on its own is being called, not asked. Hand the model the name
            # back so it has something to answer rather than an empty string.
            return True, stripped or text.strip()
        if self.is_awake(now):
            self.stay_awake(now)  # each exchange resets the clock
            return True, text
        return False, text
