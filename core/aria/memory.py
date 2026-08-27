"""What she knows about him between sessions, and how long she's known him.

Memory and identity turn out to be the same problem seen from two sides. The persona
tells her to notice things and bring them up later, but until now there was no later —
so she invented one. "That's the third time this week, isn't it?" on a fresh process
with an empty history is charming exactly once, and unsettling the moment you notice.
Real recall is the fix, and so is knowing when she has none.

Two things live here:

- **Notes.** Short third-person facts about him, saved either because he asked or
  because a background pass thought they were worth keeping. Capped, deduped, and
  stored as plain readable JSON so he can open the file and delete anything he
  doesn't like the look of.
- **Continuity.** When they first met, when they last spoke, how many times. This is
  what lets her say "it's been three days" truthfully instead of guessing.

The write path is deliberately off the critical path — extraction is another call to a
9B, and nothing about remembering is worth making a reply slower.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from uuid import uuid4

log = logging.getLogger(__name__)

#: Enough for a long relationship, small enough that the whole set fits the prompt
#: without crowding out the persona. Past this the oldest auto-notes go first.
MAX_NOTES = 120

#: Notes he asked for by name are never evicted to make room for ones she guessed at.
SOURCE_USER, SOURCE_AUTO = "you", "auto"


@dataclass
class Note:
    text: str
    created_at: float
    source: str = SOURCE_AUTO
    #: Stable identity, so the panel can edit and delete an exact note.
    #:
    #: `forget` matches on text and matches loosely on purpose — said out loud, "forget
    #: about the demo" should catch the note however it happens to be worded. That is
    #: right for a voice command and wrong for a button next to one specific line: the
    #: substring rule cuts both ways, so deleting "he has a cat called Widget" would
    #: also take a note that just said "cat". Notes written before this field existed
    #: get an id on load, so an old memory.json needs no migration.
    id: str = field(default_factory=lambda: uuid4().hex[:12])

    @property
    def age_days(self) -> float:
        return (time.time() - self.created_at) / 86400


@dataclass
class Continuity:
    first_seen: float = 0.0
    last_seen: float = 0.0
    sessions: int = 0


@dataclass
class Memory:
    path: Path
    max_notes: int = MAX_NOTES
    notes: list[Note] = field(default_factory=list)
    continuity: Continuity = field(default_factory=Continuity)
    #: Set at session start and held for the process lifetime. Reading the clock per
    #: turn would mean "you last spoke 0 minutes ago" forever, which is true and useless.
    _gap_seconds: float = 0.0

    # --- persistence ----------------------------------------------------------
    def load(self) -> "Memory":
        if not self.path.exists():
            return self
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            # A corrupt file must not stop her booting. Move it aside so it can be
            # looked at rather than silently overwritten with an empty one.
            log.error("memory file unreadable (%s); starting fresh", e)
            try:
                self.path.rename(self.path.with_suffix(".corrupt.json"))
            except OSError:
                pass
            return self
        self.notes = [Note(**n) for n in raw.get("notes", [])]
        self.continuity = Continuity(**raw.get("continuity", {}))
        return self

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "notes": [asdict(n) for n in self.notes],
            "continuity": asdict(self.continuity),
        }
        # Write beside and replace, so an interrupted save can't truncate the file
        # she has been keeping for months.
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    # --- session bookkeeping --------------------------------------------------
    def begin_session(self, now: float | None = None) -> None:
        now = now or time.time()
        self._gap_seconds = (now - self.continuity.last_seen) if self.continuity.last_seen else 0.0
        if not self.continuity.first_seen:
            self.continuity.first_seen = now
        self.continuity.sessions += 1
        self.continuity.last_seen = now
        self.save()

    def touch(self, now: float | None = None) -> None:
        """Mark the relationship as current. Called as turns happen, so a crash
        doesn't make it look like they never spoke."""
        self.continuity.last_seen = now or time.time()

    # --- notes ----------------------------------------------------------------
    @staticmethod
    def _key(text: str) -> str:
        return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()

    def add(self, text: str, source: str = SOURCE_AUTO) -> bool:
        """Save a note. Returns False if it was junk or already known."""
        text = " ".join(text.strip().split())
        if len(text) < 4 or len(text) > 200:
            return False

        key = self._key(text)
        for existing in self.notes:
            other = self._key(existing.text)
            # Substring either way catches the common case: the same fact learned
            # twice, once tersely and once with detail.
            if key in other or other in key:
                if len(key) > len(other):
                    existing.text, existing.created_at = text, time.time()
                if source == SOURCE_USER:
                    existing.source = SOURCE_USER
                self.save()
                return False

        self.notes.append(Note(text, time.time(), source))
        if len(self.notes) > self.max_notes:
            # Evict the oldest thing she guessed at; anything he asked for stays.
            auto = [n for n in self.notes if n.source == SOURCE_AUTO]
            self.notes.remove(min(auto, key=lambda n: n.created_at) if auto else self.notes[0])
        self.save()
        return True

    # --- editing one exact note, from the panel -------------------------------
    def update(self, note_id: str, text: str) -> bool:
        """Rewrite one note in place. Returns False if it is gone or the text is junk.

        Keeps `created_at` and `source`: correcting a wrong note does not make it newly
        learned, and it does not stop being something he told her rather than something
        she guessed at. Both are what the eviction rule and the panel's "you told her /
        she noticed" label read.
        """
        text = " ".join(text.strip().split())
        if not (4 <= len(text) <= 200):
            return False
        for note in self.notes:
            if note.id == note_id:
                note.text = text
                self.save()
                return True
        return False

    def remove(self, note_id: str) -> bool:
        """Delete exactly one note. The panel's × should never take a second one."""
        before = len(self.notes)
        self.notes = [n for n in self.notes if n.id != note_id]
        if len(self.notes) == before:
            return False
        self.save()
        return True

    def forget(self, query: str) -> list[str]:
        """Drop every note matching `query`. Returns what was removed."""
        key = self._key(query)
        if not key:
            return []
        gone = [n for n in self.notes if key in self._key(n.text) or self._key(n.text) in key]
        self.notes = [n for n in self.notes if n not in gone]
        if gone:
            self.save()
        return [n.text for n in gone]

    # --- prompt blocks --------------------------------------------------------
    def notes_block(self) -> str:
        if not self.notes:
            return ""
        lines = "\n".join(f"- {n.text}" for n in sorted(self.notes, key=lambda n: n.created_at))
        return (
            "\n\nWhat you know about him, from before:\n"
            f"{lines}\n"
            "These are your own memories. Use them when they fit, naturally, the way a "
            "person would — don't recite them and don't bring up all of them at once."
        )

    def continuity_block(self, now: float | None = None) -> str:
        now = now or time.time()
        local = time.localtime(now)
        part = (
            "the middle of the night" if local.tm_hour < 5
            else "the morning" if local.tm_hour < 12
            else "the afternoon" if local.tm_hour < 18
            else "the evening" if local.tm_hour < 22
            else "late at night"
        )
        out = [
            "\n\nRight now:",
            f"- It is {time.strftime('%A %d %B, %H:%M', local)} — {part}.",
        ]

        if self.continuity.sessions <= 1:
            out.append("- This is the first time you have ever spoken. You do not know "
                       "him yet, and pretending otherwise would be a lie.")
        else:
            out.append(f"- You have talked {self.continuity.sessions} times before.")
            if self.continuity.first_seen:
                days = (now - self.continuity.first_seen) / 86400
                out.append(f"- You have known him about {self._span(days * 86400)}.")
            if self._gap_seconds > 3600:
                out.append(f"- It has been {self._span(self._gap_seconds)} since you last "
                           f"spoke.")
        return "\n".join(out)

    @staticmethod
    def _span(seconds: float) -> str:
        if seconds < 5400:
            return f"{max(1, round(seconds / 60))} minutes"
        if seconds < 86400 * 1.5:
            return f"{max(1, round(seconds / 3600))} hours"
        if seconds < 86400 * 14:
            return f"{round(seconds / 86400)} days"
        if seconds < 86400 * 60:
            return f"{round(seconds / 86400 / 7)} weeks"
        return f"{round(seconds / 86400 / 30)} months"


#: Asked of the LLM after a turn, in the background. Written to fail closed: a 9B told
#: to "extract anything interesting" will cheerfully record that he said hello, and a
#: memory that is wrong or trivial is worse than no memory — she repeats it back days
#: later with total confidence and he has no idea where it came from.
EXTRACT_PROMPT = """You keep notes about the person in this conversation.

Read the exchange and decide whether it contains a fact worth remembering weeks from
now. Reply with ONE short line, written about them in the third person, like:
  works on a voice assistant called Aria
  has a demo on Friday
  their sister is called Mei
  takes their coffee black

Only durable things: their name, their work, their plans, what they like and dislike,
people in their life, things that happened to them. Not the weather. Not what they just
asked you to do. Nothing about you or about the assistant's own behaviour. Nothing you
are guessing at.

If there is nothing that clears that bar — which is most of the time — reply with
exactly NONE."""


def parse_extraction(reply: str) -> str | None:
    """Pull a usable note out of the extractor's reply, or None."""
    line = reply.strip().splitlines()[0].strip() if reply.strip() else ""
    line = line.strip("-•*\"' ").strip()
    if not line or line.upper().startswith("NONE") or len(line) < 4:
        return None
    # It sometimes answers in the first or second person despite being told not to.
    if re.match(r"^(i |you |we |aria )", line, re.I):
        return None
    return line[:200]
