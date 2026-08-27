"""Who she is, in a file he can edit — and the way back if he breaks it.

`config.PERSONA` is the built-in, and it stays in code where it cannot be lost. This
module lays an optional override on top: `core/data/persona.txt`, next to `memory.json`,
gitignored for the same reason. What she remembers and who she is are both his, and a
file he cannot find is one he cannot correct.

Three things shape it.

**The built-in is never overwritten.** Reset deletes the override rather than restoring a
copy, so the default cannot drift or be half-saved. Whatever happens to the file, closing
Aria and reopening her with no override gives back exactly the persona this project was
tuned against.

**An empty persona is refused, not saved.** It is the one edit that silently produces a
completely different assistant: with no character text she keeps answering, fluently and
as nobody, and the failure looks like the model having changed rather than a text box
having been cleared.

**Only the persona is editable, not the system prompt.** The prompt she actually receives
is assembled per turn — persona, then screen rules, the character's emotion vocabulary,
her notes, the clock, and a one-turn reminder if she just asked a question. Those are
machinery, and several of them are load-bearing for the cache. Editing the whole assembly
would mean re-deriving all of it on every save. The panel shows the assembled result
read-only instead, which is what someone actually wants when they ask "why did she say
that".
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

#: Short enough to catch an accidental clear or a stray keystroke, long enough that a
#: deliberately terse persona still works. Two words is a real edit; two characters is a
#: mistake nobody meant to save.
MIN_LENGTH = 20
#: Everything here is prefix-cached by Ollama and re-sent on every turn, so an enormous
#: persona is paid for on each one. Past this it is almost certainly a paste accident.
MAX_LENGTH = 20_000


class PersonaStore:
    def __init__(self, path: Path, builtin: str):
        self.path = path
        self.builtin = builtin

    @property
    def is_custom(self) -> bool:
        return self.path.exists()

    def load(self) -> str:
        """The persona in force. Falls back to the built-in on anything unreadable."""
        if not self.path.exists():
            return self.builtin
        try:
            text = self.path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError) as e:
            # UnicodeDecodeError is not an OSError, and catching only the latter meant a
            # file saved as UTF-16 or ANSI by another editor took core down on startup —
            # a traceback before she ever loads, from the one file a person is most
            # likely to open in something else.
            log.error("cannot read %s (%s); using the built-in persona", self.path, e)
            return self.builtin
        if len(text) < MIN_LENGTH:
            # A file that exists but is empty is the state left behind by a failed save
            # or a cleared editor. Booting as nobody would be a strange thing to do
            # about it.
            log.warning("%s is too short to be a persona; using the built-in", self.path)
            return self.builtin
        return text

    def save(self, text: str) -> tuple[bool, str]:
        """Write a new persona. Returns (ok, message-for-the-user)."""
        text = text.strip()
        if len(text) < MIN_LENGTH:
            return False, (
                "That's too short to be a persona. Nothing was saved — she's still "
                "herself."
            )
        if len(text) > MAX_LENGTH:
            return False, (
                f"That's {len(text):,} characters. She re-reads all of it every turn, "
                f"so the limit is {MAX_LENGTH:,}. Nothing was saved."
            )
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Beside and replace, so an interrupted save cannot leave her half-written.
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(self.path)
        except OSError as e:
            log.error("could not save persona: %s", e)
            return False, f"Couldn't write the file: {e}"
        return True, "Saved. It applies to her next reply."

    def reset(self) -> tuple[bool, str]:
        """Drop the override. The built-in comes back because it never left."""
        if not self.path.exists():
            return True, "Already the built-in persona."
        try:
            self.path.unlink()
        except OSError as e:
            return False, f"Couldn't remove the file: {e}"
        return True, "Back to the built-in persona."
