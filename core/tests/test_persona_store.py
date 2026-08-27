"""Editing who she is, and getting her back.

The persona is the most load-bearing text in this project — nearly every behaviour that
needed defending was defended in it. Putting a text box in front of it is worth doing and
worth guarding, because the two ways it goes wrong are both silent: a cleared box leaves
a fluent assistant who is nobody in particular, and a save that quietly failed leaves the
old persona running while the panel shows the new one.
"""

from __future__ import annotations

import pytest

from aria.config import PERSONA
from aria.persona_store import MAX_LENGTH, MIN_LENGTH, PersonaStore

GOOD = "You are Aria. You live on this desktop and you are unbearably fond of him."


@pytest.fixture
def store(tmp_path) -> PersonaStore:
    return PersonaStore(tmp_path / "persona.txt", PERSONA)


def test_the_builtin_is_what_you_get_with_no_override(store):
    assert store.load() == PERSONA
    assert store.is_custom is False


def test_saving_replaces_it(store):
    ok, _ = store.save(GOOD)
    assert ok
    assert store.load() == GOOD
    assert store.is_custom is True


def test_reset_brings_back_the_builtin(store):
    store.save(GOOD)
    ok, _ = store.reset()
    assert ok
    assert store.load() == PERSONA
    assert store.is_custom is False


def test_reset_deletes_rather_than_rewrites(store):
    # The built-in must never be *copied* into the file. A copy can drift, or be
    # half-written by an interrupted save, and then the way back is gone too.
    store.save(GOOD)
    store.reset()
    assert not store.path.exists()


def test_reset_with_nothing_to_reset_is_not_an_error(store):
    ok, _ = store.reset()
    assert ok


@pytest.mark.parametrize("text", ["", "   ", "\n\n", "be nice"])
def test_an_empty_or_tiny_persona_is_refused(store, text):
    # The failure this exists for: with no character text she keeps answering, fluently
    # and as nobody, and it reads as the model having changed rather than a box having
    # been cleared.
    ok, message = store.save(text)
    assert not ok
    assert "short" in message.lower()
    assert store.load() == PERSONA, "and she is untouched"


def test_a_pasted_novel_is_refused_with_the_number(store):
    ok, message = store.save("x" * (MAX_LENGTH + 1))
    assert not ok
    assert f"{MAX_LENGTH:,}" in message, "say what the limit is, not just that there is one"
    assert store.load() == PERSONA


def test_the_limits_leave_room_for_the_real_persona():
    assert MIN_LENGTH < len(PERSONA) < MAX_LENGTH


def test_whitespace_is_trimmed_but_shape_is_kept(store):
    store.save(f"\n\n{GOOD}\n\nSecond line.\n\n")
    assert store.load() == f"{GOOD}\n\nSecond line."


def test_an_unreadable_file_falls_back_rather_than_booting_as_nobody(store):
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_bytes(b"\xff\xfe\x00 not valid utf-8 \xff")
    assert store.load() == PERSONA


def test_a_file_left_empty_by_a_failed_save_falls_back(store):
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text("", encoding="utf-8")
    assert store.load() == PERSONA
    assert store.is_custom is True, "it exists, so the panel should still offer a reset"


def test_a_save_is_atomic(store):
    # Written beside and replaced, so an interrupted save cannot leave her half-written.
    store.save(GOOD)
    store.save("You are Aria, and this is the second and much longer version of you.")
    assert "second" in store.load()
    assert not list(store.path.parent.glob("*.tmp")), "no temp file left behind"
