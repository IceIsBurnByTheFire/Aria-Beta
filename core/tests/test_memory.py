"""Memory: what gets kept, what gets rejected, and what survives a restart.

The failure that matters here isn't losing a note — it's keeping a wrong one. She
repeats it back weeks later with total confidence and he has no idea where it came
from, so the extractor is written to fail closed and these guard that.
"""

from __future__ import annotations

import time

import pytest

from aria.memory import SOURCE_AUTO, SOURCE_USER, Memory, Note, parse_extraction
from aria.memory_intent import as_note, memory_command


@pytest.fixture
def mem(tmp_path) -> Memory:
    return Memory(tmp_path / "memory.json")


# --- what the extractor accepts -------------------------------------------
@pytest.mark.parametrize("reply", [
    "NONE",
    "none",
    "NONE.",
    "",
    "   ",
    "I think he likes coffee",      # first person — it's talking as her
    "You have a demo on Friday",    # second person
    "Aria should remember this",    # about her, not him
    "abc",                          # too short to be a fact
])
def test_rejects_junk(reply):
    assert parse_extraction(reply) is None


@pytest.mark.parametrize("reply,expected", [
    ("has a demo on Friday", "has a demo on Friday"),
    ("- takes his coffee black", "takes his coffee black"),
    ('"his sister is called Mei"', "his sister is called Mei"),
    ("works on a voice assistant\nand other things", "works on a voice assistant"),
])
def test_accepts_facts(reply, expected):
    assert parse_extraction(reply) == expected


# --- dictated notes keep his words -----------------------------------------
@pytest.mark.parametrize("said,quoted", [
    ("I take my coffee black", "I take my coffee black"),
    ("that my sister is called Mei", "my sister is called Mei"),
    ("I'm working on a voice assistant.", "I'm working on a voice assistant"),
])
def test_as_note_quotes_him_verbatim(said, quoted):
    # Rewriting the pronouns instead produced "he take his coffee black" — fixing the
    # agreement needs a verb table, and quoting is both grammar-proof and truer to
    # what he actually said.
    note = as_note(said)
    assert f'"{quoted}"' in note
    assert note.startswith("he asked you to remember")


# --- spoken commands -------------------------------------------------------
def test_remember_and_forget_are_recognised():
    assert memory_command("remember that I take my coffee black") == (
        "remember", "I take my coffee black")
    assert memory_command("forget about the demo")[0] == "forget"
    assert memory_command("what do you remember about me")[0] == "recall"


@pytest.mark.parametrize("text", [
    "I can't remember where I put it",   # not addressed to her
    "remember to call mum",              # a reminder; there are no timers here
    "hello",
])
def test_leaves_ordinary_speech_alone(text):
    assert memory_command(text) is None


# --- storage ---------------------------------------------------------------
def test_notes_survive_a_restart(mem, tmp_path):
    mem.add("has a demo on Friday", SOURCE_USER)
    assert [n.text for n in Memory(tmp_path / "memory.json").load().notes] == [
        "has a demo on Friday"]


def test_duplicates_collapse_and_keep_the_longer_wording(mem):
    assert mem.add("has a demo")
    assert not mem.add("has a demo on Friday")  # same fact, more detail
    assert [n.text for n in mem.notes] == ["has a demo on Friday"]


def test_forget_removes_by_substring(mem):
    mem.add("has a demo on Friday")
    mem.add("takes his coffee black")
    assert mem.forget("demo") == ["has a demo on Friday"]
    assert [n.text for n in mem.notes] == ["takes his coffee black"]


def test_eviction_spares_what_he_asked_for(mem):
    mem.max_notes = 3
    mem.add("his sister is called Mei", SOURCE_USER)
    for i in range(5):
        mem.add(f"auto fact number {i}", SOURCE_AUTO)
    kept = [n.text for n in mem.notes]
    assert "his sister is called Mei" in kept, "a note he asked for was evicted"
    assert len(kept) == 3


def test_corrupt_file_is_set_aside_not_lost(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text("{ this is not json", encoding="utf-8")
    assert Memory(path).load().notes == []
    assert path.with_suffix(".corrupt.json").exists(), "the old file must be recoverable"


# --- continuity ------------------------------------------------------------
def test_first_session_says_they_have_never_met(mem):
    mem.begin_session()
    assert "first time" in mem.continuity_block()


def test_later_sessions_report_the_gap(mem):
    now = time.time()
    mem.begin_session(now - 86400 * 3)
    mem.begin_session(now)
    block = mem.continuity_block(now)
    assert "3 days" in block and "first time" not in block


# --- editing one exact note, from the panel --------------------------------
# `forget` matches loosely so it works said out loud. A button next to one specific
# line needs the opposite, so the panel addresses notes by id.
def test_every_note_gets_an_id(mem):
    mem.add("he has a cat called Widget")
    assert mem.notes[0].id, "without one the panel can only address notes by text"


def test_ids_are_unique(mem):
    for i in range(5):
        mem.add(f"note number {i} about him")
    assert len({n.id for n in mem.notes}) == len(mem.notes)


def test_notes_written_before_ids_existed_still_load(tmp_path):
    # An old memory.json has no `id` field. Booting must not need a migration step.
    path = tmp_path / "memory.json"
    path.write_text(
        '{"notes": [{"text": "takes his coffee black", "created_at": 1.0,'
        ' "source": "you"}], "continuity": {}}',
        encoding="utf-8",
    )
    loaded = Memory(path).load()
    assert loaded.notes[0].id, "an id should be minted on load"
    assert loaded.notes[0].text == "takes his coffee black"


def test_update_rewrites_exactly_one_note(mem):
    mem.add("he has a cat")
    mem.add("his cat is called Widget")
    target = mem.notes[0].id
    assert mem.update(target, "he has two cats")
    assert {n.text for n in mem.notes} == {"he has two cats", "his cat is called Widget"}


def test_update_keeps_when_and_who_said_it(mem):
    mem.add("he takes his coffee black", source="you")
    note = mem.notes[0]
    when, source = note.created_at, note.source
    mem.update(note.id, "he takes his coffee white")
    # Correcting a note does not make it newly learned, and does not stop it being
    # something he said. Both are what eviction and the panel's label read.
    assert mem.notes[0].created_at == when
    assert mem.notes[0].source == source


@pytest.mark.parametrize("bad", ["", "  ", "no", "x" * 201])
def test_update_refuses_junk(mem, bad):
    mem.add("he has a cat called Widget")
    assert not mem.update(mem.notes[0].id, bad)
    assert mem.notes[0].text == "he has a cat called Widget"


def test_update_of_a_vanished_note_says_so(mem):
    assert not mem.update("nosuchid", "something perfectly reasonable")


def _overlapping(mem) -> Memory:
    """Two notes where one's text contains the other's.

    Built directly rather than through `add`, which dedupes exactly this shape — but
    the state still arises, from an edit that narrows one note or from the file being
    opened by hand. It is the case where addressing a note by text stops being safe.
    """
    mem.notes.append(Note("cat", time.time()))
    mem.notes.append(Note("he has a cat called Widget", time.time()))
    return mem


def test_remove_takes_exactly_the_note_it_was_given(mem):
    _overlapping(mem)
    widget = next(n.id for n in mem.notes if "Widget" in n.text)
    assert mem.remove(widget)
    assert [n.text for n in mem.notes] == ["cat"]


def test_forget_would_have_taken_both(mem):
    # Not a bug in `forget`: said out loud, "forget about the cat" should catch the note
    # however it happens to be worded. It is the wrong rule for a button next to one
    # specific line, which is the whole reason the panel works by id.
    _overlapping(mem)
    assert len(mem.forget("he has a cat called Widget")) == 2


def test_remove_of_a_vanished_note_says_so(mem):
    assert not mem.remove("nosuchid")


def test_edits_survive_a_reload(tmp_path):
    mem = Memory(tmp_path / "memory.json")
    mem.add("he has a cat")
    mem.update(mem.notes[0].id, "he has a dog")
    assert Memory(tmp_path / "memory.json").load().notes[0].text == "he has a dog"
