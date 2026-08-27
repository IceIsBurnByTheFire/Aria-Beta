"""The wake gate: what she answers, what she lets pass, and what reaches the model.

Two failures matter and they pull in opposite directions. Missing her name means she
ignores you and you say it again — annoying. Answering the room means she talks over a
podcast or someone else's conversation, which is the thing the gate exists to stop.
Whisper mishears "Aria" often enough that the fuzzy pass has to be generous, and that
generosity is exactly what risks the second failure — so both edges are pinned here.
"""

from __future__ import annotations

import pytest

from aria.wake import WakeWord


@pytest.fixture
def wake() -> WakeWord:
    return WakeWord(word="aria", window_s=30.0)


# --- hearing her name ------------------------------------------------------
@pytest.mark.parametrize("said", [
    "Aria, what's the weather",
    "hey aria are you there",
    "ARIA",
    "so aria, what do you think",
    # What Whisper actually returns for her name, which is the whole reason the
    # match is fuzzy rather than exact.
    "area, what's the weather",
    "arya what time is it",
    "ariya can you hear me",
])
def test_hears_her_name(wake, said):
    assert wake.hears_name(said), said


@pytest.mark.parametrize("said", [
    "what's the weather",
    "I was reading about Maria yesterday",
    "are you aware of the time",
    "the aerial is broken",
    "tell me about Syria",
])
def test_does_not_hear_her_name_in_ordinary_speech(wake, said):
    assert not wake.hears_name(said), said


# --- what reaches the model ------------------------------------------------
@pytest.mark.parametrize("said,expected", [
    ("Aria, what's the weather", "what's the weather"),
    ("hey aria what's the weather", "what's the weather"),
    ("what's the weather, aria", "what's the weather,"),
])
def test_her_name_is_stripped_before_the_model_sees_it(wake, said, expected):
    # Left in, every single turn arrives with her being addressed by name and she
    # starts treating it as remarkable.
    assert wake.strip_name(said).rstrip() == expected.rstrip()


def test_being_called_by_name_alone_still_says_something(wake):
    # "Aria?" strips to nothing; handing the model an empty string is worse than
    # handing it the word she was called.
    answer, text = wake.should_answer("Aria?")
    assert answer and text.strip()


# --- the window ------------------------------------------------------------
def test_ignores_the_room_until_named(wake):
    answer, _ = wake.should_answer("so then he said the build was broken", now=100.0)
    assert not answer


def test_follow_ups_need_no_name(wake):
    assert wake.should_answer("aria, what's the weather", now=100.0)[0]
    assert wake.should_answer("and tomorrow?", now=105.0)[0]


def test_the_window_closes(wake):
    assert wake.should_answer("aria, what's the weather", now=100.0)[0]
    assert not wake.should_answer("unrelated chatter", now=131.0)[0]


def test_each_exchange_extends_the_window(wake):
    wake.should_answer("aria, hello", now=100.0)
    for t in (120.0, 145.0, 170.0):  # each within 30s of the last, never of the first
        assert wake.should_answer("and then?", now=t)[0], t


def test_her_name_wakes_her_even_when_the_window_lapsed(wake):
    wake.should_answer("aria, hello", now=100.0)
    assert wake.should_answer("aria, you there?", now=500.0)[0]
