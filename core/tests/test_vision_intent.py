"""Screen-referential intent detection.

The asymmetry matters: a false negative means Aria answers without looking and the
user rephrases. A false positive means a screenshot is taken and sent somewhere
nobody asked to send it. The patterns are tuned for the first failure, not the second.
"""

import pytest

from aria.vision.intent import capture_command, wants_screen

SCREEN_REFERENTIAL = [
    "what's on my screen",
    "can you see my screen?",
    "look at my screen for a second",
    "what am I looking at",
    "what do you see",
    "what does this error say",
    "read this traceback for me",
    "can you explain this stack trace",
    "look at this",
    "check this out",
    "what's wrong here",
    "what is this",
    "explain this code",
    "what's going on with this dialog",
    "what does this say",
    "what does it say",
]

NOT_SCREEN_REFERENTIAL = [
    "what's the capital of France",
    "hey, can you hear me okay",
    "tell me a joke",
    "explain what a compiler does",
    "I saw a screen door yesterday",
    "this is really interesting",
    "what do you think about that",
    "read me the news",
    "how does this work in general",  # "this" without a look-verb or artefact
]


@pytest.mark.parametrize("text", SCREEN_REFERENTIAL)
def test_detects_screen_questions(text):
    assert wants_screen(text), text


@pytest.mark.parametrize("text", NOT_SCREEN_REFERENTIAL)
def test_leaves_ordinary_speech_alone(text):
    assert not wants_screen(text), text


def test_detection_is_case_insensitive():
    assert wants_screen("WHAT'S ON MY SCREEN")
    assert wants_screen("What Does This Error Say")


@pytest.mark.parametrize("text", [
    "watch my screen",
    "start watching my screen",
    "you can start looking at my screen",
])
def test_enable_commands(text):
    assert capture_command(text) is True, text


@pytest.mark.parametrize("text", [
    "stop watching my screen",
    "stop looking at my screen",
    "don't watch my screen",
])
def test_disable_commands(text):
    assert capture_command(text) is False, text


def test_no_command_in_ordinary_speech():
    for text in ["what's on my screen", "hello", "read this error"]:
        assert capture_command(text) is None, text


def test_disable_wins_over_enable():
    # "stop watching my screen" contains "watch my screen" as a substring; the
    # disable check must run first or turning it off would turn it on.
    assert capture_command("stop watching my screen") is False
