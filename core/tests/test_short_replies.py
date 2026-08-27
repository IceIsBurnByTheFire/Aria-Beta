"""Short answers, and the filter that used to eat them.

"Okay." "Thanks." "Bye." "So?" — six ordinary things to say to an assistant, all
transcribed perfectly by Whisper and then thrown away before she ever saw them. She did
nothing, which from the outside is indistinguishable from not having heard.

The filter was not wrong to exist. Whisper invents exactly these phrases from silence,
and its own confidence signals do not help: asked to transcribe **digital silence** it
returns "Thank you." at `no_speech_prob` 0.000 with an `avg_logprob` of -0.28, which is
*better* than a genuine "Okay." at -0.68. Both signals point the wrong way.

What separates them is whether anyone was talking, which only the VAD knows. Measured:

    real one-word turns                        0.35 - 0.58 s voiced
    silence, hiss, hum, a click, a breath      0.00 s voiced
"""

from __future__ import annotations

import numpy as np
import pytest

from aria.audio.vad import Endpointer, SileroVAD
from aria.config import Config
from aria.stt.whisper import _looks_hallucinated

MIN = 0.25


def hallucinated(text: str, voiced_s: float | None) -> bool:
    return _looks_hallucinated(text, voiced_s, MIN)


@pytest.mark.parametrize("word", ["Okay.", "Thanks.", "Thank you.", "Bye.", "So?", "Oh."])
def test_a_short_reply_backed_by_speech_reaches_her(word):
    assert not hallucinated(word, voiced_s=0.45), f"{word!r} is a real thing to say"


@pytest.mark.parametrize("word", ["Okay.", "Thank you.", "you", "Thanks for watching."])
def test_the_same_words_backed_by_nothing_are_still_dropped(word):
    assert hallucinated(word, voiced_s=0.0)


def test_an_unmeasured_caller_gets_the_old_blunt_behaviour():
    # A path with no VAD behind it cannot tell the two apart, and guessing "real" there
    # would let silence hallucinations back in through the side door.
    assert hallucinated("Okay.", voiced_s=None)


def test_ordinary_speech_is_never_touched_by_any_of_this():
    for text in ("What's the weather like?", "Okay, do that one.", "Thanks, that helps."):
        assert not hallucinated(text, voiced_s=0.0), f"{text!r} is not on the list"


def test_empty_is_always_discarded():
    assert hallucinated("", voiced_s=9.9)
    assert hallucinated("...", voiced_s=9.9)


def test_the_threshold_has_margin_on_both_sides():
    """The measured gap is 0.00 s against 0.35 s, and the threshold sits between them.

    Pinned because a value that drifts up eats real words again and one that drifts
    down lets a click through, and neither shows up as a test failure anywhere else.
    """
    assert Config().stt.min_voiced_ms == 250
    assert hallucinated("Okay.", 0.10), "a transient should not count as speech"
    assert not hallucinated("Okay.", 0.35), "the shortest real word measured"


# --- the measurement itself ---------------------------------------------------
def voiced_seconds(audio: np.ndarray) -> float:
    cfg = Config()
    ep = Endpointer(
        SileroVAD(cfg.vad.model_path, cfg.audio.sample_rate), cfg.vad,
        frame_samples=cfg.audio.frame_samples, sample_rate=cfg.audio.sample_rate,
    )
    n = cfg.audio.frame_samples
    for i in range(0, len(audio) - n, n):
        if (utt := ep.feed(audio[i:i + n])) is not None:
            return utt.voiced_s
    return ep._voiced_s


def test_silence_measures_as_no_speech():
    assert voiced_seconds(np.zeros(32000, dtype=np.float32)) == 0.0


def test_hiss_measures_as_no_speech():
    rng = np.random.default_rng(0)
    assert voiced_seconds(rng.normal(0, 0.01, 32000).astype(np.float32)) == 0.0


def test_a_click_measures_as_no_speech():
    click = np.zeros(32000, dtype=np.float32)
    click[8000:8400] = 0.3
    assert voiced_seconds(click) == 0.0


# --- one speculation at a time ------------------------------------------------
class FakeTask:
    def __init__(self, done: bool):
        self._done = done
        self.cancelled = False

    def done(self) -> bool:
        return self._done

    def cancel(self) -> None:
        self.cancelled = True


def speculating_loop(existing):
    """A `VoiceLoop` with just enough of it to run `speculate`."""
    import asyncio

    from aria.loop import VoiceLoop
    from aria.state import State, StateMachine

    ln = VoiceLoop.__new__(VoiceLoop)
    ln.state = StateMachine()
    ln.state.to(State.LISTENING)
    ln._spec_task = existing
    ln._spec_frames = 7
    ln.stt = type("S", (), {"transcribe": staticmethod(lambda *a: None)})()
    started = []
    ln._loop = None

    def fake_create_task(coro):
        coro.close()
        started.append(True)
        return FakeTask(done=False)

    return ln, started, fake_create_task, asyncio


def run_speculate(existing, monkeypatch):
    import asyncio

    ln, started, fake_create_task, _ = speculating_loop(existing)
    monkeypatch.setattr(asyncio, "create_task", fake_create_task)
    ln.speculate(np.zeros(512, dtype=np.float32), frames=9, voiced_s=0.4)
    return started


def test_a_second_pause_does_not_start_a_competing_transcription(monkeypatch):
    """Measured: 331 ms alone, 626 ms with one abandoned pass still on the GPU.

    `task.cancel()` cannot stop a thread, so before this a hesitant sentence with three
    short pauses had three transcriptions competing — and the mechanism that exists to
    make a turn *faster* was what made it slow.
    """
    in_flight = FakeTask(done=False)
    assert run_speculate(in_flight, monkeypatch) == []
    assert not in_flight.cancelled, "the running one is left alone to finish"


def test_a_finished_one_is_replaced_as_before(monkeypatch):
    assert run_speculate(FakeTask(done=True), monkeypatch) == [True]


def test_the_first_pause_of_a_turn_still_speculates(monkeypatch):
    assert run_speculate(None, monkeypatch) == [True]
