"""VAD and endpointing.

The regression these guard is specific and nasty: Silero v5's ONNX graph accepts a bare
512-sample frame without error and returns ~0.0 for everything, including obvious
speech, unless 64 samples of context from the previous frame are prepended. The symptom
is a VAD that simply never fires — no exception, no warning.
"""

from __future__ import annotations

import numpy as np
import pytest

from aria.audio.vad import Endpointer, SileroVAD
from aria.config import Config

cfg = Config()
pytestmark = pytest.mark.skipif(
    not cfg.vad.model_path.exists() or not cfg.tts.model_path.exists(),
    reason="model files not downloaded (run: uv run python -m aria.setup_models)",
)

SR = 16000
FRAME = 512


@pytest.fixture(scope="module")
def speech() -> np.ndarray:
    """Real synthesised speech at 16 kHz. Tones will not do — Silero is trained to
    reject non-speech, so a sine wave passes a broken VAD just as well as a fixed one.

    The voice is pinned rather than taken from the config. These assertions are about
    Silero's thresholds against a fixed waveform, so leaving it on Aria's own voice
    means changing how she sounds silently retunes the VAD tests — which is exactly
    what happened when she moved to af_bella.
    """
    from dataclasses import replace

    from aria.tts.kokoro_backend import KokoroTTS

    fixed = replace(cfg.tts, voice="af_heart", emotion_voice=False)
    audio = KokoroTTS(fixed).synth("The quick brown fox jumps over the lazy dog.")
    n = int(len(audio) * SR / cfg.tts.sample_rate)
    return np.interp(
        np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio
    ).astype(np.float32)


def probabilities(audio: np.ndarray) -> np.ndarray:
    vad = SileroVAD(cfg.vad.model_path, SR)
    return np.array(
        [vad(audio[i : i + FRAME]) for i in range(0, len(audio) - FRAME + 1, FRAME)]
    )


def silence(ms: int) -> np.ndarray:
    return np.zeros(int(SR * ms / 1000), dtype=np.float32)


def test_speech_is_detected(speech):
    probs = probabilities(speech)
    assert probs.max() > 0.9, f"VAD never fired on speech (max {probs.max():.3f})"
    assert (probs >= 0.5).sum() >= 10


def test_silence_is_not_detected():
    probs = probabilities(silence(2000))
    assert probs.max() < 0.2, f"VAD fired on silence (max {probs.max():.3f})"


def test_endpointer_emits_one_utterance(speech):
    audio = np.concatenate([silence(400), speech, silence(1200)])
    ep = _endpointer()
    got = [u for i in range(0, len(audio) - FRAME + 1, FRAME)
           if (u := ep.feed(audio[i : i + FRAME])) is not None]
    assert len(got) == 1
    # Pre-roll adds a little at the front; the trailing silence must be trimmed off.
    assert len(speech) / SR <= got[0].duration_s <= len(speech) / SR + 0.6


def test_endpointer_handles_back_to_back_turns(speech):
    audio = np.concatenate([silence(400), speech, silence(1200), speech, silence(1200)])
    ep = _endpointer()
    got = [u for i in range(0, len(audio) - FRAME + 1, FRAME)
           if (u := ep.feed(audio[i : i + FRAME])) is not None]
    assert len(got) == 2, "state did not reset cleanly between turns"


def test_short_noise_is_rejected(speech):
    blip = np.concatenate([silence(400), speech[: int(SR * 0.1)], silence(1200)])
    ep = _endpointer()
    got = [u for i in range(0, len(blip) - FRAME + 1, FRAME)
           if (u := ep.feed(blip[i : i + FRAME])) is not None]
    assert got == [], "a 100ms blip should not count as a turn"


def _endpointer() -> Endpointer:
    return Endpointer(SileroVAD(cfg.vad.model_path, SR), cfg.vad, FRAME, SR)
