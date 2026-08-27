"""Shared helpers for driving the pipeline from synthesised speech instead of a mic."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aria.audio.vad import Endpointer, SileroVAD, Utterance  # noqa: E402
from aria.config import Config  # noqa: E402


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Linear resample. No anti-alias filter, which is fine here only because Kokoro
    output carries almost no energy above 8 kHz."""
    n_out = int(len(x) * sr_out / sr_in)
    return np.interp(
        np.linspace(0, len(x) - 1, n_out), np.arange(len(x)), x
    ).astype(np.float32)


def with_silence(speech: np.ndarray, sr: int, lead_ms=400, tail_ms=1200) -> np.ndarray:
    lead = np.zeros(int(sr * lead_ms / 1000), dtype=np.float32)
    tail = np.zeros(int(sr * tail_ms / 1000), dtype=np.float32)
    return np.concatenate([lead, speech, tail])


def endpoint(cfg: Config, audio: np.ndarray, on_maybe_final=None) -> Utterance | None:
    """Feed audio through the real endpointer in real-time-sized frames."""
    ep = Endpointer(
        SileroVAD(cfg.vad.model_path, cfg.audio.sample_rate),
        cfg.vad,
        frame_samples=cfg.audio.frame_samples,
        sample_rate=cfg.audio.sample_rate,
        on_maybe_final=on_maybe_final,
    )
    n = cfg.audio.frame_samples
    for i in range(0, len(audio) - n + 1, n):
        if (utt := ep.feed(audio[i : i + n])) is not None:
            return utt
    return None


def speak_into_pipeline(cfg: Config, synth, question: str, on_maybe_final=None):
    """Synthesise `question` with `synth` and run it through the real endpointer.

    Takes a callable rather than the TTS object so a test can pass the *unspied*
    synth and keep its own instrumentation out of the recording.

    Returns (utterance, source_speech_seconds).
    """
    spoken = synth(question)
    audio = with_silence(
        resample(spoken, cfg.tts.sample_rate, cfg.audio.sample_rate), cfg.audio.sample_rate
    )
    utt = endpoint(cfg, audio, on_maybe_final=on_maybe_final)
    return utt, len(spoken) / cfg.tts.sample_rate


def rebase(cfg: Config, utt: Utterance) -> None:
    """Replace the endpointer's wall-clock stamps with the hold live use would incur.

    We fed it seconds of audio in milliseconds, so its measured hold is meaningless.
    Reporting it as-is would flatter every latency number by ~600 ms.
    """
    now = time.perf_counter()
    utt.endpointed_at = now
    utt.speech_ended_at = now - cfg.vad.end_silence_ms / 1000
