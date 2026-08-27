"""Silero VAD over onnxruntime, plus the endpointing state machine.

Deliberately no torch: the ONNX model is 2 MB and runs a 32 ms frame in ~0.13 ms on
CPU, so pulling in the whole torch stack (and the Blackwell CUDA problem with it) buys
nothing.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import onnxruntime as ort


#: Samples of the *previous* frame that Silero v5 expects prepended to the current one.
#: This is not optional and not documented in the ONNX signature: the graph accepts a
#: bare 512-sample frame without complaint and then returns ~0.0 for everything,
#: including obvious speech. A VAD that silently never fires is a miserable bug, so if
#: you touch this, re-run tests/test_vad.py.
_CONTEXT_SAMPLES = {16000: 64, 8000: 32}


class SileroVAD:
    """Speech probability for one 512-sample frame at a time.

    Stateful twice over: an LSTM state *and* a rolling audio context. Frames must be
    fed in order, and `reset()` called between unrelated streams.
    """

    def __init__(self, model_path: Path, sample_rate: int = 16000):
        if sample_rate not in _CONTEXT_SAMPLES:
            raise ValueError(f"Silero supports 8k/16k, got {sample_rate}")
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1  # single frame at a time; threads only add overhead
        self._sess = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._sr = np.array(sample_rate, dtype=np.int64)
        self._context_size = _CONTEXT_SAMPLES[sample_rate]
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self._context_size), dtype=np.float32)

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self._context_size), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        window = np.concatenate(
            [self._context, frame.reshape(1, -1).astype(np.float32)], axis=1
        )
        out, self._state = self._sess.run(
            None, {"input": window, "state": self._state, "sr": self._sr}
        )
        self._context = window[:, -self._context_size :]
        return float(out[0][0])


@dataclass
class Utterance:
    """A complete user turn, with the timing needed to measure response latency."""

    audio: np.ndarray
    #: perf_counter at the moment speech actually stopped. This is the honest zero
    #: point for latency: the user stopped talking here and starts waiting here.
    speech_ended_at: float
    #: perf_counter at the moment the endpointer *decided* speech had stopped. The gap
    #: between the two is `end_silence_ms` and is unavoidable — we cannot know a turn
    #: is over until enough silence has passed.
    endpointed_at: float
    #: Frames of the source buffer this utterance's audio covers. A speculative
    #: transcription taken at N frames is still valid iff N >= this.
    frames: int = 0
    #: Seconds of this utterance that Silero actually called speech.
    #:
    #: The one number that separates a real short reply from something Whisper invented,
    #: and Whisper itself does not have it: asked to transcribe *digital silence* it
    #: returns "Thank you." at `no_speech_prob` 0.000 and a **better** `avg_logprob`
    #: than a genuine "Okay." Measured, real one-word turns run 0.35-0.58 s voiced and
    #: every non-speech case — silence, hiss, hum, a click, a breath — is 0.00 s.
    voiced_s: float = 0.0

    @property
    def duration_s(self) -> float:
        return len(self.audio) / 16000

    @property
    def endpoint_delay_ms(self) -> float:
        return (self.endpointed_at - self.speech_ended_at) * 1000


class Endpointer:
    """Turns a stream of frames into complete utterances.

    Keeps a pre-roll ring buffer so the audio handed to Whisper starts slightly before
    the trigger — without it the first phoneme is reliably clipped and short words get
    transcribed as nothing.
    """

    def __init__(
        self,
        vad: SileroVAD,
        cfg,
        frame_samples: int,
        sample_rate: int,
        on_maybe_final: Callable[[np.ndarray, int], None] | None = None,
    ):
        self._vad = vad
        self._cfg = cfg
        self._frame_samples = frame_samples
        self._sample_rate = sample_rate
        self._on_maybe_final = on_maybe_final
        frame_ms = frame_samples / sample_rate * 1000

        self._preroll: deque[np.ndarray] = deque(
            maxlen=max(1, int(cfg.preroll_ms / frame_ms))
        )
        self._end_frames_needed = max(1, int(cfg.end_silence_ms / frame_ms))
        self._speculate_frames = max(1, int(cfg.speculate_after_ms / frame_ms))
        self._max_frames = int(cfg.max_utterance_ms / frame_ms)

        self._in_speech = False
        self._speech_run = 0
        self._silence_run = 0
        self._buffer: list[np.ndarray] = []
        self._voiced = 0
        self._last_speech_at = 0.0

    @property
    def in_speech(self) -> bool:
        """True while the user is mid-utterance. M2's barge-in trigger reads this."""
        return self._in_speech

    def reset(self) -> None:
        self._vad.reset()
        self._in_speech = False
        self._speech_run = self._silence_run = self._voiced = 0
        self._buffer.clear()
        self._preroll.clear()

    @property
    def _voiced_s(self) -> float:
        return self._voiced * self._frame_samples / self._sample_rate

    def feed(self, frame: np.ndarray) -> Utterance | None:
        """Feed one frame. Returns an Utterance on the frame that completes a turn."""
        is_speech = self._vad(frame) >= self._cfg.threshold
        now = time.perf_counter()

        if not self._in_speech:
            self._preroll.append(frame)
            self._speech_run = self._speech_run + 1 if is_speech else 0
            if self._speech_run >= self._cfg.start_frames:
                self._in_speech = True
                self._silence_run = 0
                # The frames that triggered the onset were speech and count as such —
                # on a one-word turn they are a fifth of the whole utterance.
                self._voiced = self._speech_run
                self._buffer = list(self._preroll)  # pre-roll carries the onset
                self._preroll.clear()
                self._last_speech_at = now
            return None

        self._buffer.append(frame)
        if is_speech:
            self._voiced += 1
            self._silence_run = 0
            self._last_speech_at = now
        else:
            self._silence_run += 1
            # Exactly at the threshold, not past it, so this fires once per pause.
            if self._silence_run == self._speculate_frames and self._on_maybe_final:
                self._on_maybe_final(
                    np.concatenate(self._buffer), len(self._buffer), self._voiced_s
                )

        too_long = len(self._buffer) >= self._max_frames
        if self._silence_run >= self._end_frames_needed or too_long:
            return self._finish(now)
        return None

    def _finish(self, now: float) -> Utterance | None:
        audio = np.concatenate(self._buffer) if self._buffer else np.zeros(0, np.float32)
        source_frames = len(self._buffer)
        voiced_s = self._voiced_s
        self._in_speech = False
        self._speech_run = self._silence_run = self._voiced = 0
        self._buffer = []
        self._preroll.clear()

        # Drop the trailing silence that existed only so we could detect the endpoint.
        # Whisper is slower and more hallucination-prone with a long silent tail.
        trailing = self._end_frames_needed * self._frame_samples
        audio = audio[:-trailing] if trailing < len(audio) else audio[:0]

        if len(audio) / self._sample_rate * 1000 < self._cfg.min_utterance_ms:
            return None  # a cough, a chair creak, a stray consonant

        return Utterance(
            audio=audio,
            speech_ended_at=self._last_speech_at,
            endpointed_at=now,
            frames=max(0, source_frames - self._end_frames_needed),
            voiced_s=voiced_s,
        )
