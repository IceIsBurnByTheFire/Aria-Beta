"""Acoustic echo cancellation, so the mic can be open while Aria is talking.

The problem this solves is specific: on speakers, the mic hears Aria's own voice, the
VAD reads it as the user talking, and barge-in fires the instant she starts — she
interrupts herself into silence. Headphones dodge it entirely, which is why every
milestone before this one assumed them.

**Why not WebRTC AEC, which the roadmap called for.** It does not install on Windows.
`webrtc-audio-processing` ships a setup.py that assumes a POSIX layout and dies with a
`ValueError` during build; `speexdsp` needs SWIG and libspeexdsp headers that aren't
there. `pyaec` was the way through: a prebuilt `aec.dll` for x86_64-msvc wrapping
SpeexDSP's canceller via the Rust `aec-rs` crate. Measured on this machine, it clears
40-50 dB of echo for 0.55 ms of CPU per 32 ms frame.

**Speex is the older algorithm and that is a real trade.** AEC3 has a proper
double-talk detector; Speex has a lightweight one and distorts near-end speech that
overlaps the echo. For barge-in that is survivable: what has to work is *the VAD not
firing on echo alone*, and the moment it does fire on real speech, playback stops and
the rest of the utterance arrives clean. Transcribing the overlapping fragment is not
a goal here — cutting her off promptly is.
"""

from __future__ import annotations

import ctypes
import logging

import numpy as np

log = logging.getLogger(__name__)

#: Speex converges only over echo that lands inside the filter, so this has to cover
#: the whole round trip: output buffering, flight time across the desk, input
#: buffering. Measured on the synthetic bench, 64 ms of filter collapses to 5 dB once
#: the delay reaches 150 ms, while 200 ms holds 41 dB — and costs 0.1 ms more per
#: frame. The cost of guessing high is negligible; the cost of guessing low is that
#: this whole module silently does nothing.
DEFAULT_FILTER_MS = 200

#: How far above the measured echo floor a frame must sit before it counts as the user
#: rather than leftover echo — about 16 dB. The usable window is narrow and was found
#: by measuring both ends against a real room: at 3 the echo transient at the start of
#: a phrase trips barge-in, and by 10 the user cannot interrupt at all, even at full
#: volume. Cancellation is worst exactly where speech restarts after a pause, and that
#: transient is what this has to clear.
SPEECH_MARGIN = 6.0

#: How fast the echo-floor estimate adapts — about a 0.6 s time constant. It learns
#: only from frames it has already classified as echo, which keeps a burst of near-end
#: speech from teaching it that loud residual is normal. Tracking the *typical* echo
#: ratio matters: an earlier version followed the minimum instead, settled two orders
#: of magnitude below the real echo level, and left the gate open on everything.
_FLOOR_LEARN = 0.05

#: Per-frame decay of the reference envelope — about a 300 ms release at 32 ms frames.
#: Has to outlast the round-trip delay, or the gate reopens inside every gap between
#: Aria's words. Too long and the user waits that much longer to be heard after she
#: stops.
_ENV_RELEASE = 0.9


class EchoCanceller:
    """Speex AEC over numpy frames, with the DLL's per-frame Python overhead removed.

    `pyaec`'s own wrapper rebuilds a ctypes array from a Python list on every call and
    returns another list. At 31 frames a second that is pure waste, so this calls the
    library directly against numpy buffers.
    """

    def __init__(
        self,
        frame_samples: int,
        sample_rate: int,
        filter_ms: int = DEFAULT_FILTER_MS,
        speech_margin: float = SPEECH_MARGIN,
    ):
        import pyaec  # imported here so a missing DLL degrades to "no speaker mode"

        self._lib = pyaec.lib
        self._frame = frame_samples
        self._margin = speech_margin
        self.filter_taps = int(sample_rate * filter_ms / 1000)
        self._handle = self._lib.AecNew(frame_samples, self.filter_taps, sample_rate, True)
        if not self._handle:
            raise RuntimeError("Speex AEC refused to initialise")

        # Reused across frames: the audio thread should not be allocating.
        self._mic_i16 = np.zeros(frame_samples, dtype=np.int16)
        self._ref_i16 = np.zeros(frame_samples, dtype=np.int16)
        self._out_i16 = np.zeros(frame_samples, dtype=np.int16)
        self._p = ctypes.POINTER(ctypes.c_int16)

        self._floor = 0.5  # residual-to-reference ratio; starts pessimistic
        self._ref_env = 0.0
        self.suppressed_frames = 0

    def _ptr(self, a: np.ndarray):
        return a.ctypes.data_as(self._p)

    def process(self, mic: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Return `mic` with the echo of `reference` removed. Both float32, one frame.

        `reference` is what the speaker actually played, not what was queued — see
        `Playback.reference_for`. Handing it the queue instead lines the filter up
        against audio that has not been emitted yet, and it never converges.
        """
        np.multiply(np.clip(mic, -1.0, 1.0), 32767, out=self._mic_i16, casting="unsafe")
        np.multiply(np.clip(reference, -1.0, 1.0), 32767, out=self._ref_i16, casting="unsafe")
        self._lib.AecCancelEcho(
            self._handle,
            self._ptr(self._mic_i16),
            self._ptr(self._ref_i16),
            self._ptr(self._out_i16),
            self._frame,
        )
        residual = self._out_i16.astype(np.float32) / 32767.0
        return self._suppress(residual, reference)

    def _suppress(self, residual: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Second stage: decide whether what survived is the user or leftover echo.

        The linear filter alone is not enough and no amount of filter length fixes
        it. Measured against real Kokoro speech through a simulated room, Speex
        plateaus around 14 dB whether the tail is 200 ms or 800 ms — and Silero still
        hears speech in that residual, which is the self-interruption bug intact.

        So this compares each frame against the echo floor the canceller is currently
        achieving. Residual that tracks the reference is echo and gets zeroed; a frame
        that jumps well above the floor is the user, and passes untouched.

        **This is a real trade, not a free win.** Near-end speech quieter than the
        residual echo is indistinguishable from it, so whispering at Aria while she is
        mid-sentence at volume will not interrupt her. Speaking normally will. The
        alternative — passing everything through — is the bug this milestone exists to
        remove.
        """
        # Compare against a decaying envelope of recent reference level, not this
        # frame's. The echo reaches the mic a round trip late, so in the pause between
        # two of Aria's words the reference is silent while her previous word is still
        # arriving — an instantaneous ratio divides a loud residual by nothing, reads
        # it as the user, and opens the gate on exactly the frames it exists to close.
        ref_rms = float(np.sqrt(np.mean(reference**2)))
        self._ref_env = max(ref_rms, self._ref_env * _ENV_RELEASE)
        if self._ref_env < 1e-4:
            return residual  # nothing playing and the tail has passed

        res_rms = float(np.sqrt(np.mean(residual**2)))
        ratio = res_rms / self._ref_env

        if ratio > self._margin * self._floor:
            return residual  # the user — and deliberately not learned from
        # Echo, so this frame is evidence about the floor. Starting pessimistic means
        # the gate is shut for the first half-second of a reply and opens as the
        # estimate settles, which errs toward not interrupting herself.
        self._floor += _FLOOR_LEARN * (ratio - self._floor)
        self.suppressed_frames += 1
        return np.zeros_like(residual)

    def close(self) -> None:
        if self._handle:
            self._lib.AecDestroy(self._handle)
            self._handle = None

    def __del__(self):
        self.close()


def available() -> tuple[bool, str]:
    """Whether echo cancellation can run, and why not when it can't.

    Speaker mode is the only caller. Headphone users never load the DLL at all, which
    is deliberate: a native binary should not be a hard dependency of the default path.
    """
    try:
        import pyaec
    except ImportError as e:
        return False, f"pyaec is not installed ({e})"
    if pyaec.lib is None:
        return False, "pyaec is installed but its aec.dll failed to load"
    return True, ""
