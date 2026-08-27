"""Speaker output built around being interruptible.

The stream stays open for the process lifetime and plays silence when idle. Opening a
stream costs tens of milliseconds, and paying that on every reply is a real chunk of
the latency budget for no reason.

It also keeps a 16 kHz copy of everything it plays, which M6's echo canceller uses as
its reference signal — see `reference_for`.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)


class _Resampler:
    """24 kHz to 16 kHz, stateful across chunks. Kokoro's rate down to the mic's.

    Exactly 2:3, so it is upsample by two, low-pass, drop every third sample. The
    filter state has to carry between chunks: reset it per chunk and every TTS
    sentence boundary puts a click in the reference the canceller then tries to find
    in the room, which is a good way to make it diverge.
    """

    L, M = 2, 3

    def __init__(self, taps: int = 97):
        # Cutoff is 16 kHz Nyquist expressed against the 48 kHz intermediate rate.
        fc = 8000 / 48000
        n = np.arange(taps) - (taps - 1) / 2
        h = 2 * fc * np.sinc(2 * fc * n) * np.hamming(taps)
        self._h = (h * self.L / h.sum()).astype(np.float32)
        self._tail = np.zeros(taps - 1, dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        up = np.zeros(len(x) * self.L, dtype=np.float32)
        up[:: self.L] = x
        y = np.convolve(np.concatenate((self._tail, up)), self._h, mode="valid")
        self._tail = up[-(len(self._h) - 1) :] if len(up) >= len(self._h) - 1 else (
            np.concatenate((self._tail, up))[-(len(self._h) - 1) :]
        )
        return y[:: self.M].astype(np.float32)


class Playback:
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        sample_rate: int = 24000,
        device: int | str | None = None,
        blocksize: int = 1024,
    ):
        self._loop = loop
        self._sr = sample_rate
        self._lock = threading.Lock()
        self._buf: deque[np.ndarray] = deque()
        self._eos = False
        self._drained = asyncio.Event()
        self._drained.set()
        self._level = 0.0
        self._samples_played = 0
        #: True while a Discord voice call is draining the buffer instead of the speaker.
        self._external = False

        # --- echo-cancellation reference (M6) ---------------------------------
        #: A 16 kHz copy of this utterance, resampled once on the writer thread
        #: rather than per frame on the audio callback. Sized for a long reply; a
        #: reply that outruns it simply stops feeding the canceller.
        self._ref_sr = 16000
        self._ref = np.zeros(self._ref_sr * 60, dtype=np.float32)
        self._ref_len = 0
        self._ref_overflowed = False
        self._resampler = _Resampler()

        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            blocksize=blocksize,
            channels=1,
            dtype="float32",
            device=device,
            callback=self._on_audio,
        )

    def start(self) -> None:
        self._stream.start()

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()

    @property
    def level(self) -> float:
        """RMS of the last block written to the device. M3 drives lip sync off this."""
        return self._level

    @property
    def seconds_played(self) -> float:
        """Audio actually emitted since the last `begin()`.

        M2 needs this: on barge-in, history must record what was *spoken*, not what was
        generated, or the conversation silently drifts.
        """
        return self._samples_played / self._sr

    def reference_for(self, n: int, delay_ms: int = 0) -> np.ndarray:
        """The `n` samples whose echo is arriving at the mic right now, at 16 kHz.

        Keyed off `_samples_played` — the device's own count — and not off what is
        sitting in the queue. Queued-but-unplayed audio is the wrong reference: the
        filter would be hunting for a signal the room has not heard yet, and it never
        converges. Returns silence when nothing is playing, which is the honest answer
        and costs the canceller nothing.

        `delay_ms` walks the read head back by the round trip, so the filter only has
        to model the room's tail rather than the whole path. It is not an optimisation:
        echo arriving later than the filter is long is not reduced at all, and the
        measured round trip on this machine is 512 ms against a 400 ms filter.
        """
        with self._lock:
            head = self._samples_played * self._ref_sr // self._sr
            head -= delay_ms * self._ref_sr // 1000
            head = min(head, self._ref_len)
            start = head - n
            if head <= 0:
                return np.zeros(n, dtype=np.float32)
            if start >= 0:
                return self._ref[start:head].copy()
            out = np.zeros(n, dtype=np.float32)
            out[-head:] = self._ref[:head]
            return out

    def begin(self) -> None:
        """Open a new utterance. Clears counters and the drained flag."""
        with self._lock:
            self._buf.clear()
            self._eos = False
            self._samples_played = 0
            self._ref_len = 0
            self._ref_overflowed = False
        self._drained.clear()

    def write(self, audio: np.ndarray) -> None:
        chunk = np.asarray(audio, dtype=np.float32).reshape(-1)
        # Resample here, on the synthesis thread. The audio callback has 21 ms to
        # fill a block and no business running a convolution.
        ref = self._resampler(chunk)
        with self._lock:
            self._buf.append(chunk)
            end = self._ref_len + len(ref)
            if end <= len(self._ref):
                self._ref[self._ref_len : end] = ref
                self._ref_len = end
            elif not self._ref_overflowed:
                self._ref_overflowed = True
                log.warning("echo reference buffer full; cancellation stops for this reply")

    def end_of_stream(self) -> None:
        """No more audio is coming; drain once the buffer empties."""
        with self._lock:
            self._eos = True
            empty = not self._buf
        if empty:
            self._drained.set()

    def flush(self) -> float:
        """Stop immediately, dropping everything queued. Returns seconds played.

        This is the barge-in path. It must be instant — no fade, no waiting for the
        current chunk. A cut-off word sounds like an interruption, which is correct.
        """
        with self._lock:
            self._buf.clear()
            self._eos = True
            played = self._samples_played / self._sr
        self._level = 0.0
        self._drained.set()
        return played

    async def wait_drained(self) -> None:
        await self._drained.wait()

    # --- routing --------------------------------------------------------------
    def route_external(self, external: bool) -> None:
        """Send audio to `pull()` instead of the sound card, or back again.

        Exactly one consumer may drain the buffer. Two would each get roughly half the
        samples — she would come out of both the speakers and the voice call, chopped
        into alternating fragments, and `seconds_played` would count the sum. So the
        local callback goes silent while something else is pulling.
        """
        with self._lock:
            self._external = external

    def pull(self, frames: int) -> np.ndarray | None:
        """Take up to `frames` samples for an external consumer.

        Returns `None` once the utterance is finished, which is how a Discord
        `AudioSource` signals end of stream. Short reads are padded with silence rather
        than short-returned: a voice packet has a fixed size, and an underrun should be
        a moment of quiet rather than a timing glitch.

        Shares every counter with the local path, deliberately. `seconds_played` is what
        decides which words go into history after barge-in, and a second, separate
        notion of "how much has actually been heard" is how that quietly starts lying.
        """
        out = np.zeros(frames, dtype=np.float32)
        pos, finished = self._take(out, frames)
        if finished and pos == 0:
            self._finish()
            return None
        self._level = float(np.sqrt(np.mean(out[:pos] ** 2))) if pos else 0.0
        if finished:
            self._finish()
        return out

    def _take(self, out: np.ndarray, frames: int) -> tuple[int, bool]:
        """Copy buffered audio into `out`. Returns (samples written, utterance done)."""
        pos = 0
        with self._lock:
            while pos < frames and self._buf:
                chunk = self._buf[0]
                take = min(frames - pos, len(chunk))
                out[pos : pos + take] = chunk[:take]
                if take == len(chunk):
                    self._buf.popleft()
                else:
                    self._buf[0] = chunk[take:]
                pos += take
            self._samples_played += pos
            return pos, self._eos and not self._buf

    def _finish(self) -> None:
        if not self._drained.is_set():
            self._loop.call_soon_threadsafe(self._drained.set)

    # --- audio thread ---------------------------------------------------------
    def _on_audio(self, outdata, frames, time_info, status) -> None:
        out = outdata[:, 0]
        if self._external:
            # Something else is draining the buffer. Emit silence and touch nothing —
            # not the counters, not the drained flag, not the level.
            out[:] = 0.0
            return

        pos, finished = self._take(out, frames)
        if pos < frames:
            out[pos:] = 0.0  # underrun or idle: silence, never garbage
        self._level = float(np.sqrt(np.mean(out[:pos] ** 2))) if pos else 0.0

        if finished:
            self._finish()
