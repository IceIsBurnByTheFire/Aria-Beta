"""Aria in a Discord voice call: her ears and her mouth, moved onto the network.

The whole feature is two format conversions and a routing decision. Nothing about the
conversation changes — the same VAD, the same endpointer, the same speculative
transcription, the same barge-in, the same turn handler. What changes is where the
frames come from and where the audio goes.

**The rates are kind, which is most of why this is tractable.** Discord speaks 48 kHz
stereo 16-bit both ways. Silero and Whisper want 16 kHz mono, and 48/16 is exactly 3.
Kokoro produces 24 kHz mono, and 48/24 is exactly 2. Both are integer ratios, so there
is no drift to accumulate and no fractional resampler to keep in phase.

**No echo canceller is needed, and that is not luck.** Discord never sends you your own
audio, so her voice is not in the stream she is listening to — the entire problem M6
exists to solve does not arise here. Barge-in works out of the box.

**Exclusive, not additive.** While she is in a call the sound card is closed and the
speakers are silent. He is at one desk with one headset: listening to both ears would
hear every sentence twice and endpoint on whichever copy arrived first, and speaking to
both would put her voice in the room and in the call a beat apart.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

#: Discord's wire format, both directions. Not configurable — it is what the gateway
#: speaks.
DISCORD_SR = 48000
DISCORD_CHANNELS = 2
#: One Opus frame. discord.py's player asks for exactly this every 20 ms.
FRAME_MS = 20
FRAME_SAMPLES_48K = DISCORD_SR * FRAME_MS // 1000        # 960
FRAME_BYTES = FRAME_SAMPLES_48K * DISCORD_CHANNELS * 2   # 3840, s16le stereo

#: Kokoro's rate, and how much of it makes one Discord packet. Exactly half of 960,
#: because 48/24 is 2 — the conversion is an interpolation, not a resampler.
KOKORO_SR = 24000
FRAME_SAMPLES_24K = KOKORO_SR * FRAME_MS // 1000         # 480


class Decimator:
    """48 kHz down to 16 kHz: low-pass, then keep every third sample.

    The filter is the point. Dropping two samples in three without it folds everything
    above 8 kHz back down into the speech band, and the result is not obviously broken —
    it is a slight harshness that makes Whisper worse in a way nobody attributes to
    resampling. State carries between chunks for the same reason it does in
    `playback._Resampler`: reset per chunk and every packet boundary becomes a click.
    """

    M = 3

    def __init__(self, taps: int = 65):
        fc = 7800 / DISCORD_SR  # just under the 8 kHz Nyquist of the target rate
        n = np.arange(taps) - (taps - 1) / 2
        h = 2 * fc * np.sinc(2 * fc * n) * np.hamming(taps)
        self._h = (h / h.sum()).astype(np.float32)
        self._tail = np.zeros(taps - 1, dtype=np.float32)
        #: Which sample of each group of three to keep, carried across chunks. A packet
        #: is 960 samples, which divides by 3 — but a dropped or partial packet would
        #: otherwise shift the phase and put a click in the middle of a word.
        self._phase = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y = np.convolve(np.concatenate((self._tail, x)), self._h, mode="valid")
        self._tail = np.concatenate((self._tail, x))[-(len(self._h) - 1):]
        out = y[self._phase :: self.M]
        self._phase = (self._phase - len(y)) % self.M
        return out.astype(np.float32)


def to_discord_frame(mono24: np.ndarray) -> bytes:
    """24 kHz mono float32 -> one 20 ms 48 kHz stereo s16le packet.

    Upsampling is exactly 2x, done by linear interpolation between neighbours rather
    than by repeating each sample. Repetition is a zero-order hold, which is a crude
    low-pass with an audible edge on sibilants; interpolation costs one add and sounds
    clean. Kokoro carries almost nothing above 8 kHz anyway, so no anti-image filter is
    needed on top.
    """
    x = np.asarray(mono24, dtype=np.float32).reshape(-1)
    up = np.empty(len(x) * 2, dtype=np.float32)
    up[0::2] = x
    up[1:-1:2] = (x[:-1] + x[1:]) * 0.5
    if len(up) > 1:
        up[-1] = x[-1]

    np.clip(up, -1.0, 1.0, out=up)
    pcm = (up * 32767.0).astype("<i2")
    # Interleave to stereo by duplicating: she is one voice, not a stage.
    return np.repeat(pcm, DISCORD_CHANNELS).tobytes()


def dave_decrypt(session, user_id: int, payload: bytes) -> bytes | None:
    """Peel Discord's end-to-end layer off one audio packet.

    DAVE is Discord's end-to-end encryption for calls. `discord-ext-voice-recv` has no
    support for it and hands the still-encrypted payload straight to Opus, which reports
    `corrupted stream` — accurately, since that is what it is looking at. discord.py
    does have the session (via `davey`); it simply is not wired to the receive path,
    because the receive path is a third-party extension.

    **Turning DAVE off is not an option, which cost a detour worth recording.** The
    obvious fix is to stop advertising support so the call downgrades — and Discord
    answers the voice handshake with close code **4017, "This channel requires a client
    supporting E2EE via the DAVE Protocol"**. A channel can mandate it. So the choice is
    to implement it or to have no voice at all.

    Returns None when the packet cannot be decrypted, which the caller treats as a
    dropped frame rather than an error.
    """
    if session is None:
        return payload  # no session: an ordinary, transport-encrypted call
    ready = session.ready
    if callable(ready):
        ready = ready()
    if not ready:
        # Mid-handshake, or every participant is in passthrough. Either way the payload
        # is not E2EE-wrapped yet, so it is already what Opus wants.
        return payload
    try:
        import davey

        return session.decrypt(user_id, davey.MediaType.audio, payload)
    except Exception as e:  # noqa: BLE001 — one frame, never the call
        log.debug("DAVE decrypt failed for user %s: %s", user_id, e)
        return None


class Ears:
    """Turns received voice packets into frames the existing VAD thread understands.

    Called from discord.py's receive thread, once per packet per speaker. Does the
    conversion and hands off; anything slower than the packet rate here backs up the
    socket.
    """

    def __init__(self, frame_samples: int, submit: Callable[[np.ndarray], None]):
        self._frame_samples = frame_samples
        self._submit = submit
        self._decimate = Decimator()
        self._pending = np.zeros(0, dtype=np.float32)
        self.packets = 0
        #: Frames actually handed to the VAD. Distinct from `packets` on purpose: audio
        #: can decode perfectly and still never reach the endpointer if the re-blocking
        #: is wrong, and the two counts are what tell those apart.
        self.frames = 0

    def feed(self, pcm: bytes) -> None:
        """One 48 kHz stereo s16le packet from Discord."""
        if not pcm:
            return
        self.packets += 1
        stereo = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        # Downmix before decimating: half the filtering work, and the two channels of a
        # Discord voice packet are the same signal anyway.
        mono48 = stereo.reshape(-1, DISCORD_CHANNELS).mean(axis=1)
        self._pending = np.concatenate((self._pending, self._decimate(mono48)))

        # Silero is strict about frame size, so re-block to exactly what it wants
        # rather than passing whatever a packet happened to produce.
        n = self._frame_samples
        while len(self._pending) >= n:
            self._submit(self._pending[:n].copy())
            self._pending = self._pending[n:]
            self.frames += 1

    def reset(self) -> None:
        self._pending = np.zeros(0, dtype=np.float32)
