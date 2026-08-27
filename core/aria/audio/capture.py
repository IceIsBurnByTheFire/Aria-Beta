"""Microphone capture and the VAD worker that turns it into utterances."""

from __future__ import annotations

import asyncio
import queue
import threading
from typing import Callable

import numpy as np
import sounddevice as sd

from ..config import Config
from .aec import EchoCanceller
from .playback import Playback
from .vad import Endpointer, SileroVAD, Utterance


class VoiceListener:
    """Captures the mic and posts complete utterances onto an asyncio queue.

    Capture and VAD run on their own threads, never on the event loop. Real-time audio
    and asyncio do not mix: one slow coroutine or GC pause on the loop would drop
    frames, and dropped frames mean clipped words.
    """

    def __init__(
        self,
        cfg: Config,
        loop: asyncio.AbstractEventLoop,
        utterances: asyncio.Queue[Utterance],
        on_speech_start: Callable[[], None] | None = None,
        on_maybe_final: Callable[[np.ndarray, int, float], None] | None = None,
        playback: "Playback | None" = None,
        aec: "EchoCanceller | None" = None,
    ):
        self._cfg = cfg
        self._loop = loop
        self._out = utterances
        self._on_speech_start = on_speech_start
        self._on_maybe_final = on_maybe_final

        self._endpointer = Endpointer(
            SileroVAD(cfg.vad.model_path, cfg.audio.sample_rate),
            cfg.vad,
            frame_samples=cfg.audio.frame_samples,
            sample_rate=cfg.audio.sample_rate,
            on_maybe_final=self._bridge_maybe_final,
        )

        # (mic frame, echo reference) — the reference is sampled in the audio callback
        # rather than on the VAD thread, so queue latency cannot smear the alignment
        # the canceller depends on.
        self._frames: queue.Queue[tuple[np.ndarray, np.ndarray | None] | None] = queue.Queue(
            maxsize=256
        )
        self._stream: sd.InputStream | None = None
        self._worker: threading.Thread | None = None
        self._muted = threading.Event()
        self._dropped = 0
        self._processed = 0

        self._playback = playback
        self._aec = aec
        self._cancelled_frames = 0

    @property
    def in_speech(self) -> bool:
        return self._endpointer.in_speech

    @property
    def dropped_frames(self) -> int:
        return self._dropped

    @property
    def processed_frames(self) -> int:
        return self._processed

    @property
    def cancelled_frames(self) -> int:
        """Frames that had echo subtracted. Zero here while speaker mode is on means
        the reference never arrived — the failure looks exactly like no AEC at all."""
        return self._cancelled_frames

    @property
    def is_muted(self) -> bool:
        return self._muted.is_set()

    def set_muted(self, muted: bool) -> None:
        if muted:
            self._muted.set()
            self._endpointer.reset()
        else:
            self._muted.clear()

    def start(self) -> None:
        self._worker = threading.Thread(target=self._run, name="aria-vad", daemon=True)
        self._worker.start()
        self.use_microphone(True)

    def use_microphone(self, on: bool) -> None:
        """Open or close the sound card, leaving the VAD thread running either way.

        This is how a Discord voice call takes over her hearing: the mic closes and
        `submit` becomes the only source. Exclusive on purpose — he is at the same desk
        wearing the same headset, so listening to both would hear every sentence twice
        and endpoint on whichever copy arrived first.

        Deliberately not `set_muted`. Mute is his: it means she may not listen at all,
        and it must keep meaning that while she is in a call.
        """
        if on and self._stream is None:
            self._stream = sd.InputStream(
                samplerate=self._cfg.audio.sample_rate,
                blocksize=self._cfg.audio.frame_samples,
                channels=1,
                dtype="float32",
                device=self._cfg.audio.input_device,
                callback=self._on_audio,
            )
            self._stream.start()
        elif not on and self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            # Whatever was half-said into the microphone is not the start of what will
            # be said into the call.
            self._endpointer.reset()

    @property
    def using_microphone(self) -> bool:
        return self._stream is not None

    def submit(self, frame: np.ndarray) -> None:
        """Feed one frame from somewhere that is not the sound card.

        Same queue, same VAD thread, same endpointer — so speculation, barge-in and the
        turn shape all behave identically no matter which ear the audio came in through.
        Frames arrive from a network thread, and the queue is the thread boundary that
        was already there.
        """
        if self._muted.is_set():
            return
        try:
            self._frames.put_nowait((frame, None))
        except queue.Full:
            self._dropped += 1

    def stop(self) -> None:
        self.use_microphone(False)
        self._frames.put(None)
        if self._worker is not None:
            self._worker.join(timeout=2.0)

    # --- audio thread ---------------------------------------------------------
    def _on_audio(self, indata, frames, time_info, status) -> None:
        if self._muted.is_set():
            return
        reference = None
        if self._aec is not None and self._playback is not None:
            # Cheap slice of an already-resampled buffer. The cancellation itself is
            # 0.55 ms and belongs on the VAD thread, not here.
            reference = self._playback.reference_for(
                frames, self._cfg.barge_in.aec_delay_ms
            )
        try:
            # copy: sounddevice reuses the buffer after the callback returns
            self._frames.put_nowait((indata[:, 0].copy(), reference))
        except queue.Full:
            self._dropped += 1  # VAD thread is wedged; better to drop than block audio

    def _bridge_maybe_final(
        self, audio: np.ndarray, frames: int, voiced_s: float
    ) -> None:
        """Called on the VAD thread when a pause looks like it might be a turn end."""
        if self._on_maybe_final:
            self._loop.call_soon_threadsafe(
                self._on_maybe_final, audio, frames, voiced_s
            )

    # --- vad thread -----------------------------------------------------------
    def _run(self) -> None:
        was_speaking = False
        while True:
            item = self._frames.get()
            if item is None:
                return
            frame, reference = item
            if len(frame) != self._cfg.audio.frame_samples:
                continue  # Silero is strict about frame size

            if self._aec is not None and reference is not None and reference.any():
                # Only while audio is actually playing. Running the filter against
                # silence teaches it that the echo path is nothing, and it then has
                # to re-converge at the top of every reply.
                frame = self._aec.process(frame, reference)
                self._cancelled_frames += 1

            self._processed += 1
            utterance = self._endpointer.feed(frame)

            if self._endpointer.in_speech and not was_speaking and self._on_speech_start:
                self._loop.call_soon_threadsafe(self._on_speech_start)
            was_speaking = self._endpointer.in_speech

            if utterance is not None:
                self._loop.call_soon_threadsafe(self._out.put_nowait, utterance)
