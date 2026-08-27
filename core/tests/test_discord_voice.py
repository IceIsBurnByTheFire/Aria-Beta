"""Voice-call audio: the two format conversions, and the routing that must stay exclusive.

Everything here fails *quietly* if it is wrong, which is why it is worth pinning. A
wrong sample rate is not silence — it is a voice at the wrong pitch, or speech that
Whisper transcribes slightly worse for reasons nobody attributes to resampling. A frame
one byte off is a stutter. And two consumers draining one playback buffer sounds like
her cutting in and out, while `seconds_played` quietly counts double and starts putting
the wrong words into history.
"""

from __future__ import annotations

import numpy as np
import pytest

from aria.audio.playback import Playback
from aria.discord_voice import (
    DISCORD_CHANNELS,
    DISCORD_SR,
    FRAME_BYTES,
    FRAME_SAMPLES_24K,
    FRAME_SAMPLES_48K,
    Decimator,
    Ears,
    dave_decrypt,
    to_discord_frame,
)

SPEECH_SR = 16000


def tone(hz: float, seconds: float, sr: int) -> np.ndarray:
    t = np.arange(int(sr * seconds), dtype=np.float32) / sr
    return np.sin(2 * np.pi * hz * t).astype(np.float32)


def dominant_hz(x: np.ndarray, sr: int) -> float:
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.fft.rfftfreq(len(x), 1 / sr)[int(np.argmax(spectrum))])


# --- her voice, going out ----------------------------------------------------
def test_a_packet_is_exactly_one_discord_frame():
    # discord.py asks for 20 ms every 20 ms and does not tolerate a short read.
    out = to_discord_frame(np.zeros(FRAME_SAMPLES_24K, dtype=np.float32))
    assert len(out) == FRAME_BYTES == 3840


def test_it_is_stereo_16_bit_at_48k():
    pcm = np.frombuffer(to_discord_frame(tone(440, 0.02, 24000)), dtype="<i2")
    assert len(pcm) == FRAME_SAMPLES_48K * DISCORD_CHANNELS
    # One voice, duplicated across both channels rather than panned.
    assert np.array_equal(pcm[0::2], pcm[1::2])


def test_pitch_survives_the_upsample():
    # The failure that is easy to ship: treat 24 kHz audio as 48 kHz and she comes out
    # an octave low and half speed, which sounds like a broken model rather than a rate.
    src = tone(300, 0.25, 24000)
    pcm = np.frombuffer(to_discord_frame(src), dtype="<i2").astype(np.float32) / 32768
    mono = pcm[0::2]
    assert abs(dominant_hz(mono, DISCORD_SR) - 300) < 8


def test_loud_audio_clips_instead_of_wrapping():
    # int16 overflow wraps to full-scale noise of the opposite sign - a horrible crackle
    # rather than the mild distortion clipping gives.
    pcm = np.frombuffer(to_discord_frame(np.full(FRAME_SAMPLES_24K, 4.0, np.float32)),
                        dtype="<i2")
    assert pcm.max() == 32767 and pcm.min() >= 0


# --- his voice, coming in ----------------------------------------------------
def test_decimation_keeps_speech_pitch():
    src = tone(440, 0.5, DISCORD_SR)
    out = Decimator()(src)
    assert abs(len(out) - len(src) / 3) <= 2
    assert abs(dominant_hz(out, SPEECH_SR) - 440) < 12


def test_it_filters_instead_of_just_dropping_samples():
    # 12 kHz cannot exist at 16 kHz, so naive decimation folds it down to 4 kHz and
    # lands it in the middle of the speech band. Nothing errors; Whisper just gets
    # slightly worse forever.
    aliased = Decimator()(tone(12000, 0.5, DISCORD_SR))
    assert float(np.sqrt(np.mean(aliased**2))) < 0.05, "12 kHz should be filtered away"


def test_a_tone_survives_being_split_across_packets():
    # State has to carry between chunks. Reset per packet and every 20 ms boundary is a
    # click, which the VAD then happily reports as speech.
    src = tone(440, 0.5, DISCORD_SR)
    d = Decimator()
    out = np.concatenate([d(src[i : i + FRAME_SAMPLES_48K])
                          for i in range(0, len(src), FRAME_SAMPLES_48K)])
    assert abs(dominant_hz(out, SPEECH_SR) - 440) < 12
    assert np.abs(np.diff(out)).max() < 0.2, "a boundary click would show up here"


def test_ears_reblock_to_exactly_what_silero_wants():
    # Silero v5 requires exactly 512 samples at 16 kHz. A packet produces 320, so the
    # remainder has to be carried rather than padded or dropped.
    got: list[np.ndarray] = []
    ears = Ears(512, got.append)
    packet = (tone(440, 0.02, DISCORD_SR) * 16000).astype("<i2")
    stereo = np.repeat(packet, DISCORD_CHANNELS).tobytes()
    for _ in range(20):
        ears.feed(stereo)
    assert got, "no frames came out at all"
    assert all(len(f) == 512 for f in got)
    assert ears.packets == 20


def test_ears_ignore_an_empty_packet():
    got = []
    ears = Ears(512, got.append)
    ears.feed(b"")
    assert not got and ears.packets == 0


# --- end-to-end encryption ---------------------------------------------------
class FakeSession:
    """Stands in for `davey.DaveSession`."""

    def __init__(self, ready=True, out=b"plain", raises=False):
        self.ready = ready
        self._out = out
        self._raises = raises
        self.calls: list[tuple] = []

    def decrypt(self, user_id, media_type, packet):
        self.calls.append((user_id, packet))
        if self._raises:
            raise RuntimeError("bad key")
        return self._out


def test_no_session_means_an_ordinary_call(monkeypatch):
    assert dave_decrypt(None, 1, b"opus") == b"opus"


def test_a_session_that_is_not_ready_yet_passes_the_packet_through():
    # Mid-handshake, or everyone is in passthrough. The payload is not wrapped, so
    # decrypting it would destroy audio that was already fine.
    assert dave_decrypt(FakeSession(ready=False), 1, b"opus") == b"opus"


def test_a_ready_session_decrypts():
    session = FakeSession(out=b"decoded")
    assert dave_decrypt(session, 4242, b"wrapped") == b"decoded"
    assert session.calls == [(4242, b"wrapped")]


def test_ready_may_be_a_method_rather_than_a_flag():
    # davey is a Rust binding; `ready` introspects as a builtin and could be either.
    session = FakeSession(out=b"decoded")
    session.ready = lambda: True
    assert dave_decrypt(session, 1, b"wrapped") == b"decoded"


def test_a_failed_decrypt_is_a_dropped_frame_not_an_exception():
    # It must never reach the router thread: that catches, logs, and then calls
    # stop_listening() in its finally, which ends reception for the whole call.
    assert dave_decrypt(FakeSession(raises=True), 1, b"wrapped") is None


# --- routing: exactly one consumer -------------------------------------------
class FakeLoop:
    def call_soon_threadsafe(self, fn, *a):
        fn(*a)


@pytest.fixture
def playback(monkeypatch) -> Playback:
    # No sound card in a test run, and none needed: the buffer is the thing under test.
    class FakeStream:
        def __init__(self, **kw): pass
        def start(self): pass
        def stop(self): pass
        def close(self): pass

    monkeypatch.setattr("aria.audio.playback.sd.OutputStream", FakeStream)
    return Playback(FakeLoop(), sample_rate=24000)


def test_pull_drains_the_same_buffer_the_speakers_would(playback):
    playback.begin()
    playback.write(np.full(FRAME_SAMPLES_24K, 0.5, np.float32))
    got = playback.pull(FRAME_SAMPLES_24K)
    assert got is not None and np.allclose(got, 0.5)
    assert playback.seconds_played == pytest.approx(0.02, abs=1e-6)


def test_pull_returns_none_only_when_the_reply_is_over(playback):
    playback.begin()
    playback.write(np.full(FRAME_SAMPLES_24K, 0.5, np.float32))
    assert playback.pull(FRAME_SAMPLES_24K) is not None
    # Mid-reply with nothing buffered yet: silence, not the end. Returning b"" here
    # would end the packet stream every time generation fell behind realtime.
    assert playback.pull(FRAME_SAMPLES_24K) is not None
    playback.end_of_stream()
    assert playback.pull(FRAME_SAMPLES_24K) is None


def test_a_short_read_is_padded_not_shortened(playback):
    playback.begin()
    playback.write(np.full(100, 0.5, np.float32))
    got = playback.pull(FRAME_SAMPLES_24K)
    assert len(got) == FRAME_SAMPLES_24K, "a voice packet is a fixed size"
    assert np.allclose(got[100:], 0.0)


def test_the_speaker_goes_silent_while_the_call_is_pulling(playback):
    playback.route_external(True)
    playback.begin()
    playback.write(np.full(FRAME_SAMPLES_24K, 0.5, np.float32))

    out = np.zeros((256, 1), dtype=np.float32)
    playback._on_audio(out, 256, None, None)
    assert np.allclose(out, 0.0), "two consumers would each get half of every reply"
    assert playback.seconds_played == 0.0, "and seconds_played would count double"

    assert np.allclose(playback.pull(FRAME_SAMPLES_24K)[:256], 0.5), "the call still gets it"


def test_barge_in_still_truncates_by_what_the_call_actually_played(playback):
    # `seconds_played` is what decides which words go into history when he talks over
    # her. Routed to a call it has to keep meaning the same thing.
    playback.route_external(True)
    playback.begin()
    playback.write(np.full(FRAME_SAMPLES_24K * 5, 0.5, np.float32))
    playback.pull(FRAME_SAMPLES_24K)
    playback.pull(FRAME_SAMPLES_24K)
    assert playback.flush() == pytest.approx(0.04, abs=1e-6)


# --- what counts as hearing him -----------------------------------------------
# The extension fills transmission gaps with `SilencePacket`, which carries a canned
# `OPUS_SILENCE` payload: no end-to-end layer to peel, and valid Opus. So it decodes
# unconditionally, whatever is wrong with the call. Counting that as hearing is what
# made a call where every real word failed to decrypt announce itself as working and
# then say nothing for the rest of the session.


class FakePacket:
    def __init__(self, data: bytes, silence: bool = False, ssrc: int = 7):
        self.decrypted_data = data
        self.ssrc = ssrc
        self._silence = silence

    def is_silence(self) -> bool:
        return self._silence


class FakeData:
    def __init__(self, packet):
        self.packet = packet


class FakeUser:
    """A speaker the sink can attribute a packet to. Unattributable packets take a
    different branch — there is no key to try — and that is not what these test."""

    id = 4242


@pytest.fixture
def sink(monkeypatch):
    """A `_Sink` with the decoder and the DAVE session stubbed out.

    `dave` is the switch the tests flip: None is an ordinary call, an object is an
    end-to-end one, and `dave_fails` decides whether his speech survives it.
    """
    import discord

    from aria.discord_bot import _Sink

    class FakeDecoder:
        def decode(self, raw, fec=False):
            if raw == b"undecodable":
                raise ValueError("corrupted stream")
            return b"\x00" * (FRAME_SAMPLES_48K * DISCORD_CHANNELS * 2)

    monkeypatch.setattr(discord.opus, "Decoder", FakeDecoder)

    frames: list = []
    s = _Sink(Ears(512, frames.append))
    s.frames_out = frames
    s.dave = None
    monkeypatch.setattr(type(s), "_dave_session", lambda self: self.dave)
    return s


def speech(sink, n=5, data=b"opus"):
    for _ in range(n):
        sink.write(FakeUser(), FakeData(FakePacket(data)))


def generated_silence(sink, n=5):
    for _ in range(n):
        sink.write(FakeUser(), FakeData(FakePacket(b"silence", silence=True)))


def test_generated_silence_is_not_hearing_him(sink):
    generated_silence(sink, 10)
    assert sink.decoded == 10, "it decodes, and it should — it ends his turn"
    assert sink.heard == 0, "but nobody said anything"


def test_silence_still_reaches_the_endpointer(sink):
    # Discord stops transmitting entirely between words, so these invented frames are
    # the only thing that lets a turn close. Not counting them must not mean dropping
    # them: without this the measured symptom was a reply 37 seconds late.
    generated_silence(sink, 60)
    assert sink.frames_out, "the pause never reached the VAD, so no turn can ever end"


def test_real_speech_is_hearing_him(sink):
    speech(sink, 3)
    assert (sink.heard, sink.failed) == (3, 0)


def test_a_call_where_only_silence_gets_through_still_complains(sink, capsys):
    """The regression. Every word he says fails the end-to-end layer; the gaps between
    them decode perfectly. Gating the complaint on `decoded` meant the first generated
    frame muted it permanently, and the console then showed nothing at all — which is
    exactly what a working call looks like from the outside."""
    sink.dave = object()
    monkeypatch_target = "aria.discord_bot.dave_decrypt"
    import aria.discord_bot as bot

    original = bot.dave_decrypt
    bot.dave_decrypt = lambda *a, **k: None
    try:
        speech(sink, 3)
        generated_silence(sink, 3)   # this is what used to silence the warning
        capsys.readouterr()          # discard the first complaint
        sink._reported_at = 0.0
        speech(sink, 3)
    finally:
        bot.dave_decrypt = original

    assert sink.heard == 0
    assert "none have been usable" in capsys.readouterr().out, (
        f"{monkeypatch_target} failing for every word went unreported"
    )


def test_hearing_him_once_is_enough_to_stop_complaining(sink, capsys):
    # The other half of the same judgement: a call that works must stay quiet, or the
    # warning is noise and gets ignored on the call where it matters.
    speech(sink, 1)
    capsys.readouterr()
    sink.write(FakeUser(), FakeData(FakePacket(b"undecodable")))
    assert capsys.readouterr().out == ""
