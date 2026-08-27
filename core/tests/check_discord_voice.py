"""Can she actually get into the voice call, and does audio leave the machine?

A simulated call cannot tell you about Opus, PyNaCl, gateway permissions, or whether
the channel id in `.env` points at something she is allowed to enter — and every one of
those fails at connect time with an error that names none of them. So this spends one
real connection: join the configured channel, play two seconds of real Kokoro speech,
and report what the voice client says happened.

It cannot tell you it *sounded* right. Sit in the channel while it runs for that.

Run:  uv run --directory . python tests/check_discord_voice.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import discord  # noqa: E402
import numpy as np  # noqa: E402

from aria.config import Config  # noqa: E402
from aria.discord_bot import DiscordBot  # noqa: E402
from aria.discord_voice import Ears  # noqa: E402
from aria.tts.kokoro_backend import KokoroTTS  # noqa: E402

RED, GREEN, YELLOW, DIM, RESET = (
    "\033[31m", "\033[32m", "\033[33m", "\033[2m", "\033[0m",
)
results: list[tuple[bool, str]] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    results.append((ok, label))
    mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"  [{mark}] {label}" + (f"  — {detail}" if detail else ""))


class Buffer:
    """Stands in for `Playback`, so this checks the wire and not the voice loop."""

    def __init__(self, audio: np.ndarray):
        self._audio = audio
        self.pos = 0

    def pull(self, n: int):
        if self.pos >= len(self._audio):
            return None
        chunk = self._audio[self.pos : self.pos + n]
        self.pos += n
        if len(chunk) < n:
            chunk = np.pad(chunk, (0, n - len(chunk)))
        return chunk


LISTEN_S = 12


async def listen_for_him(bot, heard: dict) -> None:
    """The half that sending cannot tell you about: can she hear *him*.

    This exists because of a real call that looked perfect from the console. She joined,
    the console said "hearing you in the call", and she never answered a word — and the
    reason was that the only packets getting through were the ones the extension
    *invents* to fill the gaps between his words. Those carry a canned Opus silence
    payload with no end-to-end layer on it, so they decode whatever is wrong with the
    call, while every real packet failed to decrypt.

    So the numbers are reported separately, and which one is zero says where it broke:

      nothing arrived            he was muted, or Discord was not transmitting
      arrived, none decoded      the end-to-end layer, or Opus
      only silence decoded       his speech specifically is failing to decrypt
      decoded, no frames         the re-blocking between here and the VAD
      frames, but silent         the audio is arriving as digital silence
    """
    sink = bot._sink
    if sink is None:
        return
    before = (sink.heard, sink.decoded, sink.failed, heard["packets"])
    print(f"\n{YELLOW}Say something in the channel — listening for {LISTEN_S}s.{RESET}")
    for left in range(LISTEN_S, 0, -1):
        print(f"{DIM}  {left:2d}s  speech={sink.heard - before[0]} "
              f"silence={(sink.decoded - before[1]) - (sink.heard - before[0])} "
              f"unusable={sink.failed - before[2]} "
              f"frames={heard['packets'] - before[3]}{RESET}", end="\r")
        await asyncio.sleep(1)
    print(" " * 78, end="\r")

    spoke = sink.heard - before[0]
    frames = heard["packets"] - before[3]
    check(sink.decoded - before[1] > 0, "packets reached her at all",
          "nothing arrived — were you in the channel and unmuted?")
    check(spoke > 0, "and some of them were you speaking, not generated silence",
          f"{sink.failed - before[2]} unusable — {sink.why}" if sink.failed > before[2]
          else "only the extension's own silence got through")
    check(frames > 0, "the audio reached the endpointer", f"{frames} frames")
    check(heard["loudest"] > 0.005, "and it was not digital silence",
          f"loudest frame {heard['loudest']:.3f}")


async def main() -> int:
    cfg = Config()
    if not cfg.discord.token:
        print(f"\n{YELLOW}No ARIA_DISCORD_TOKEN in core/.env.{RESET}\n")
        return 1
    if not cfg.discord.voice_channel:
        print(f"\n{YELLOW}No ARIA_DISCORD_VOICE_CHANNEL in core/.env.{RESET}\n")
        return 1

    print(f"\n{DIM}Synthesising a line with Kokoro…{RESET}")
    tts = KokoroTTS(cfg.tts)
    speech = tts.synth("Hey. If you can hear this, the voice call is working.", None)
    print(f"{DIM}  {len(speech) / cfg.tts.sample_rate:.1f}s of audio at "
          f"{cfg.tts.sample_rate} Hz{RESET}")

    heard = {"packets": 0, "loudest": 0.0}

    def took_a_frame(frame) -> None:
        heard["packets"] += 1
        heard["loudest"] = max(heard["loudest"], float(np.abs(frame).max()))

    ears = Ears(cfg.audio.frame_samples, took_a_frame)
    buffer = Buffer(speech)
    done = asyncio.Event()

    bot = DiscordBot(cfg.discord, on_message=lambda *_: asyncio.sleep(0))
    bot.attach_audio(ears, buffer)

    @bot.event
    async def on_ready():
        try:
            print(f"\n{DIM}Connected as {bot.user}{RESET}")
            check(discord.opus.is_loaded() or _load_opus(),
                  "Opus is available (needed to encode and decode voice)")

            channel = bot.get_channel(cfg.discord.voice_channel)
            check(isinstance(channel, discord.VoiceChannel),
                  "the configured id is a voice channel she can see",
                  getattr(channel, "name", str(cfg.discord.voice_channel)))
            if not isinstance(channel, discord.VoiceChannel):
                return

            perms = channel.permissions_for(channel.guild.me)
            check(perms.connect, "she is allowed to connect")
            check(perms.speak, "and allowed to speak")

            problem = await bot.join_voice(cfg.discord.voice_channel)
            check(not problem, "she joined the call", problem or channel.name)
            if problem:
                return
            check(bot.in_voice_call, "and the voice client reports connected")
            check(bot.voice.is_listening(), "receiving is armed")

            bot.start_speaking()
            await asyncio.sleep(0.3)
            check(bot.voice.is_playing(), "she is sending voice packets")

            t0 = time.perf_counter()
            while bot.voice and bot.voice.is_playing() and time.perf_counter() - t0 < 20:
                await asyncio.sleep(0.2)
            played = time.perf_counter() - t0
            expected = len(speech) / cfg.tts.sample_rate
            check(abs(played - expected) < 1.5,
                  "the whole line was sent, in real time",
                  f"{played:.1f}s sent vs {expected:.1f}s of audio")
            check(buffer.pos >= len(speech), "the buffer was drained to the end")

            await listen_for_him(bot, heard)
        finally:
            await bot.leave_voice()
            check(not bot.in_voice_call, "and she left cleanly")
            done.set()
            await bot.close()

    def _load_opus() -> bool:
        try:
            discord.opus._load_default()
            return discord.opus.is_loaded()
        except Exception:
            return False

    try:
        await asyncio.wait_for(bot.start(cfg.discord.token), timeout=90)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass
    except discord.LoginFailure:
        print(f"\n{RED}Discord refused the token.{RESET}\n")
        return 1

    failed = [label for ok, label in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    for label in failed:
        print(f"  {RED}FAILED{RESET}: {label}")
    print(f"\n{DIM}This proves the audio left the machine. Whether it *sounded* right "
          f"needs you in the channel.{RESET}\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
