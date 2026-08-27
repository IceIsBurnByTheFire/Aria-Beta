"""Entry point: `aria` or `python -m aria`."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

import sounddevice as sd

from .config import Config
from .loop import VoiceLoop


def _list_devices() -> None:
    print(sd.query_devices())
    print(f"\ndefault input : {sd.query_devices(kind='input')['name']}")
    print(f"default output: {sd.query_devices(kind='output')['name']}")


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="aria", description="Aria voice loop (M1)")
    p.add_argument("--list-devices", action="store_true", help="show audio devices and exit")
    p.add_argument("--list-voices", action="store_true", help="show Kokoro voices and exit")
    p.add_argument("--voice", help="Kokoro voice, e.g. af_bella")
    p.add_argument(
        "--language",
        choices=["auto", "en", "zh"],
        help="which language she listens and speaks in. Sets the Whisper language, the "
             "Kokoro voice and phonemiser, and adds a prompt block. auto (the default) "
             "follows whatever you speak, per turn, and changes voice with it; en and "
             "zh pin her to one, which is worth doing in a noisy room. zh is "
             "Traditional Chinese.",
    )
    p.add_argument(
        "--flat-voice",
        action="store_true",
        help="stop the [emotion] markers colouring the delivery. She still emotes on "
             "screen, she just sounds the same doing it.",
    )
    p.add_argument("--llm-model", help="Ollama model tag")
    p.add_argument(
        "--llm-backend",
        choices=["ollama", "openrouter", "groq", "google"],
        help="where the conversation model runs. Local by default: private, offline, "
             "unmetered, and what the latency budget was measured against.",
    )
    p.add_argument("--cloud-model", help="model id for the chosen provider")
    p.add_argument(
        "--list-cloud-models",
        action="store_true",
        help="show each provider's free tier and what it is serving right now, then "
             "exit. Free model ids are retired and renamed often, so the live list "
             "beats any hardcoded one.",
    )
    p.add_argument("--input-device", help="mic device index or name")
    p.add_argument("--output-device", help="speaker device index or name")
    p.add_argument(
        "--end-silence",
        type=int,
        metavar="MS",
        help="trailing silence that ends a turn (default 600). The dominant term in "
             "perceived latency — lower feels snappier but cuts people off.",
    )
    p.add_argument(
        "--no-barge-in",
        action="store_true",
        help="disable voice interruption entirely. The blunt fix for speakers; prefer "
             "--speaker-mode, which keeps interruption working.",
    )
    p.add_argument(
        "--no-idle-talk",
        action="store_true",
        help="stop her speaking first. By default she calls out after a few minutes "
             "of silence, then backs off and gives up rather than nagging.",
    )
    p.add_argument(
        "--wake-word",
        nargs="?",
        const="aria",
        metavar="WORD",
        help="only answer when spoken to by name (default 'aria'). Saying it opens a "
             "short window, so follow-ups don't need it again.",
    )
    p.add_argument(
        "--speaker-mode",
        action="store_true",
        help="subtract Aria's own voice from the mic so barge-in works without "
             "headphones. Needs the echo canceller; says so plainly if it is missing.",
    )
    p.add_argument(
        "--memory",
        action="store_true",
        help="show everything she remembers about you, and where the file lives",
    )
    p.add_argument(
        "--no-discord",
        action="store_true",
        help="don't connect to Discord this run, even with a token configured",
    )
    p.add_argument(
        "--speak-discord",
        action="store_true",
        help="read Discord replies out loud as well as posting them. Off by default: "
             "you are usually on Discord because you are not at the desk.",
    )
    p.add_argument("--list-monitors", action="store_true", help="show screens and exit")
    p.add_argument(
        "--watch-screen",
        action="store_true",
        help="start with screen watching armed. Off by default — capture is invasive, "
             "so it stays off until you ask for it, by flag or out loud.",
    )
    p.add_argument("--monitor", type=int, metavar="N", help="which screen to capture")
    p.add_argument("--vision-backend", choices=["ollama", "claude"],
                   help="local model or the Claude API for screen reading")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _show_memory() -> None:
    """Print everything she remembers, and where it lives.

    Memory she can't be shown is memory he can't correct, and a note that is quietly
    wrong is the failure mode that matters here — so this always prints the path, even
    when there is nothing in it yet.
    """
    import time

    from .config import Config
    from .memory import SOURCE_USER, Memory

    cfg = Config()
    memory = Memory(cfg.memory.path, cfg.memory.max_notes).load()
    print(f"\n  {cfg.memory.path}")

    if not cfg.memory.path.exists():
        print("\n  Nothing yet — the file is written the first time she runs.\n")
        return

    c = memory.continuity
    if c.sessions:
        first = time.strftime("%d %B %Y", time.localtime(c.first_seen))
        print(f"  {c.sessions} sessions since {first}")

    if not memory.notes:
        print("\n  She hasn't kept any notes yet.\n")
        return

    print(f"\n  {len(memory.notes)} of {memory.max_notes} notes "
          f"(* = you asked her to remember it)\n")
    for note in sorted(memory.notes, key=lambda n: n.created_at):
        mark = "*" if note.source == SOURCE_USER else " "
        when = time.strftime("%d %b", time.localtime(note.created_at))
        print(f"  {mark} {when}  {note.text}")
    print('\n  Say "forget <something>" to drop one, or edit the file directly.\n')


def _list_cloud_models() -> int:
    """Show each provider's free tier, and what OpenRouter is giving away right now.

    Worth a flag rather than a doc note: free model ids are retired and renamed
    regularly, so anything written down would be wrong within months, and the failure
    it causes — a 404 on the first thing you say — does not look like a stale config
    file. Only OpenRouter has a public unauthenticated catalogue; the others need a key,
    so for those this prints the default and where to look.
    """
    import httpx

    from .config import CLOUD_PROVIDERS, Config

    cfg = Config().llm
    print()
    for name, p in CLOUD_PROVIDERS.items():
        here = " <- current" if name == cfg.backend else ""
        have = "key set" if os.getenv(p.key_env) else f"no {p.key_env}"
        print(f"  {name}{here}")
        print(f"    default   {p.default_model}")
        print(f"    free tier {p.limits}")
        print(f"    keys      {p.keys_url}  ({have})")
        if p.caveat:
            print(f"    note      {p.caveat}")
        print()

    try:
        reply = httpx.get("https://openrouter.ai/api/v1/models", timeout=25.0)
        reply.raise_for_status()
        free = sorted(
            (m for m in reply.json()["data"] if m["id"].endswith(":free")),
            key=lambda m: -int(m.get("context_length") or 0),
        )
    except Exception as e:  # noqa: BLE001 — a catalogue we could not fetch is not fatal
        print(f"  (Could not fetch OpenRouter's live list: {type(e).__name__})\n")
        return 0

    if free:
        print(f"  OpenRouter is serving {len(free)} free models right now:\n")
        for m in free:
            mark = "*" if m["id"] == cfg.active_cloud_model else " "
            print(f"  {mark} {m['id']:<52} {int(m.get('context_length') or 0):>9,} ctx")

    print("\n  Choose with ARIA_LLM_BACKEND and ARIA_CLOUD_MODEL in core/.env.")
    print("  A model id is per-provider: a Groq id means nothing to Google.\n")
    return 0


def _as_device(value: str | None) -> int | str | None:
    if value is None:
        return None
    return int(value) if value.isdigit() else value


def main(argv: list[str] | None = None) -> int:
    # The metrics line is drawn with box characters and the console defaults to cp1252
    # on Windows the moment stdout stops being a terminal. Piping her output to a file
    # then dies on a UnicodeEncodeError halfway through a turn, which looks like a bug
    # in the voice loop and is really just the encoding.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = _parse(argv)

    if args.list_devices:
        _list_devices()
        return 0

    if args.list_monitors:
        from .vision.capture import list_monitors

        for i, mon in enumerate(list_monitors()):
            label = "all screens combined" if i == 0 else mon.get("name", "monitor")
            primary = "  [primary]" if mon.get("is_primary") else ""
            print(f"  {i}: {mon['width']}x{mon['height']} at "
                  f"({mon['left']},{mon['top']})  {label}{primary}")
        return 0

    if args.memory:
        _show_memory()
        return 0

    if args.list_cloud_models:
        return _list_cloud_models()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    cfg = Config(verbose=args.verbose)
    # Before the individual overrides below, so --voice can still pick a different
    # voice within the chosen language.
    from .config import apply_language
    # The loop announces which one it landed on at startup — a run pinned to the wrong
    # language does not error, it just stops understanding him.
    apply_language(cfg, args.language or cfg.language)
    if args.voice:
        cfg.tts.voice = args.voice
    if args.llm_model:
        cfg.llm.model = args.llm_model
    if args.llm_backend:
        cfg.llm.backend = args.llm_backend
    if args.cloud_model:
        cfg.llm.cloud_model = args.cloud_model
    if args.end_silence:
        cfg.vad.end_silence_ms = args.end_silence
    if args.no_barge_in:
        cfg.barge_in.enabled = False
    if args.speaker_mode:
        cfg.barge_in.speaker_mode = True
    if args.flat_voice:
        cfg.tts.emotion_voice = False
    if args.no_idle_talk:
        cfg.idle.enabled = False
    if args.wake_word:
        cfg.wake.enabled = True
        cfg.wake.word = args.wake_word.lower()
    if args.watch_screen:
        cfg.vision.enabled = True
    if args.monitor is not None:
        cfg.vision.monitor = args.monitor
    if args.vision_backend:
        cfg.vision.backend = args.vision_backend
    if args.no_discord:
        # The token is the enable switch, so removing it is how you turn her off for a
        # run — nothing downstream has to learn a second way to be disabled.
        cfg.discord.token = ""
    if args.speak_discord:
        cfg.discord.speak_replies = True
    cfg.audio.input_device = _as_device(args.input_device)
    cfg.audio.output_device = _as_device(args.output_device)

    if args.list_voices:
        from .tts.kokoro_backend import KokoroTTS

        cfg.validate()
        for v in KokoroTTS(cfg.tts).voices():
            print(v)
        return 0

    loop = VoiceLoop(cfg)
    try:
        asyncio.run(loop.run())
    except KeyboardInterrupt:
        print("\n")
    except FileNotFoundError as e:
        print(f"\n{e}", file=sys.stderr)
        return 1
    finally:
        print(loop.session.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
