"""M5: the privacy path — arming, capturing, and the watching indicator.

Deliberately does not need a vision model. What it checks is the part that must be
right regardless of backend: capture never happens unless armed, the overlay is told
every time the state changes, and a screen question while disarmed neither quietly
grabs a frame nor quietly pretends the capability does not exist.

Run:  uv run --directory . python tests/e2e_vision.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import websockets  # noqa: E402
from _harness import rebase, speak_into_pipeline  # noqa: E402

from aria.config import Config  # noqa: E402
from aria.loop import VoiceLoop  # noqa: E402

RED, GREEN, RESET = "\033[31m", "\033[32m", "\033[0m"
results: list[tuple[bool, str]] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    results.append((ok, label))
    mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"  [{mark}] {label}" + (f"  — {detail}" if detail else ""))


async def mock_overlay(url, received, ready):
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"type": "hello", "model_name": "test", "emotions": []}))
        ready.set()
        try:
            async for raw in ws:
                msg = json.loads(raw)
                if msg["type"] in ("vision", "notice", "subtitle"):
                    received.append(msg)
        except websockets.ConnectionClosed:
            pass


async def main() -> int:
    cfg = Config()
    # The privacy path is backend-agnostic, but arming preflights whichever backend
    # is configured. Pin the local one so a flipped default or an empty API balance
    # can never fail this test for a reason it isn't testing.
    cfg.vision.backend = "ollama"
    cfg.validate()
    # Never touch her real memory. These conversations are fiction, and a test run
    # that leaves "he has a demo on Friday" in the file she loads tomorrow is a bug
    # in the tests, not a feature.
    cfg.memory.path = Path(tempfile.mkdtemp(prefix="aria-test-")) / "memory.json"
    cfg.memory.auto_extract = False
    loop = VoiceLoop(cfg)
    await loop.setup(listen=False)

    url = f"ws://{cfg.server.host}:{cfg.server.port}"
    seen: list[dict] = []
    ready = asyncio.Event()
    client = asyncio.create_task(mock_overlay(url, seen, ready))
    await asyncio.wait_for(ready.wait(), timeout=5)
    await asyncio.sleep(0.1)

    check(not loop._watching, "screen watching starts OFF")

    # Count captures without needing a working vision model.
    captures = {"n": 0}
    real_capture = loop.capture.capture

    def spy(monitor=None):
        captures["n"] += 1
        return real_capture(monitor)

    loop.capture.capture = spy

    async def say(text: str) -> None:
        utt, _ = speak_into_pipeline(cfg, loop.tts.synth, text)
        if utt is None:
            raise RuntimeError(f"VAD produced no utterance for {text!r}")
        rebase(cfg, utt)
        await loop._handle_turn(utt)
        await asyncio.sleep(0.2)

    # --- a screen question while disarmed must NOT capture --------------------
    print("\n--- disarmed ---")
    await say("What is on my screen?")
    check(captures["n"] == 0, "no capture while disarmed", f"{captures['n']} captures")

    # Not capturing is only half of it. Silence here used to send the question to the
    # chat model with no image and no explanation, and it answered "I can't see your
    # screen" — which sounds like a capability Aria lacks rather than a switch that
    # is off, and never told the user the words that would turn it on.
    final = [m["text"] for m in seen if m["type"] == "subtitle" and m.get("final")]
    check(bool(final) and "watch my screen" in final[-1].lower(),
          "disarmed answer says how to turn it on",
          repr(final[-1] if final else None))

    # --- arming ---------------------------------------------------------------
    print("\n--- arming by voice ---")
    await say("Watch my screen.")
    vision_events = [m for m in seen if m["type"] == "vision"]
    check(loop._watching, "voice command armed watching")
    check(any(m.get("watching") for m in vision_events),
          "overlay told watching turned on", str(len(vision_events)) + " vision events")
    check(any(m["type"] == "notice" and "ON" in m.get("text", "") for m in seen),
          "a visible notice was sent too")

    # --- a screen question while armed SHOULD capture -------------------------
    print("\n--- armed ---")
    before = captures["n"]
    await say("What does this error say?")
    check(captures["n"] == before + 1, "captured once while armed",
          f"{captures['n'] - before} captures")
    check(any(m.get("capturing") for m in seen if m["type"] == "vision"),
          "overlay saw the capturing pulse")

    # --- an ordinary question must not capture --------------------------------
    before = captures["n"]
    await say("What is the capital of France?")
    check(captures["n"] == before, "ordinary question captured nothing",
          f"{captures['n'] - before} captures")

    # --- disarming ------------------------------------------------------------
    print("\n--- disarming by voice ---")
    await say("Stop watching my screen.")
    check(not loop._watching, "voice command disarmed watching")
    check(any(m["type"] == "vision" and not m.get("watching") for m in seen),
          "overlay told watching turned off")

    client.cancel()
    loop.shutdown()

    failed = [label for ok, label in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    for label in failed:
        print(f"  {RED}FAILED{RESET}: {label}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
