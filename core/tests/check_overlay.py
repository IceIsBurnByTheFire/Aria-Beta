"""Verify the event stream an overlay would receive.

Runs a real turn with a mock overlay attached and reports what arrived: the state
sequence, the viseme envelope, and the subtitles. This is the core half of M3 — if this
looks right, the renderer has everything it needs.

Run:  uv run --directory . python tests/check_overlay.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from collections import Counter
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


async def mock_overlay(url: str, received: list[dict], ready: asyncio.Event) -> None:
    """Stands in for the Electron renderer: says hello, records everything."""
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({
            "type": "hello",
            "model_format": "live2d",
            "model_name": "haru",
            "expressions": ["F01", "F02", "F03"],
            "motions": ["idle", "wave"],
        }))
        ready.set()
        try:
            async for raw in ws:
                received.append(json.loads(raw))
        except websockets.ConnectionClosed:
            pass


async def main() -> int:
    cfg = Config()
    cfg.validate()
    # Never touch her real memory. These conversations are fiction, and a test run
    # that leaves "he has a demo on Friday" in the file she loads tomorrow is a bug
    # in the tests, not a feature.
    cfg.memory.path = Path(tempfile.mkdtemp(prefix="aria-test-")) / "memory.json"
    cfg.memory.auto_extract = False

    loop = VoiceLoop(cfg)
    await loop.setup(listen=False)

    url = f"ws://{cfg.server.host}:{cfg.server.port}"
    received: list[dict] = []
    ready = asyncio.Event()
    client = asyncio.create_task(mock_overlay(url, received, ready))
    await asyncio.wait_for(ready.wait(), timeout=5)
    await asyncio.sleep(0.1)

    check(loop.server.connected, "overlay connected")
    check(loop.server.capabilities.get("model_name") == "haru",
          "hello received", str(loop.server.capabilities.get("expressions")))

    utt, _ = speak_into_pipeline(
        cfg, loop.tts.synth, "What is the capital of France?",
        on_maybe_final=loop.speculate,
    )
    if utt is None:
        print("FAIL: VAD produced no utterance")
        return 1
    await asyncio.sleep((cfg.vad.end_silence_ms - cfg.vad.speculate_after_ms) / 1000)
    rebase(cfg, utt)
    await loop._handle_turn(utt)
    await asyncio.sleep(0.2)

    kinds = Counter(m["type"] for m in received)
    states = [m["value"] for m in received if m["type"] == "state"]
    visemes = [m["open"] for m in received if m["type"] == "viseme"]
    subtitles = [m for m in received if m["type"] == "subtitle"]

    print(f"\n  events: {dict(kinds)}")
    print(f"  states: {' → '.join(states)}")
    if visemes:
        open_frames = sorted(v for v in visemes if v > 0)
        clipped = sum(1 for v in visemes if v >= 1.0) / max(1, len(open_frames))
        median = open_frames[len(open_frames) // 2] if open_frames else 0
        print(f"  viseme: {len(visemes)} frames, range {min(visemes):.2f}-{max(visemes):.2f}, "
              f"median-while-open {median:.2f}, clipped {clipped:.0%}")
    if subtitles:
        print(f'  final subtitle: "{subtitles[-1]["text"][:70]}"')

    check("thinking" in states and "speaking" in states,
          "state sequence covers thinking and speaking")
    check(states[-1] == "idle", "ends back at idle", states[-1] if states else "none")
    check(len(visemes) > 20, "viseme stream is dense enough to animate",
          f"{len(visemes)} frames")
    check(bool(visemes) and max(visemes) > 0.15,
          "mouth actually opens", f"peak {max(visemes):.2f}" if visemes else "no frames")
    check(bool(visemes) and visemes[-1] == 0.0,
          "mouth closes when speech ends",
          f"last {visemes[-1]:.2f}" if visemes else "no frames")
    check(any(s.get("final") for s in subtitles), "a final subtitle was sent")

    client.cancel()
    loop.shutdown()

    failed = [label for ok, label in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    for label in failed:
        print(f"  {RED}FAILED{RESET}: {label}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
