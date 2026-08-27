"""Barge-in: cutting Aria off mid-sentence, and what she remembers afterwards.

Uses the manual interrupt path rather than a real voice, so the cut lands at a known
moment and the assertions are deterministic. `_on_speech_start` — the voice trigger —
routes into exactly the same `interrupt()`, and its grace window is checked separately
at the end.

Run:  uv run --directory . python tests/e2e_bargein.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _harness import rebase, speak_into_pipeline  # noqa: E402

from aria.config import Config  # noqa: E402
from aria.loop import VoiceLoop  # noqa: E402
from aria.state import State  # noqa: E402

TALKATIVE = """You are Aria, a warm voice companion. Answer out loud in five or six
full sentences. Never use markdown, asterisks, bullet points or emoji."""

RED, GREEN, RESET = "\033[31m", "\033[32m", "\033[0m"
results: list[tuple[bool, str]] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    results.append((ok, label))
    mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"  [{mark}] {label}" + (f"  — {detail}" if detail else ""))


async def turn_until_speaking(loop: VoiceLoop, task: asyncio.Task, timeout=20.0) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if loop.state.current is State.SPEAKING:
            return True
        if task.done():
            return False
        await asyncio.sleep(0.01)
    return False


async def main() -> int:
    cfg = Config()
    cfg.validate()
    # Never touch her real memory. These conversations are fiction, and a test run
    # that leaves "he has a demo on Friday" in the file she loads tomorrow is a bug
    # in the tests, not a feature.
    cfg.memory.path = Path(tempfile.mkdtemp(prefix="aria-test-")) / "memory.json"
    cfg.memory.auto_extract = False
    cfg.persona = TALKATIVE
    cfg.llm.num_predict = 300

    loop = VoiceLoop(cfg)
    await loop.setup(listen=False)

    # Spy on what TTS was actually asked to say, so "generated" can be compared
    # against "spoken".
    generated: list[str] = []
    real_synth = loop.tts.synth

    def spy(text: str, emotion: str | None = None):
        generated.append(text)
        return real_synth(text, emotion)

    loop.tts.synth = spy

    # ---- turn 1: let her start, then cut her off -----------------------------
    print("\n--- turn 1: interrupt after ~1.2s of speech ---")
    utt, _ = speak_into_pipeline(cfg, real_synth, "Tell me about the Roman Empire.")
    if utt is None:
        print("FAIL: VAD produced no utterance")
        return 1
    rebase(cfg, utt)

    task = asyncio.create_task(loop._handle_turn(utt))
    if not await turn_until_speaking(loop, task):
        print("FAIL: never reached SPEAKING")
        return 1

    await asyncio.sleep(1.2)
    t_cut = time.perf_counter()
    loop.interrupt()
    await task
    stop_ms = (time.perf_counter() - t_cut) * 1000

    said = next(
        (m["content"] for m in reversed(loop._history) if m["role"] == "assistant"), ""
    )
    full = " ".join(generated)
    played = loop.playback.seconds_played

    print(f'\n  generated: "{full[:110]}…"  ({len(full)} chars)')
    print(f'  recorded : "{said[:110]}…"  ({len(said)} chars)')
    print(f"  audio played: {played:.2f}s\n")

    check(stop_ms < 100, "playback stops promptly", f"{stop_ms:.0f}ms")
    check(bool(said), "something was recorded", f"{len(said)} chars")
    check(len(said) < len(full), "recorded text is shorter than generated",
          f"{len(said)} < {len(full)}")
    check(full.replace(" ", "").startswith(said.replace(" ", "")),
          "recorded text is a prefix of generated")
    check(loop.session.turns[-1].interrupted, "turn flagged as interrupted")
    check(loop.playback.seconds_played <= 1.5,
          "no queued audio leaked past the cut", f"{played:.2f}s played")

    # ---- turn 2: the conversation must still make sense ----------------------
    print("\n--- turn 2: does it recover ---")
    generated.clear()
    utt2, _ = speak_into_pipeline(cfg, real_synth, "Sorry, go on.")
    if utt2 is None:
        print("FAIL: VAD produced no utterance on turn 2")
        return 1
    rebase(cfg, utt2)
    await loop._handle_turn(utt2)

    check(len(loop.session.turns) == 2, "second turn completed")
    check(not loop.session.turns[-1].interrupted, "second turn ran to completion")
    check(loop.state.current is State.IDLE, "returned to IDLE",
          str(loop.state.current))

    roles = [m["role"] for m in loop._history]
    check(roles == ["user", "assistant", "user", "assistant"],
          "history alternates cleanly", str(roles))

    # ---- the voice trigger's grace window ------------------------------------
    print("\n--- barge-in grace window ---")
    loop.state.to(State.SPEAKING)
    loop._speaking_since = time.perf_counter()
    loop._on_speech_start()
    check(loop.state.current is State.SPEAKING,
          "speech inside the grace window does not interrupt")

    loop._speaking_since = time.perf_counter() - cfg.barge_in.grace_ms / 1000 - 0.05
    loop._on_speech_start()
    check(loop.state.current is State.LISTENING,
          "speech after the grace window does interrupt")

    loop.state.to(State.IDLE)
    cfg.barge_in.enabled = False
    loop.state.to(State.SPEAKING)
    loop._speaking_since = time.perf_counter() - 1.0
    loop._on_speech_start()
    check(loop.state.current is State.SPEAKING,
          "--no-barge-in suppresses the voice trigger")

    loop.shutdown()

    failed = [label for ok, label in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    for label in failed:
        print(f"  {RED}FAILED{RESET}: {label}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
