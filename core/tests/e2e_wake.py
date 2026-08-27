"""The wake gate against the real pipeline: does the room actually get ignored?

`test_wake.py` checks the matcher. This checks the thing that matters — that an
utterance she wasn't addressed in costs nothing past STT: no LLM call, no speech, no
history, no memory. And that being named still gets a real answer.

Run:  uv run --directory . python tests/e2e_wake.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _harness import rebase, speak_into_pipeline  # noqa: E402

from aria.config import Config  # noqa: E402
from aria.loop import VoiceLoop  # noqa: E402

RED, GREEN, DIM, RESET = "\033[31m", "\033[32m", "\033[2m", "\033[0m"
results: list[tuple[bool, str]] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    results.append((ok, label))
    mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"  [{mark}] {label}" + (f"  — {detail}" if detail else ""))


async def main() -> int:
    cfg = Config()
    cfg.validate()
    cfg.memory.path = Path(tempfile.mkdtemp(prefix="aria-test-")) / "memory.json"
    cfg.memory.auto_extract = False
    cfg.wake.enabled = True
    cfg.wake.window_s = 30.0

    loop = VoiceLoop(cfg)
    await loop.setup(listen=False)

    llm_calls = {"n": 0}
    real_stream = loop.llm.stream

    def counting_stream(payload):
        llm_calls["n"] += 1
        return real_stream(payload)

    loop.llm.stream = counting_stream

    async def say(text: str) -> None:
        utt, _ = speak_into_pipeline(cfg, loop.tts.synth, text)
        if utt is None:
            raise RuntimeError(f"VAD produced no utterance for {text!r}")
        rebase(cfg, utt)
        await loop._handle_turn(utt)

    # --- the room ------------------------------------------------------------
    print("\n--- speech she was not addressed in ---")
    before = llm_calls["n"]
    await say("So then he told me the whole build was broken again.")
    check(llm_calls["n"] == before, "no LLM call", f"{llm_calls['n'] - before} calls")
    check(loop._history == [], "nothing entered the conversation history")
    check(loop.playback.seconds_played == 0.0, "she stayed quiet")

    # --- addressed by name ---------------------------------------------------
    print("\n--- addressed by name ---")
    await say("Aria, what is the capital of France?")
    check(llm_calls["n"] == before + 1, "the LLM ran once")
    asked = next((m["content"] for m in loop._history if m["role"] == "user"), "")
    print(f"{DIM}  model was asked: {asked!r}{RESET}")
    check("aria" not in asked.lower(), "her name was stripped before the model saw it",
          repr(asked))
    check(bool(asked.strip()), "something was still left to answer")

    # --- the window ----------------------------------------------------------
    print("\n--- follow-up, no name ---")
    n = llm_calls["n"]
    await say("And what about Germany?")
    check(llm_calls["n"] == n + 1, "a follow-up inside the window needs no name")

    loop.shutdown()

    failed = [label for ok, label in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    for label in failed:
        print(f"  {RED}FAILED{RESET}: {label}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
