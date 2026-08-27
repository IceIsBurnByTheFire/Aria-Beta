"""End-to-end pipeline check without a microphone.

Synthesises a question with Kokoro, plays it through the real Silero endpointer frame
by frame, then runs the resulting utterance through the real turn handler. That covers
VAD endpointing, Whisper, Ollama, the chunker, Kokoro and playback — everything except
the mic driver itself.

Run:  uv run --directory . python tests/e2e_offline.py
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

QUESTIONS = [
    "What is the capital of France?",
    "Hey, can you hear me okay?",
    "Explain what a compiler does, briefly.",
]


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

    failures = 0
    for question in QUESTIONS:
        print(f"\n{'─' * 70}\nspeaking into the pipeline: {question!r}")

        t0 = time.perf_counter()
        utt, source_s = speak_into_pipeline(
            cfg, loop.tts.synth, question, on_maybe_final=loop.speculate
        )
        if utt is None:
            print("  FAIL: VAD never produced an utterance")
            failures += 1
            continue

        print(f"  VAD: {utt.duration_s:.2f}s captured (source speech {source_s:.2f}s), "
              f"endpointer ran in {(time.perf_counter() - t0) * 1000:.0f}ms")

        # Live, speculation starts this far before the endpoint decision. Sleeping the
        # real interval is what makes the measurement honest: the speculative task gets
        # exactly the head start a real microphone would give it, and no more.
        await asyncio.sleep(
            (cfg.vad.end_silence_ms - cfg.vad.speculate_after_ms) / 1000
        )
        rebase(cfg, utt)
        await loop._handle_turn(utt)

    loop.shutdown()
    print(loop.session.summary())

    if len(loop.session.turns) < len(QUESTIONS):
        print(f"\n{len(QUESTIONS) - len(loop.session.turns)} turn(s) produced no reply")
        failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
