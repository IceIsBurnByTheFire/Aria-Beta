"""Live microphone plumbing check.

Nothing needs to be said out loud — this verifies the capture stream opens, frames flow
to the VAD thread at the expected rate, and nothing is dropped. If you do speak, it
reports what it heard.

Run:  uv run python tests/check_mic.py [seconds]
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import sounddevice as sd  # noqa: E402

from aria.audio.capture import VoiceListener  # noqa: E402
from aria.audio.vad import Utterance  # noqa: E402
from aria.config import Config  # noqa: E402


async def main(seconds: float) -> int:
    cfg = Config()
    cfg.validate()
    loop = asyncio.get_running_loop()
    utterances: asyncio.Queue[Utterance] = asyncio.Queue()

    speech_starts = 0

    def on_speech_start() -> None:
        nonlocal speech_starts
        speech_starts += 1
        print("  … speech detected")

    print(f"input device: {sd.query_devices(kind='input')['name']}")
    listener = VoiceListener(cfg, loop, utterances, on_speech_start=on_speech_start)

    t0 = time.perf_counter()
    listener.start()
    print(f"listening for {seconds:.0f}s — say something if you like\n")
    await asyncio.sleep(seconds)
    elapsed = time.perf_counter() - t0
    listener.stop()

    expected = elapsed * cfg.audio.sample_rate / cfg.audio.frame_samples
    got = listener.processed_frames
    ratio = got / expected if expected else 0

    print(f"\nframes processed : {got} (expected ~{expected:.0f}, {ratio:.1%})")
    print(f"frames dropped   : {listener.dropped_frames}")
    print(f"speech starts    : {speech_starts}")
    print(f"utterances       : {utterances.qsize()}")

    while not utterances.empty():
        u = utterances.get_nowait()
        peak = float(np.abs(u.audio).max())
        print(f"  {u.duration_s:.2f}s, peak {peak:.3f}")

    ok = True
    if ratio < 0.9:
        print("\nFAIL: frames are not arriving at the expected rate")
        ok = False
    if listener.dropped_frames:
        print("\nFAIL: frames dropped — the VAD thread cannot keep up")
        ok = False
    if ok:
        print("\nOK: capture plumbing healthy")
    return 0 if ok else 1


if __name__ == "__main__":
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    raise SystemExit(asyncio.run(main(secs)))
