"""Fetch the model files that are too large to commit.

Whisper downloads itself through huggingface on first use; these three do not.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

from .config import MODELS_DIR

KOKORO = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
SILERO = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data"

FILES = {
    "silero_vad.onnx": f"{SILERO}/silero_vad.onnx",
    "kokoro-v1.0.onnx": f"{KOKORO}/kokoro-v1.0.onnx",
    "voices-v1.0.bin": f"{KOKORO}/voices-v1.0.bin",
}


def _progress(done: int, block: int, total: int) -> None:
    if total > 0:
        pct = min(100, done * block * 100 // total)
        print(f"\r    {pct:3d}%  ({total / 1e6:.0f} MB)", end="", flush=True)


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dest = MODELS_DIR / name
        if dest.exists():
            print(f"  {name} — already present")
            continue
        print(f"  {name}")
        try:
            urllib.request.urlretrieve(url, dest, reporthook=_progress)
            print()
        except Exception as e:
            dest.unlink(missing_ok=True)
            print(f"\n  FAILED: {e}", file=sys.stderr)
            return 1
    print(f"\nModels in {MODELS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
