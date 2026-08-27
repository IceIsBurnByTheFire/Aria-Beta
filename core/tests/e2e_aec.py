"""M6: does Aria interrupt herself on open speakers?

ERLE in dB is not the acceptance criterion — the VAD is. What breaks speaker mode is
the endpointer firing on Aria's own voice, so this drives real Kokoro speech through a
simulated room into the real Silero endpointer and asks three questions:

  1. Without cancellation, does the echo trigger a turn?   (must be yes, or the test
     proves nothing — that failure is the entire reason M6 exists)
  2. With cancellation, does it stay quiet?                (must be yes)
  3. With cancellation, does *real* speech still get through? (must be yes, or we have
     traded self-interruption for deafness)

No speakers or microphone required: the room is synthetic and deterministic. For the
real acoustic path on real hardware, run `tests/check_aec.py` with speakers on.

Run:  uv run --directory . python tests/e2e_aec.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _harness import resample  # noqa: E402

from aria.audio.aec import EchoCanceller, available  # noqa: E402
from aria.audio.playback import _Resampler  # noqa: E402
from aria.audio.vad import Endpointer, SileroVAD  # noqa: E402
from aria.config import Config  # noqa: E402
from aria.tts.kokoro_backend import KokoroTTS  # noqa: E402

RED, GREEN, YELLOW, RESET = "\033[31m", "\033[32m", "\033[33m", "\033[0m"
results: list[tuple[bool, str]] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    results.append((ok, label))
    mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"  [{mark}] {label}" + (f"  — {detail}" if detail else ""))


def speech_onsets(cfg: Config, audio: np.ndarray) -> int:
    """How many times the endpointer decides speech has started.

    This — not a completed utterance — is what barge-in listens to: `_on_speech_start`
    fires on the transition into speech and calls `interrupt()` immediately. Counting
    completed utterances would miss the bug entirely, since Aria's own voice is
    continuous and never endpoints while she is still talking.
    """
    ep = Endpointer(
        SileroVAD(cfg.vad.model_path, cfg.audio.sample_rate),
        cfg.vad,
        frame_samples=cfg.audio.frame_samples,
        sample_rate=cfg.audio.sample_rate,
    )
    n = cfg.audio.frame_samples
    onsets, was_speaking = 0, False
    for i in range(0, len(audio) - n + 1, n):
        ep.feed(audio[i : i + n])
        if ep.in_speech and not was_speaking:
            onsets += 1
        was_speaking = ep.in_speech
    return onsets


def room(reference: np.ndarray, sr: int, delay_ms: float, gain: float) -> np.ndarray:
    """What the mic hears: the speaker, late, quieter, plus two wall reflections.

    Deliberately harsher than a desk setup. A canceller that survives 0.7 gain at
    120 ms will survive a laptop speaker 40 cm away.
    """
    delay = int(sr * delay_ms / 1000)
    echo = np.zeros(len(reference) + delay + 2000, dtype=np.float32)
    for offset, g in ((0, 1.0), (313, 0.4), (911, 0.2)):
        d = delay + offset
        echo[d : d + len(reference)] += reference * g
    return (echo[: len(reference)] * gain).astype(np.float32)


def cancel(mic: np.ndarray, reference: np.ndarray, cfg: Config) -> tuple[np.ndarray, float]:
    """Run the real canceller frame by frame. Returns (residual, ms per frame)."""
    n = cfg.audio.frame_samples
    aec = EchoCanceller(n, cfg.audio.sample_rate, cfg.barge_in.aec_filter_ms)
    out = np.zeros_like(mic)
    frames = len(mic) // n
    t0 = time.perf_counter()
    for i in range(frames):
        s = slice(i * n, (i + 1) * n)
        out[s] = aec.process(mic[s], reference[s])
    per_frame = (time.perf_counter() - t0) / max(frames, 1) * 1000
    aec.close()
    return out, per_frame


def main() -> int:
    ok, why = available()
    if not ok:
        print(f"{YELLOW}Echo cancellation unavailable: {why}{RESET}")
        print("Speaker mode cannot be tested on this machine.")
        return 1

    cfg = Config()
    cfg.validate()
    sr = cfg.audio.sample_rate

    print("Synthesising Aria's voice and a user interrupting her…")
    tts = KokoroTTS(cfg.tts)
    aria_24k = tts.synth(
        "I was reading about the Roman road network, which is genuinely enormous. "
        "They built over eighty thousand kilometres of it, and quite a lot survives."
    )
    user_24k = tts.synth("Hey, stop for a second.")

    # The reference is what Playback would hand the canceller: the same 24 kHz audio,
    # through the same resampler, so the test exercises that path rather than a
    # convenient shortcut.
    reference = _Resampler()(aria_24k)
    user = resample(user_24k, cfg.tts.sample_rate, sr)

    echo = room(reference, sr, delay_ms=120, gain=0.7)
    print(f"  reference {len(reference) / sr:.1f}s, echo at 120 ms / 0.7 gain, "
          f"filter {cfg.barge_in.aec_filter_ms} ms\n")

    # --- 1. the bug M6 exists to fix ------------------------------------------
    print("--- without cancellation ---")
    raw_onsets = speech_onsets(cfg, echo)
    check(raw_onsets > 0,
          "echo alone triggers barge-in (the self-interruption bug)",
          f"{raw_onsets} onsets — each one cuts her off")
    if raw_onsets == 0:
        print(f"{YELLOW}  The simulated echo never triggers the VAD, so the rest of "
              f"this test proves nothing.{RESET}")

    # --- 2. echo alone must not trigger ---------------------------------------
    print("\n--- with cancellation, Aria talking alone ---")
    residual, per_frame = cancel(echo, reference, cfg)
    erle = 10 * np.log10(np.sum(echo**2) / (np.sum(residual**2) + 1e-12))
    cancelled_onsets = speech_onsets(cfg, residual)
    check(cancelled_onsets == 0,
          "echo alone no longer triggers barge-in",
          f"{cancelled_onsets} onsets, {erle:.1f} dB removed")

    budget = cfg.audio.frame_samples / sr * 1000
    check(per_frame < budget * 0.25,
          "cancellation fits the frame budget",
          f"{per_frame:.2f} ms per {budget:.0f} ms frame")

    # --- 3. the user must still get through -----------------------------------
    print("\n--- with cancellation, user speaking over her ---")
    # Drop the user's voice in partway, where the filter has converged — which is
    # also when real barge-in happens.
    start = int(sr * 2.0)
    mixed = echo.copy()
    end = min(start + len(user), len(mixed))
    mixed[start:end] += user[: end - start]

    residual, _ = cancel(mixed, reference, cfg)
    check(speech_onsets(cfg, residual) > 0,
          "real speech over the echo still triggers barge-in")

    # --- 4. it must hold across rooms, not just the tuned one -----------------
    # 250 ms is the Bluetooth case: a delay past the filter length isn't degraded,
    # it's uncancelled, and this is what catches a filter set too short.
    print("\n--- across echo delays and levels ---")
    for delay_ms, gain in ((20, 0.9), (120, 0.5), (250, 0.5)):
        e = room(reference, sr, delay_ms, gain)
        # Compensate the bulk delay, which is what `Playback.reference_for` does live.
        # Testing the uncompensated path would measure a configuration that never
        # ships — and it is genuinely marginal: at 250 ms uncompensated the suppressor
        # is tight enough that her *timbre* decides the outcome, and switching her
        # voice from af_heart to af_bella turned this case from silent-and-hearing
        # into deaf. Compensated, both voices pass.
        aligned = np.zeros(len(e), dtype=np.float32)
        shift = int(sr * delay_ms / 1000)
        take = min(len(reference), len(e) - shift)
        aligned[shift : shift + take] = reference[:take]
        alone, _ = cancel(e, aligned, cfg)

        m = e.copy()
        stop = min(start + len(user), len(m))
        m[start:stop] += user[: stop - start]
        both, _ = cancel(m, aligned, cfg)

        quiet = speech_onsets(cfg, alone) == 0
        heard = speech_onsets(cfg, both) > 0
        check(quiet and heard, f"{delay_ms} ms delay at {gain} gain",
              "silent alone, hears the user" if quiet and heard
              else ("self-interrupts" if not quiet else "deaf to the user"))

    failed = [label for ok_, label in results if not ok_]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    for label in failed:
        print(f"  {RED}FAILED{RESET}: {label}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
