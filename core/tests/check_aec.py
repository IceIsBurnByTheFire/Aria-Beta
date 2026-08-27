"""Will speaker mode work in *your* room? Plays out loud and listens.

`e2e_aec.py` proves the algorithm against a simulated room. This measures the real
one: your speakers, your mic, your buffering. It plays a few seconds of Aria's voice
through the speakers while recording the microphone, then reports whether the echo
would have interrupted her and whether cancellation stops it.

Turn the speakers on and set them to the volume you actually use. Don't wear the
headphones — the point is the path through the air.

Run:  uv run --directory . python tests/check_aec.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from e2e_aec import speech_onsets  # noqa: E402

from aria.audio.aec import EchoCanceller, available  # noqa: E402
from aria.audio.playback import _Resampler  # noqa: E402
from aria.config import Config  # noqa: E402
from aria.tts.kokoro_backend import KokoroTTS  # noqa: E402

GREEN, RED, YELLOW, DIM, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m",
)

LINE = ("I was reading about the Roman road network, which is genuinely enormous. "
        "They built over eighty thousand kilometres of it, and quite a lot survives.")


def main() -> int:
    ok, why = available()
    if not ok:
        print(f"{RED}Echo cancellation unavailable: {why}{RESET}")
        return 1

    cfg = Config()
    cfg.validate()
    sr = cfg.audio.sample_rate

    print("Synthesising…")
    reference = _Resampler()(KokoroTTS(cfg.tts).synth(LINE))

    print(f"{YELLOW}Playing out loud now — speakers on, headphones off.{RESET}")
    recorded = sd.playrec(
        reference.reshape(-1, 1),
        samplerate=sr,
        channels=1,
        device=(cfg.audio.input_device, cfg.audio.output_device),
    )
    sd.wait()
    mic = recorded[:, 0].astype(np.float32)

    if dump := os.getenv("ARIA_AEC_DUMP"):
        np.savez(dump, mic=mic, reference=reference)
        print(f"{DIM}  saved recording to {dump}{RESET}")

    # Cross-correlate to find the round trip. Both streams start together at the API,
    # so whatever offset shows up is device buffering plus flight time.
    n = min(len(mic), len(reference), sr * 4)
    corr = np.correlate(mic[:n] - mic[:n].mean(), reference[:n] - reference[:n].mean(), "full")
    delay_samples = int(np.argmax(corr[n - 1 :]))
    delay_ms = delay_samples / sr * 1000

    mic_rms = float(np.sqrt(np.mean(mic**2)))
    ref_rms = float(np.sqrt(np.mean(reference**2)))
    print(f"\n  mic level {mic_rms:.4f} rms, reference {ref_rms:.4f} rms")
    print(f"  measured round trip {delay_ms:.0f} ms "
          f"(filter covers {cfg.barge_in.aec_filter_ms} ms)")

    if mic_rms < 1e-3:
        print(f"\n{YELLOW}The microphone heard almost nothing. Either the speakers were "
              f"off or muted, or the mic is not the default device.{RESET}")
        print("Nothing to cancel, so this test can't tell you anything. Turn the "
              "volume up and run it again.")
        return 1
    # What the filter has to cover is the round trip *minus* the configured
    # compensation, not the round trip itself — that is the whole point of
    # aec_delay_ms. Warn on the residual.
    residual_delay = abs(delay_ms - cfg.barge_in.aec_delay_ms)
    if residual_delay > cfg.barge_in.aec_filter_ms:
        print(f"\n{YELLOW}After compensating {cfg.barge_in.aec_delay_ms} ms, "
              f"{residual_delay:.0f} ms of delay is left over and the filter covers "
              f"{cfg.barge_in.aec_filter_ms} ms. Echo past the filter isn't reduced, "
              f"it's untouched.{RESET}")

    frame = cfg.audio.frame_samples

    def cancel_with(delay_ms: float) -> tuple[int, float]:
        """Cancel using a reference shifted by `delay_ms`, the way the live path does."""
        shift = int(sr * delay_ms / 1000)
        aligned = np.zeros(len(mic), dtype=np.float32)
        take = min(len(reference), len(mic) - shift)
        if take > 0:
            aligned[shift : shift + take] = reference[:take]
        aec = EchoCanceller(frame, sr, cfg.barge_in.aec_filter_ms,
                            cfg.barge_in.aec_speech_margin)
        cleaned = np.zeros_like(mic)
        for i in range(len(mic) // frame):
            s = slice(i * frame, (i + 1) * frame)
            cleaned[s] = aec.process(mic[s], aligned[s])
        erle_ = 10 * np.log10(np.sum(mic**2) / (np.sum(cleaned**2) + 1e-12))
        return speech_onsets(cfg, cleaned), erle_

    before = speech_onsets(cfg, mic)
    configured, erle = cancel_with(cfg.barge_in.aec_delay_ms)
    measured, erle_measured = cancel_with(delay_ms)

    print(f"\n  barge-in triggers on the raw mic : {before}")
    print(f"  after cancelling at the configured {cfg.barge_in.aec_delay_ms} ms: "
          f"{configured}  ({erle:.1f} dB removed)")
    if residual_delay > 32:
        print(f"  after cancelling at the measured {delay_ms:.0f} ms: "
              f"{measured}  ({erle_measured:.1f} dB removed)")
    # Only worth changing if it actually buys something. The estimate wanders tens of
    # milliseconds between runs and the filter absorbs that, so a difference alone is
    # not a reason to retune.
    if configured > 0 and measured == 0:
        print(f"\n{YELLOW}Set barge_in.aec_delay_ms = {delay_ms:.0f}. The configured "
              f"{cfg.barge_in.aec_delay_ms} ms still lets echo trigger barge-in; the "
              f"measured value doesn't.{RESET}")
    after = configured
    print()

    if before == 0:
        print(f"{DIM}The raw echo never triggered the VAD, so your speakers are quiet "
              f"enough that speaker mode was not the blocker anyway.{RESET}")
        return 0
    if after == 0:
        print(f"{GREEN}Speaker mode works here. Run with --speaker-mode and drop the "
              f"headphones.{RESET}")
        return 0
    print(f"{RED}Echo still triggers barge-in after cancellation.{RESET}")
    print("Try, in order: lower the speaker volume, move the mic away from the "
          "speakers, raise barge_in.aec_filter_ms if the round trip above was large, "
          "or fall back to --no-barge-in.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
