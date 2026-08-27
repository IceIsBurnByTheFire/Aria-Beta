"""faster-whisper wrapper.

CTranslate2 rather than torch, which sidesteps the Blackwell CUDA problem entirely.
Note the compute type: int8 variants fail on this GPU with
CUBLAS_STATUS_NOT_SUPPORTED, so float16 is not a preference, it is the only option.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
from faster_whisper import WhisperModel

from ..config import STTConfig

log = logging.getLogger(__name__)

#: Whisper's codes for the languages that come back in Han characters.
_CHINESE = {"zh", "yue"}


@dataclass(frozen=True)
class Transcript:
    """What he said, and what language he said it in.

    A value rather than an attribute on the model, because two transcriptions are in
    flight at once by design — the speculative one started during the pause, and the
    real one if the pause turned out to be mid-sentence. A `last_language` field would
    be whichever finished second.
    """

    text: str
    #: Whisper's guess, or the language it was told. None when nothing was said.
    language: str | None = None

    def __bool__(self) -> bool:
        return bool(self.text)

#: Whisper reliably invents these from silence, breath and fan noise. They arrive with
#: high confidence, so probability thresholds alone do not catch them — measured, asked
#: to transcribe *digital silence* it returns "Thank you." with `no_speech_prob` 0.000
#: and an `avg_logprob` of -0.28, which is **better** than a genuine "Okay." at -0.68.
#: Both of Whisper's own confidence signals point the wrong way here.
#:
#: Every one of these is also an ordinary thing to say to an assistant, which is the
#: whole problem: discarding them unconditionally meant "Okay.", "Thanks.", "Bye." and
#: "So?" were transcribed perfectly and then thrown away, and she simply did nothing.
#: See `_looks_hallucinated` for what decides between the two.
_HALLUCINATIONS = {
    "you", "thank you", "thanks for watching", "thank you for watching",
    "bye", "bye.", "thanks", "okay", "oh", "so", ".", "..", "...",
    "please subscribe", "subtitles by the amara.org community", "the end",
}


def _looks_hallucinated(text: str, voiced_s: float | None, min_voiced_s: float) -> bool:
    """Did Whisper invent this, or did he actually say it?

    Whisper cannot tell us, so Silero does: `voiced_s` is how much of the utterance the
    VAD called speech, and it separates the two cleanly where nothing else does.

        real "Okay." / "Thanks." / "Bye." / "So?"     0.35 - 0.58 s voiced
        silence, hiss, hum, a click, a breath         0.00 s voiced

    So a stock phrase backed by real speech is a real turn, and the same phrase backed
    by nothing is the artefact the set exists for. `voiced_s` of None means the caller
    could not measure it — the old, blunt behaviour, which is the safe default for a
    path that has no VAD behind it.
    """
    stripped = re.sub(r"[^\w\s]", "", text).strip().lower()
    if not stripped:
        return True
    if stripped not in _HALLUCINATIONS:
        return False
    return voiced_s is None or voiced_s < min_voiced_s


def _traditional_converter():
    """Simplified -> Traditional, with Taiwanese vocabulary. None if opencc is absent.

    Degrading rather than raising is the right call here: without it she still hears
    him and still answers, in the right language, in the wrong characters. That is a
    quality problem, not a broken assistant, and it should not stop her starting.
    """
    try:
        import opencc

        return opencc.OpenCC("s2twp").convert
    except Exception as e:  # noqa: BLE001
        log.warning("no Simplified->Traditional conversion (%s); transcripts of "
                    "Chinese will come back in Simplified characters", e)
        return None


class WhisperSTT:
    def __init__(self, cfg: STTConfig):
        self._cfg = cfg
        self._model = WhisperModel(
            cfg.model, device=cfg.device, compute_type=cfg.compute_type
        )
        self._to_traditional = _traditional_converter() if cfg.traditional else None

    def warmup(self) -> None:
        """First call pays PTX JIT for sm_120 — about 16s cold, 0.5s once cached.

        Doing it at startup keeps it out of the first real turn, where the user is
        sitting there waiting.
        """
        self.transcribe(np.zeros(16000, dtype=np.float32), voiced_s=0.0)

    def _run(self, audio: np.ndarray, language: str | None):
        return self._model.transcribe(
            audio,
            # None means detect it, which is what a bilingual run wants: the setting
            # that would otherwise have to be right is the one he changes mid-sentence.
            language=language,
            beam_size=self._cfg.beam_size,
            # We already ran Silero over this; Whisper's own VAD would only re-trim.
            vad_filter=False,
            # Each turn is independent. Conditioning invites the model to continue its
            # own previous hallucination.
            condition_on_previous_text=False,
            # Nothing downstream reads segment timings — the endpointer already decided
            # where the turn starts and ends — and generating them is decoded tokens
            # like any other. Measured 304 -> 282 ms median with identical text, which
            # is 7% off the term that has to fit inside a 350 ms window.
            without_timestamps=True,
        )

    def transcribe(
        self, audio: np.ndarray, voiced_s: float | None = None
    ) -> Transcript:
        segments, info = self._run(audio, self._cfg.language)
        language = getattr(info, "language", None)

        allowed = self._cfg.allowed
        if allowed and language not in allowed:
            # She speaks two languages and Whisper picks from ninety-nine, so a third
            # one is a misdetection by construction — and the reply it produces is
            # fluent, on-persona and useless ("you're speaking Korean, I can't help").
            # Redo it as the best language she actually has. Costs a second pass only
            # on turns that were already going to be wrong.
            ranked = getattr(info, "all_language_probs", None) or []
            best = next((code for code, _ in ranked if code in allowed), allowed[0])
            log.debug("detected %s, which she does not speak; redoing as %s",
                      language, best)
            segments, info = self._run(audio, best)
            language = best

        parts = [s.text for s in segments if s.no_speech_prob < 0.6]
        text = " ".join(p.strip() for p in parts).strip()

        if _looks_hallucinated(text, voiced_s, self._cfg.min_voiced_ms / 1000):
            log.debug("discarded likely hallucination: %r (%.2fs voiced)",
                      text, voiced_s if voiced_s is not None else -1.0)
            return Transcript("", language)

        if self._to_traditional and language in _CHINESE:
            # ~0.1 ms, and idempotent — text that is already Traditional comes back
            # unchanged, so this is safe on every Chinese turn rather than on a guess
            # about which ones need it. It also fixes vocabulary, not just glyphs
            # (内存 -> 記憶體), which is the half a character map would miss.
            text = self._to_traditional(text)
        return Transcript(text, language)
