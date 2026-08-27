"""Kokoro-82M over onnxruntime.

Runs on CPU at roughly RTF 0.2-0.3 on this machine, which leaves the GPU entirely to
Whisper and the LLM. That division turns out to be the right one: TTS is the only
component here small enough that CPU is genuinely fine.

It also carries the `[emotion]` markers into the delivery — see `EMOTION_VOICE`.
"""

from __future__ import annotations

import logging

import numpy as np
from kokoro_onnx import Kokoro

from ..chunker import is_cjk
from ..config import CJK_TTS_LANGS, TTSConfig
from .chinese import ChinesePhonemiser

log = logging.getLogger(__name__)

#: emotion -> (rate, pitch, gain). Rate and pitch are multipliers, gain is linear.
#:
#: Kokoro exposes speed and nothing else — no pitch, no energy. But speed and
#: resampling together give both: resampling by `p` raises pitch *and* rate by `p`, so
#: asking Kokoro for `rate / pitch` first leaves the rate correct once the resample has
#: had its way. That is the whole trick, and it is why pitch is free here.
#:
#: The numbers are deliberately small. Past about eight percent the formants shift with
#: the pitch and she stops sounding like a person and starts sounding like a cartoon
#: chipmunk — the effect has to be felt rather than heard.
EMOTION_VOICE: dict[str, tuple[float, float, float]] = {
    "happy": (1.05, 1.035, 1.06),
    "surprised": (1.09, 1.06, 1.12),
    "shy": (0.97, 1.02, 0.80),      # quieter and a touch higher reads as timid
    "sad": (0.93, 0.975, 0.85),
    "angry": (1.04, 0.98, 1.15),
    "thinking": (0.95, 1.0, 0.95),
}


class KokoroTTS:
    def __init__(self, cfg: TTSConfig):
        self._cfg = cfg
        self._kokoro = Kokoro(str(cfg.model_path), str(cfg.voices_path))
        self.sample_rate = cfg.sample_rate
        self._chinese = ChinesePhonemiser(self._kokoro.tokenizer)

    def voices(self) -> list[str]:
        return sorted(self._kokoro.get_voices())

    def warmup(self) -> None:
        """Pay every first-call cost here rather than inside the first reply.

        Both halves, when she can speak both: jieba's prefix dictionary is ~0.9 s on
        first use, and it would otherwise land on whichever sentence happens to be the
        first Chinese one — a pause in the middle of a conversation rather than during
        a loading line nobody is waiting on.
        """
        self.synth("Ready.")
        if self._cfg.lang == "zh" or self._cfg.warm_alt:
            self._chinese.load()
            self.synth("好。")

    def _for(self, text: str) -> tuple[str, str]:
        """(voice, phonemiser) for this chunk.

        The chunker's script test, reused deliberately: the thing that decided where to
        cut this chunk should be the thing that decides how to say it, or a reply can be
        cut as Chinese and spoken as English.
        """
        primary = (self._cfg.voice, self._cfg.lang)
        if not self._cfg.alt_voice:
            return primary
        if is_cjk(text) == (self._cfg.lang in CJK_TTS_LANGS):
            return primary
        return self._cfg.alt_voice, self._cfg.alt_lang

    def synth(self, text: str, emotion: str | None = None) -> np.ndarray:
        """Synthesise one chunk, coloured by `emotion`. Blocking — call in a thread."""
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.float32)

        rate, pitch, gain = 1.0, 1.0, 1.0
        if self._cfg.emotion_voice and emotion:
            rate, pitch, gain = EMOTION_VOICE.get(emotion, (1.0, 1.0, 1.0))

        voice, lang = self._for(text)
        try:
            # `zh` is not an espeak code — it means "phonemise this ourselves", because
            # these voices were not trained on espeak's Mandarin. See `chinese.py`.
            spoken, phonemes = (
                (self._chinese(text), True) if lang == "zh" else (text, False)
            )
            samples, _ = self._kokoro.create(
                spoken,
                voice=voice,
                # Ask for a rate that the pitch resample will then correct back.
                speed=self._cfg.speed * rate / pitch,
                lang=lang,
                is_phonemes=phonemes,
            )
        except Exception as e:
            # A malformed chunk must not kill the turn — skip it and keep talking.
            log.warning("TTS failed for %r: %s", text[:60], e)
            return np.zeros(0, dtype=np.float32)

        audio = np.asarray(samples, dtype=np.float32)
        if pitch != 1.0 and len(audio):
            audio = self._shift_pitch(audio, pitch)
        if gain != 1.0:
            # Clip rather than let a loud emotion wrap around into noise.
            audio = np.clip(audio * gain, -1.0, 1.0)
        return audio

    @staticmethod
    def _shift_pitch(audio: np.ndarray, factor: float) -> np.ndarray:
        """Resample by `factor`, which moves pitch and duration together.

        Linear interpolation is enough at these factors. It would alias audibly on a
        big shift, but nothing here moves more than six percent.
        """
        n = max(1, int(round(len(audio) / factor)))
        return np.interp(
            np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio
        ).astype(np.float32)
