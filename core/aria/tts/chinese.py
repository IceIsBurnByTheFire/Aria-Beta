"""Phonemising Mandarin for Kokoro, which is not what `kokoro-onnx` does by default.

**Kokoro's Chinese voices are trained on misaki phonemes, and kokoro-onnx phonemises
with espeak-ng.** Nothing errors. `lang="cmn"` is accepted, audio comes out, it is
recognisably a female Mandarin voice, and it is wrong — wrong enough that Whisper, which
is very good at Mandarin, cannot read most of it back.

Measured, same voice and same model, scored by transcribing her own speech and comparing
it to what she was asked to say:

    espeak `cmn`   0.46      早安 -> 相安,  聽起來很累 -> 听起来人类
    misaki `zh`    0.73      早安 -> 早安,  聽起來很累 -> 听起来很累

The remaining gap is mostly the scorer, not the speech: Whisper answers in Simplified,
so 這/这 and 麼/么 count as errors against a Traditional source even when the
transcription is perfect. What matters is the shape of the failures — espeak produces
*different words*, which is the difference between an accent and a wrong sentence.

**Latin runs are cut out and phonemised separately**, because a Chinese chunk is very
often not purely Chinese: package names, commands and file names have no Chinese form
and the persona explicitly tells her to leave them alone. misaki passes them through as
literal Latin text, which then reaches Kokoro's tokeniser as if it were IPA — "Rust"
becomes whatever /R/, /u/, /s/, /t/ happen to mean. espeak is already loaded for the
English voice, so the fix costs nothing but the split.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

#: A run of Latin script to hand to espeak rather than to misaki. Deliberately starts
#: on a letter: a bare number is better read by misaki, which says it in Chinese.
_LATIN_RUN = re.compile(r"[A-Za-z][A-Za-z0-9'’._+-]*")


class ChinesePhonemiser:
    """Han text to the phonemes Kokoro's `zf_*`/`zm_*` voices were trained on.

    Lazily built, because misaki pulls in jieba and jieba spends ~0.9 s building its
    prefix dictionary on first use. That is fine at warmup and unacceptable in the
    middle of the first reply, so `KokoroTTS.warmup` reaches in and pays it up front.
    """

    def __init__(self, espeak):
        #: The tokeniser Kokoro already owns, for the Latin runs. Passed in rather than
        #: constructed so there is one espeak in the process, not two.
        self._espeak = espeak
        self._g2p = None

    def load(self) -> None:
        if self._g2p is None:
            from misaki import zh

            self._g2p = zh.ZHG2P()

    def __call__(self, text: str) -> str:
        """Phonemes for one chunk. Falls back to espeak rather than raising.

        A phonemiser that throws would take out the turn. Losing the tone marks on one
        sentence is much better than losing the sentence.
        """
        self.load()
        out: list[str] = []
        at = 0
        for m in _LATIN_RUN.finditer(text):
            out.append(self._han(text[at : m.start()]))
            out.append(self._latin(m.group(0)))
            at = m.end()
        out.append(self._han(text[at:]))
        return " ".join(p for p in out if p).strip()

    def _han(self, text: str) -> str:
        if not text.strip():
            return ""
        try:
            phonemes, _ = self._g2p(text)
            return phonemes.strip()
        except Exception as e:  # noqa: BLE001 — one chunk, never the turn
            log.warning("misaki failed for %r: %s", text[:40], e)
            return self._espeak.phonemize(text, "cmn").strip()

    def _latin(self, word: str) -> str:
        try:
            return self._espeak.phonemize(word, "en-us").strip()
        except Exception as e:  # noqa: BLE001
            log.warning("espeak failed for %r: %s", word, e)
            return ""
