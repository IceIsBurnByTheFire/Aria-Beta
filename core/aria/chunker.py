"""Splitting a token stream into speakable chunks.

The asymmetry here is deliberate. The *first* chunk is cut as early as a boundary
allows, because time-to-first-audio is what a user actually perceives as
responsiveness. Later chunks are held to whole sentences, where the extra context gives
the synthesiser better prosody and nobody is waiting anyway — the first chunk is still
playing.
"""

from __future__ import annotations

import re

from .emotion import has_open_marker

#: Two alternatives, and the second is not just extra characters.
#:
#: Western punctuation needs the trailing-space lookahead, or "3.5" and "Mr." split. CJK
#: has **no spaces at all**, so the same lookahead never matches and the whole reply
#: arrives as one chunk — measured: four chunks for a line of English, one for the same
#: line of Chinese. That is the streaming design collapsing silently. Nothing errors; she
#: just synthesises the entire reply before saying a word, and the latency budget goes
#: with it. CJK terminators therefore match on their own, which is safe because 。！？ are
#: unambiguous sentence enders in a way "." is not.
_SENT_END = re.compile(
    r"""[.!?…]+["'")\]]*(?=\s|$)"""     # western, needs the space
    r"""|[。！？!?…]+[」』’”）)]*"""      # CJK, does not
)
_CLAUSE_END = re.compile(r"[,;:—](?=\s)|[，、；：]")

#: Enough CJK in the buffer to treat it as CJK. A reply that is mostly Chinese with an
#: English package name in it is still Chinese.
_CJK = re.compile(
    r"[぀-ヿ㐀-䶿一-鿿豈-﫿ｦ-ﾟ]"
)


def is_cjk(text: str) -> bool:
    """True when the text is dense enough in CJK to need CJK cut lengths.

    One CJK character is roughly one syllable, where an English character is roughly a
    fifth of one — so the same character count is about five times as much speech. The
    40-character first chunk tuned for English is ~2.5 s of audio and ~10 s of Mandarin,
    which would put most of a reply in the chunk that is supposed to buy time for the
    next one.
    """
    if not text:
        return False
    return sum(bool(_CJK.match(c)) for c in text) / len(text) > 0.2

#: A period after one of these is an abbreviation, not the end of a thought.
_ABBREV = {
    "mr", "mrs", "ms", "dr", "prof", "st", "sr", "jr", "vs", "etc",
    "eg", "ie", "approx", "no", "fig", "inc", "ltd", "dept",
}

_MARKDOWN = re.compile(r"[*_`#>|]+")
#: Public because the Discord path needs the same question answered the other way round:
#: what must be stripped before speech is exactly what counts as "she already picked an
#: emoji herself", and two definitions of that would drift.
EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF←-⇿️]+"
)
#: Anything still in brackets by the time speech is being cleaned. `emotion.extract`
#: takes the real markers out first and is deliberately narrow — single words only, so
#: it can't silently eat "[see below]" — which leaves everything else to be read aloud
#: as "bracket two days bracket". Once she had the date in her prompt she started
#: inventing them: "I'll remember it for sure. [two days] It feels like…". There is no
#: such thing as a bracket a synthesiser should say, so this takes the lot.
_BRACKETED = re.compile(r"\[[^\]]{0,60}\]")
#: What `emotion.extract` will take out later. This runs *first*, so stripping these
#: here would leave her face frozen while the words still sounded happy — M4 dying
#: silently to a one-line fix for a TTS artefact.
_MARKER_LIKE = re.compile(r"^\[\s*[A-Za-z][A-Za-z_]{1,23}\s*\]$")


def clean_for_speech(text: str) -> str:
    """Strip anything a synthesiser would read aloud as punctuation noise.

    The persona tells the model not to emit markdown, and it mostly obeys — but "mostly"
    means hearing "asterisk asterisk" a few times an hour, which is enough to matter.
    """
    text = _BRACKETED.sub(
        lambda m: m.group(0) if _MARKER_LIKE.match(m.group(0)) else "", text
    )
    text = _MARKDOWN.sub("", text)
    text = EMOJI.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_abbreviation(buf: str, dot_index: int) -> bool:
    word = re.search(r"(\w+)$", buf[:dot_index])
    return bool(word) and word.group(1).lower() in _ABBREV


def _sentence_cut(buf: str, min_index: int = 0) -> int:
    for m in _SENT_END.finditer(buf):
        if m.end() < min_index:
            continue
        if buf[m.start()] == "." and _is_abbreviation(buf, m.start()):
            continue
        return m.end()
    return -1


def _clause_cut(buf: str, min_buffer: int) -> int:
    """Earliest clause boundary, but only once the buffer proves more is coming.

    The gate is on buffer length, not on boundary position. "Well," is an excellent
    first chunk *because* the sentence continues past it — but only once we have seen
    enough text to know it does. Gating on position instead would swallow the early
    boundary and emit a long first chunk, which is the opposite of the point.
    """
    if len(buf) < min_buffer:
        return -1
    m = _CLAUSE_END.search(buf)
    return m.end() if m else -1


def _space_cut(buf: str, at: int) -> int:
    """Last word boundary at or before `at`, so we never split mid-word."""
    space = buf.rfind(" ", 0, at)
    return space + 1 if space > 0 else at


class SentenceChunker:
    def __init__(
        self,
        first_clause_after: int = 12,
        # Kokoro costs roughly 160 ms fixed plus 0.14 ms per ms of audio produced, so
        # first-chunk synthesis time is dominated by chunk *length*. 40 chars lands
        # around 300 ms; 80 chars measured at 600-1100 ms and made TTS the single
        # largest term in the budget. The first chunk only has to buy enough runway to
        # synthesise the second, and ~2 s of audio is plenty.
        first_max: int = 40,
        max_chars: int = 200,
    ):
        self._first_clause_after = first_clause_after
        self._first_max = first_max
        self._max_chars = max_chars
        self._buf = ""
        self._emitted_first = False

    def push(self, token: str) -> list[str]:
        """Feed a streamed fragment; get back any chunks that are ready to speak."""
        self._buf += token
        chunks: list[str] = []
        while (cut := self._find_cut()) > 0:
            chunk = clean_for_speech(self._buf[:cut])
            self._buf = self._buf[cut:].lstrip()
            self._emitted_first = True
            if chunk:
                chunks.append(chunk)
        return chunks

    def flush(self) -> str:
        """Whatever is left when the stream ends."""
        chunk, self._buf = clean_for_speech(self._buf), ""
        if chunk:
            self._emitted_first = True
        return chunk

    def _limits(self) -> tuple[int, int, int]:
        """(clause-after, first-max, max) for whatever script this reply is in.

        Divided rather than configured, because the ratio is a property of the writing
        system and not a preference: a CJK character is about a syllable where a Latin
        one is about a fifth of one.
        """
        if is_cjk(self._buf):
            return (
                max(4, self._first_clause_after // 3),
                max(8, self._first_max // 3),
                max(24, self._max_chars // 3),
            )
        return self._first_clause_after, self._first_max, self._max_chars

    def _find_cut(self) -> int:
        buf = self._buf
        if not buf:
            return -1

        # Never cut while an `[emotion` marker is still being streamed — half a marker
        # is unrecognisable to the parser and audible to the user.
        if has_open_marker(buf):
            return -1

        clause_after, first_max, max_chars = self._limits()

        # A complete sentence always wins, however short — "Sure." should go out now.
        if (cut := _sentence_cut(buf)) > 0:
            return cut

        if not self._emitted_first:
            if (cut := _clause_cut(buf, clause_after)) > 0:
                return cut
            if len(buf) >= first_max:
                return _space_cut(buf, first_max)
            return -1

        if len(buf) >= max_chars:
            return _space_cut(buf, max_chars)
        return -1
