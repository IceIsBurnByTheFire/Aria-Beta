"""Emotion markers: extraction, and not mangling ordinary text."""

from aria.chunker import SentenceChunker
from aria.emotion import extract, has_open_marker


def test_extracts_and_strips():
    text, emotions = extract("[happy] That actually worked.")
    assert text == "That actually worked."
    assert emotions == ["happy"]


def test_lowercases_and_keeps_order():
    _, emotions = extract("[Happy] one. [SAD] two.")
    assert emotions == ["happy", "sad"]


def test_underscored_names_survive():
    text, emotions = extract("[no_eye_highlight] Watch this.")
    assert emotions == ["no_eye_highlight"]
    assert text == "Watch this."


def test_text_without_markers_is_untouched():
    assert extract("Nothing to see here.") == ("Nothing to see here.", [])


def test_ordinary_brackets_are_left_alone():
    # A marker is letters only — prose and citations must not be eaten from the speech.
    for s in ["See item [3] on the list.", "The array is [1, 2, 3].", "Press [ to open."]:
        text, emotions = extract(s)
        assert emotions == [], s
        assert text == s, s


def test_marker_only_chunk_yields_empty_text():
    text, emotions = extract("[thinking]")
    assert text == ""
    assert emotions == ["thinking"]


HAVE = ["happy", "shy", "sad", "surprised", "angry"]


def test_a_marker_that_lost_its_opening_bracket():
    # Seen in the wild, posted to Discord: "...whatever sauce you have. shy] Tell me..."
    # The face never moved and the word arrived as text. On the voice path the same
    # thing is read aloud mid-sentence.
    text, emotions = extract("You have. shy] Tell me which one.", HAVE)
    assert "]" not in text and "shy" not in text, text
    assert emotions == ["shy"], "and she should still make the face she meant to"


def test_a_marker_that_lost_its_closing_bracket():
    text, emotions = extract("[happy That actually worked.", HAVE)
    assert text == "That actually worked."
    assert emotions == ["happy"]


def test_debris_repair_keeps_them_in_order():
    _, emotions = extract("[happy] one. sad] two. [shy] three.", HAVE)
    assert emotions == ["happy", "sad", "shy"]


def test_an_emotion_word_in_ordinary_prose_is_not_a_marker():
    # The repair is anchored to a bracket for exactly this reason. Eating the word
    # would be a far worse bug than the one it fixes.
    for s in ["I'm happy for you.", "That's sad.", "he was angry about it"]:
        assert extract(s, HAVE) == (s, []), s


def test_debris_repair_needs_the_vocabulary():
    # No character attached means no closed set to be confident about, so a bare
    # "shy]" is left alone rather than guessed at.
    assert extract("You have. shy] Tell me.") == ("You have. shy] Tell me.", [])


def test_a_word_outside_the_vocabulary_is_left_alone():
    assert extract("the total was 12] units", HAVE) == ("the total was 12] units", [])


def test_a_marker_before_punctuation_leaves_no_orphan_space():
    # Seen live on Gemini, which closes clauses with the marker rather than opening
    # them: "it's finally over [happy]." posted as "it's finally over ." Invisible in
    # speech; in text it reads as a typo in every message.
    text, emotions = extract("It's finally over [happy].", HAVE)
    assert text == "It's finally over."
    assert emotions == ["happy"]


def test_ordinary_spacing_is_left_alone():
    assert extract("Wait. Really? Yes.", HAVE) == ("Wait. Really? Yes.", [])


def test_open_marker_detection():
    assert has_open_marker("Well, [hap")
    assert not has_open_marker("Well, [happy]")
    assert not has_open_marker("no brackets at all")
    assert has_open_marker("closed [one] then [op")


def test_chunker_never_splits_a_marker():
    """A marker straddling two chunks is both unrecognisable and audible."""
    c = SentenceChunker()
    out = []
    for token in ["Sure. ", "[hap", "py] ", "It worked."]:
        out += c.push(token)
    if tail := c.flush():
        out.append(tail)

    joined = " ".join(out)
    assert "[happy]" in joined, joined
    # No chunk may contain a half-marker.
    for chunk in out:
        assert not has_open_marker(chunk), chunk


def test_emotion_survives_the_chunk_it_opens():
    c = SentenceChunker()
    chunks = []
    for token in "[happy] That worked. [sad] But not for long. ".split(" "):
        chunks += c.push(token + " ")
    if tail := c.flush():
        chunks.append(tail)

    pairs = [extract(ch) for ch in chunks]
    emotions = [e for _, es in pairs for e in es]
    assert emotions == ["happy", "sad"]
