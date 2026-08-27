"""What gets written to history when a reply is cut off."""

from aria.spoken import WrittenChunk, spoken_prefix

# Three chunks: 0-1s, 1-3s, 3-4s of audio.
REPLY = [
    WrittenChunk("Sure thing.", 1.0),
    WrittenChunk("The capital of France is Paris.", 3.0),
    WrittenChunk("Anything else?", 4.0),
]


def test_nothing_played_is_nothing_said():
    assert spoken_prefix(REPLY, 0.0) == ""
    assert spoken_prefix(REPLY, -1.0) == ""


def test_everything_played_is_everything_said():
    assert spoken_prefix(REPLY, 4.0) == \
        "Sure thing. The capital of France is Paris. Anything else?"


def test_overrun_does_not_invent_text():
    assert spoken_prefix(REPLY, 99.0) == spoken_prefix(REPLY, 4.0)


def test_exact_chunk_boundary_keeps_that_chunk():
    assert spoken_prefix(REPLY, 1.0) == "Sure thing."
    assert spoken_prefix(REPLY, 3.0) == "Sure thing. The capital of France is Paris."


def test_cut_midway_through_a_chunk_truncates_it():
    # 2.0s = 1.0s into a 2.0s chunk, so half of its six words.
    assert spoken_prefix(REPLY, 2.0) == "Sure thing. The capital of"


def test_queued_but_unplayed_chunks_are_dropped():
    # The whole reply was generated and written, but playback stopped at 1.2s. The
    # third chunk must not appear — Aria never said it.
    result = spoken_prefix(REPLY, 1.2)
    assert "Anything else?" not in result
    assert result.startswith("Sure thing.")


def test_empty_reply():
    assert spoken_prefix([], 5.0) == ""


def test_barely_started_chunk_contributes_nothing():
    # 1.05s is 2.5% into the second chunk — not a whole word.
    assert spoken_prefix(REPLY, 1.05) == "Sure thing."
