"""Chunker behaviour — the rules that decide time-to-first-audio."""

from aria.chunker import SentenceChunker, clean_for_speech


def run(text: str, **kw) -> list[str]:
    """Feed `text` word by word, the way tokens actually arrive."""
    c = SentenceChunker(**kw)
    out = []
    for word in text.split(" "):
        out += c.push(word + " ")
    if tail := c.flush():
        out.append(tail)
    return out


def test_short_sentence_goes_out_whole():
    assert run("Sure.") == ["Sure."]


def test_complete_short_sentence_beats_the_clause_minimum():
    # "Hey there!" is under first_clause_min but it is a whole sentence, so it must
    # not be held back waiting for more text.
    assert run("Hey there! What's up?") == ["Hey there!", "What's up?"]


def test_long_opening_sentence_splits_at_a_clause():
    chunks = run("Well, that error is coming from your config file, not the code.")
    assert chunks[0] == "Well,"
    assert len(chunks) > 1
    assert "".join(chunks).replace(" ", "") == \
        "Well,thaterroriscomingfromyourconfigfile,notthecode."


def test_later_chunks_are_not_split_at_clauses():
    # Once the first chunk is out nobody is waiting, so hold whole sentences for
    # better prosody.
    chunks = run("Yes. That error, which is annoying, comes from the config.")
    assert chunks[0] == "Yes."
    assert chunks[1] == "That error, which is annoying, comes from the config."


def test_abbreviations_do_not_end_a_sentence():
    assert run("Mr. Smith went home.") == ["Mr. Smith went home."]


def test_decimals_do_not_end_a_sentence():
    assert run("It costs 3.5 dollars total.") == ["It costs 3.5 dollars total."]


def test_runaway_text_is_cut_at_a_word_boundary():
    chunks = run("word " * 60)
    assert len(chunks) > 1
    assert all(not c.startswith(" ") for c in chunks)
    for c in chunks:
        assert "wordword" not in c  # never split mid-word


def test_flush_returns_unterminated_remainder():
    c = SentenceChunker()
    assert c.push("no punctuation here") == []
    assert c.flush() == "no punctuation here"
    assert c.flush() == ""


def test_markdown_and_emoji_are_stripped():
    assert clean_for_speech("**bold** and `code` and # head") == "bold and code and head"
    assert clean_for_speech("nice 😀 work") == "nice work"


def test_nothing_is_lost_across_a_stream():
    text = "First one. Second, with a clause, here. Third and last!"
    joined = " ".join(run(text))
    assert joined.replace(" ", "") == text.replace(" ", "")


# --- CJK -------------------------------------------------------------------
# Measured before this worked: the same sentence gave four chunks in English and one in
# Chinese. One chunk means the whole reply is synthesised before she says a word, so the
# streaming design stops working with nothing to show for it.
def chunks_of(text: str) -> list[str]:
    c = SentenceChunker()
    out: list[str] = []
    for ch in text:
        out += c.push(ch)
    if tail := c.flush():
        out.append(tail)
    return out


def test_chinese_splits_into_sentences():
    out = chunks_of("你好，我是艾莉亞。今天過得怎麼樣？我一直在這裡等你。")
    assert len(out) >= 3, out
    assert out[-1].endswith("。")


def test_a_cjk_terminator_needs_no_trailing_space():
    # The reason adding 。！？ to the character class alone would not have worked: the
    # western pattern requires whitespace after the mark, and Chinese never has any.
    assert len(chunks_of("好。壞。")) == 2


def test_cjk_clause_commas_open_the_first_chunk_early():
    # Same trick as English: cut at the first comma so time-to-first-audio is short.
    out = chunks_of("你好，我一直在這裡等你，真的很久了")
    assert out[0] == "你好，"


def test_cjk_chunks_are_shorter_than_english_ones():
    # A CJK character is about a syllable; a Latin one about a fifth of one. Equal
    # character counts would make the first Chinese chunk roughly four times the audio.
    long_zh = "".join("這是一個很長的句子沒有標點符號" for _ in range(6))
    assert len(chunks_of(long_zh)[0]) < 20


def test_english_is_unchanged():
    out = chunks_of("Hello, I am Aria. How was your day? I have been right here.")
    assert out[0] == "Hello,"
    assert "I am Aria." in out


def test_mixed_script_still_cuts_on_cjk_punctuation():
    out = chunks_of("我在用 Rust 寫這個。你覺得怎麼樣？")
    assert len(out) == 2
    assert "Rust" in out[0]
