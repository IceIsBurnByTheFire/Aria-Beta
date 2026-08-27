"""Speaking two languages: every path that talks to the model has to know about it.

There are four places a prompt gets assembled and they do not share an assembly step, so
adding the language block to one of them is not adding it to the system. That is exactly
what happened: it went into `_system_prompt`, and a `講中文` in a server channel came back
as *"I don't know what you're saying dude, can you speak English"* — the right persona,
the wrong language, because the public path builds its own prompt.

These tests enumerate the paths rather than checking one, which is the only shape that
would have caught it.

The second half is about the components rather than the prompt, and they fail in three
different ways: a wrong STT language is silent and looks like a broken microphone, a
wrong voice is obvious immediately, and a wrong *phonemiser* is the worst of the three —
it produces fluent, confident, wrong words in the right voice.
"""

from __future__ import annotations

import pytest

from aria.chunker import is_cjk
from aria.config import (
    BILINGUAL,
    BILINGUAL_RULES,
    CJK_TTS_LANGS,
    LANGUAGE_NOW,
    LANGUAGES,
    TRADITIONAL_CHINESE,
    ZH_ONLY_RULES,
    Config,
    apply_language,
)
from aria.loop import VoiceLoop
from aria.memory import Memory
from aria.persona_store import PersonaStore

ME = 4242


def loop_for(tmp_path, code: str):
    cfg = Config()
    cfg.memory.path = tmp_path / "m.json"
    cfg.memory.persona_path = tmp_path / "persona.txt"
    cfg.discord.owner_id = ME
    apply_language(cfg, code)

    ln = VoiceLoop.__new__(VoiceLoop)
    ln.cfg = cfg
    ln._watching = False
    ln.server = None
    ln._history = []
    ln.memory = Memory(cfg.memory.path)
    ln.persona = PersonaStore(cfg.memory.persona_path, cfg.persona)
    return ln, cfg


# --- the components that must agree ------------------------------------------
def test_chinese_points_everything_at_chinese(tmp_path):
    _, cfg = loop_for(tmp_path, "zh")
    assert cfg.stt.language == "zh", "Whisper told it is English hears nonsense"
    assert cfg.tts.lang == "zh", "espeak's Mandarin is not what these voices know"
    assert cfg.tts.voice.startswith("z"), "an English voice reading Chinese"


def test_english_is_unchanged(tmp_path):
    _, cfg = loop_for(tmp_path, "en")
    assert (cfg.stt.language, cfg.tts.lang) == ("en", "en-us")
    assert cfg.tts.voice == "af_bella"


def test_auto_lets_whisper_decide(tmp_path):
    # The one setting a bilingual speaker cannot get right in advance. Measured 5/5 on
    # short one-clause turns in both languages, the shortest at 0.78 confidence.
    _, cfg = loop_for(tmp_path, "auto")
    assert cfg.stt.language is None


def test_every_language_can_reach_the_other_script(tmp_path):
    # Not an edge case in any of them: an English reply names a 中文 file and a Chinese
    # reply names `docker`. A missing alt is not an accent — af_bella reads Han
    # characters as nothing at all.
    for code in LANGUAGES:
        _, cfg = loop_for(tmp_path, code)
        assert cfg.tts.alt_voice and cfg.tts.alt_lang, f"{code} is stuck in one script"
        primary_is_cjk = cfg.tts.lang in CJK_TTS_LANGS
        alt_is_cjk = cfg.tts.alt_lang in CJK_TTS_LANGS
        assert primary_is_cjk != alt_is_cjk, f"{code}: both voices serve one script"


def test_the_chinese_phonemiser_is_not_espeak():
    # `zh` here is ours, not espeak's — espeak refuses it outright ('zh' and 'zh-cn' are
    # both "not supported by the espeak backend"), and its `cmn`, which it does accept,
    # is the wrong alphabet for voices trained on misaki. Measured intelligibility of
    # her own speech read back by Whisper: espeak 0.46, misaki 0.73, and the espeak
    # failures are different *words* rather than a worse accent.
    assert LANGUAGES["zh"].tts_lang == "zh"
    assert LANGUAGES["zh"].tts_lang in CJK_TTS_LANGS


def test_chinese_transcripts_are_converted_to_traditional(tmp_path):
    # Whisper answers in Simplified for Mandarin whatever you ask it, so without this
    # his own words arrive on screen and in her history in the script she is told never
    # to use — and a model copies the conversation it is shown far more readily than it
    # follows a rule above it.
    for code in ("zh", "auto"):
        _, cfg = loop_for(tmp_path, code)
        assert cfg.stt.traditional, f"{code} would show him Simplified"


# --- every prompt path -------------------------------------------------------
PATHS = ["voice", "discord_dm", "public_guest", "public_owner"]


def prompt_for(ln, path: str) -> str:
    return {
        "voice": lambda: ln._system_prompt(),
        "discord_dm": lambda: ln._system_prompt(discord=True),
        "public_guest": lambda: ln._public_system_prompt(owner=False),
        "public_owner": lambda: ln._public_system_prompt(owner=True),
    }[path]()


@pytest.mark.parametrize("path", PATHS)
def test_every_prompt_path_carries_chinese(tmp_path, path):
    ln, _ = loop_for(tmp_path, "zh")
    assert ZH_ONLY_RULES in prompt_for(ln, path), f"{path} would answer in English"


@pytest.mark.parametrize("path", PATHS)
def test_every_prompt_path_carries_the_bilingual_rule(tmp_path, path):
    ln, _ = loop_for(tmp_path, "auto")
    assert BILINGUAL_RULES in prompt_for(ln, path), f"{path} would ignore his language"


@pytest.mark.parametrize("code,samples", [("zh", TRADITIONAL_CHINESE), ("auto", BILINGUAL)])
def test_a_stranger_gets_the_language_rule_without_his_register(tmp_path, code, samples):
    """The rule reaches a guest; the sample lines do not.

    Those samples are how she talks to the person she belongs to, 老兄 and 兄弟 included
    — which are 'dude' and 'mate', and `PUBLIC_PERSONA` bans both by name. Sending the
    whole block to a stranger handed them the closeness the guest persona exists to
    refuse, and it did it in the one script the pet-name test was not reading.
    """
    ln, _ = loop_for(tmp_path, code)
    guest = prompt_for(ln, "public_guest")
    assert samples not in guest, "a stranger is being shown his register"
    # The guest persona names `dude` and `mate` once, to forbid them. Scrubbed rather
    # than excluded, so a second mention anywhere else still fails — same trick the
    # pet-name test in test_discord.py uses, and lowercased for the same reason.
    rest = guest.lower().replace("not dude, not mate", "")
    for pet in ("老兄", "兄弟", "dude", "mate"):
        assert pet not in rest, pet
    # He still gets his own voice in his own server.
    assert samples in prompt_for(ln, "public_owner")


# --- which language this turn is in ------------------------------------------
# The rule and the samples describe her across a conversation. None of it knows what was
# just said, and that is what made her answer in Chinese whatever she was asked in: the
# bilingual block is mostly Chinese, and a 9B copies samples far more readily than it
# follows the rule above them. Whisper had already decided, per turn, and the answer
# went no further than a line printed to the console.


def test_the_prompt_names_the_language_that_was_just_spoken(tmp_path):
    ln, _ = loop_for(tmp_path, "auto")
    for code in ("en", "zh"):
        ln._spoken_language = code
        assert LANGUAGE_NOW[code] in ln._system_prompt(), code


def test_the_language_of_the_turn_is_the_last_word(tmp_path):
    # It only outranks a page of sample lines from the end. Anything appended after it
    # — a Discord style block, the no-question reminder — takes the position back.
    ln, _ = loop_for(tmp_path, "auto")
    ln._spoken_language = "en"
    for discord in (False, True):
        assert ln._system_prompt(discord=discord).endswith(LANGUAGE_NOW["en"])


def test_a_pinned_run_says_nothing_per_turn(tmp_path):
    # Nothing to decide: `en` and `zh` pin every component, and repeating the same
    # sentence every turn would only cost tokens and break the cached prefix.
    for code in ("en", "zh"):
        ln, _ = loop_for(tmp_path, code)
        ln._spoken_language = "en"
        assert ln._language_now() == ""


def test_nothing_is_claimed_before_anything_is_said(tmp_path):
    # First turn of a bilingual run, and no evidence either way. Guessing here would be
    # guessing at the thing this whole block exists to stop guessing at.
    ln, _ = loop_for(tmp_path, "auto")
    assert ln._language_now() == ""


def test_a_typed_turn_is_judged_by_its_script(tmp_path):
    # Discord messages arrive as characters — Whisper never saw them. They must not
    # inherit the last thing *spoken*, because moving from the desk to a phone is
    # exactly when the two disagree.
    ln, _ = loop_for(tmp_path, "auto")
    ln._spoken_language = "zh"
    ln._history = [{"role": "user", "content": "what did you do today"}]
    assert LANGUAGE_NOW["en"] in ln._system_prompt(discord=True)
    assert LANGUAGE_NOW["zh"] not in ln._system_prompt(discord=True)


def test_a_guest_is_answered_in_the_language_they_typed(tmp_path):
    ln, _ = loop_for(tmp_path, "auto")
    ln._history = [{"role": "user", "content": "你今天在做什麼"}]
    assert LANGUAGE_NOW["zh"] in ln._public_system_prompt(owner=False)


def test_the_screen_reader_gets_it_too(tmp_path):
    # The vision backends build their own prompt from VISION_PROMPT and cannot reach the
    # loop, so they need their own copy or a Chinese screen question is answered in
    # English.
    for code, block in (("zh", TRADITIONAL_CHINESE), ("auto", BILINGUAL)):
        _, cfg = loop_for(tmp_path, code)
        assert block in cfg.vision.language_prompt


def test_no_language_block_leaks_into_an_english_run(tmp_path):
    ln, cfg = loop_for(tmp_path, "en")
    for prompt in (ln._system_prompt(), ln._public_system_prompt(owner=True),
                   cfg.vision.language_prompt):
        assert TRADITIONAL_CHINESE not in prompt
        assert BILINGUAL not in prompt


def test_the_two_chinese_blocks_are_never_both_present(tmp_path):
    """They carry opposite rules, and a prompt holding both holds a contradiction.

    `zh` says *always Chinese, even when he asks in English*. `auto` says *whichever he
    just used*. Concatenating them — the obvious way to add bilingual support on top of
    what was there — produces a model that picks one at random per turn.
    """
    for code in LANGUAGES:
        ln, _ = loop_for(tmp_path, code)
        for path in PATHS:
            prompt = prompt_for(ln, path)
            assert not (TRADITIONAL_CHINESE in prompt and BILINGUAL in prompt), (
                f"{code}/{path} tells her two different things"
            )


# --- the blocks themselves ---------------------------------------------------
@pytest.mark.parametrize("block", [TRADITIONAL_CHINESE, BILINGUAL])
def test_the_instruction_is_written_in_chinese(block):
    # An instruction to use language X lands far better in language X. In English it
    # reads as a fact about her, and comes back as an English sentence agreeing to
    # speak Chinese.
    han = sum("一" <= c <= "鿿" for c in block)
    assert han > 80, f"only {han} Han characters — is this written in English?"


@pytest.mark.parametrize("block", [TRADITIONAL_CHINESE, BILINGUAL])
def test_it_carries_chinese_sample_lines(block):
    # Samples beat rules for this model class, and without Chinese ones she inherits the
    # register of the English examples and sounds translated.
    assert "「" in block


@pytest.mark.parametrize("block", [TRADITIONAL_CHINESE, BILINGUAL])
def test_it_names_simplified_as_the_thing_to_avoid(block):
    assert "簡體" in block


def test_the_bilingual_block_does_not_pin_her_to_chinese():
    # The line it inherits from the block above is the one that has to go: "他用英文問
    # 你，你還是用繁體中文回答" is the exact opposite of following him.
    assert "他用英文問你，你還是用繁體中文回答" in TRADITIONAL_CHINESE
    assert "他用英文問你，你還是用繁體中文回答" not in BILINGUAL


# --- picking a voice per chunk -----------------------------------------------
class FakeTTS:
    """`KokoroTTS._for` without loading 325 MB of ONNX to ask it a question."""

    def __init__(self, cfg):
        self._cfg = cfg

    _for = None  # bound below


def voice_for(cfg, text: str) -> tuple[str, str]:
    from aria.tts.kokoro_backend import KokoroTTS

    return KokoroTTS._for(FakeTTS(cfg), text)


@pytest.mark.parametrize("code", ["en", "zh", "auto"])
@pytest.mark.parametrize(
    "text, cjk",
    [
        ("Hey, are you there?", False),
        ("這麼早就起來了，怎麼了老兄。", True),
        ("早安。", True),
        ("我在用 Rust 寫這個。", True),      # mostly Chinese with a package name in it
    ],
)
def test_the_voice_follows_the_script_of_the_chunk(tmp_path, code, text, cjk):
    """Whatever she was started in, a chunk is said by the voice that can say it.

    Reusing the chunker's own script test on purpose: the thing that decided where to
    cut this chunk should be the thing that decides how to pronounce it, or a reply gets
    cut as Chinese and spoken as English.
    """
    _, cfg = loop_for(tmp_path, code)
    voice, phonemiser = voice_for(cfg.tts, text)
    assert is_cjk(text) == cjk, "the chunker and this test disagree about the script"
    assert (phonemiser in CJK_TTS_LANGS) == cjk, f"{code}: {text!r} phonemised as {phonemiser}"
    assert voice.startswith("z") == cjk, f"{code}: {text!r} said by {voice}"


# --- phonemising a mixed chunk -----------------------------------------------
class FakeEspeak:
    """Stands in for Kokoro's tokeniser, so the split is visible in the output."""

    def phonemize(self, text: str, lang: str) -> str:
        return f"<{lang}:{text.strip()}>"


def phonemes(text: str) -> str:
    from aria.tts.chinese import ChinesePhonemiser

    p = ChinesePhonemiser(FakeEspeak())
    return p(text)


def test_latin_runs_go_to_espeak_not_to_misaki():
    """The failure this prevents is silent and specific.

    misaki passes English through as literal Latin text, which then reaches Kokoro's
    tokeniser *as if it were IPA* — so "Rust" is pronounced as whatever /R/, /u/, /s/,
    /t/ happen to mean, inside an otherwise fluent Chinese sentence. The persona
    explicitly tells her to leave package names and commands in English, so this is the
    common case rather than a corner of one.
    """
    out = phonemes("我在用 Rust 寫這個。")
    assert "<en-us:Rust>" in out
    assert "Rust" not in out.replace("<en-us:Rust>", ""), "the raw word also leaked through"


def test_the_chinese_around_it_still_gets_tone_marks():
    out = phonemes("我在用 Rust 寫這個。")
    assert any(mark in out for mark in "↓↘↗→"), "no tones — this went through espeak"


def test_pure_chinese_never_touches_espeak():
    assert "<" not in phonemes("這麼早就起來了，怎麼了老兄。")


def test_a_bare_number_stays_with_misaki():
    # misaki says it in Chinese; espeak would say it in English in the middle of a
    # Chinese sentence.
    assert "<" not in phonemes("我有三十五個。")


def test_misaki_failing_costs_tones_not_the_turn():
    # A phonemiser that raises would take out the whole reply. Losing the tone marks on
    # one sentence is much better than losing the sentence.
    from aria.tts.chinese import ChinesePhonemiser

    def explode(_text):
        raise RuntimeError("jieba fell over")

    p = ChinesePhonemiser(FakeEspeak())
    p.load()
    p._g2p = explode
    assert p("早安。") == "<cmn:早安。>", "a broken phonemiser should degrade, not raise"


# --- the panel's one flat list of 54 voices -----------------------------------
class FakeServer:
    """Enough of `OverlayServer` for `_on_command` to finish.

    Every branch of that method ends by pushing a full settings snapshot rather than
    trusting the panel to predict the result, so the whole snapshot has to assemble for
    a test about one field.
    """

    capabilities: dict = {"emotions": []}

    def __init__(self):
        self.sent = []

    def send(self, **kw):
        self.sent.append(kw)

    def broadcast(self, **kw):
        self.sent.append(kw)


class FakeVoices:
    def voices(self):
        return ["af_bella", "af_nicole", "zf_xiaoxiao", "zm_yunxi"]


def pick_voice(tmp_path, code: str, value: str):
    ln, cfg = loop_for(tmp_path, code)
    ln.tts = FakeVoices()
    ln.server = FakeServer()
    ln.listener = None
    ln.wake = None
    ln.discord = None
    ln._on_command({"name": "set_voice", "value": value})
    return cfg.tts.voice, cfg.tts.alt_voice


def test_picking_a_mandarin_voice_changes_her_mandarin_voice(tmp_path):
    """The panel shows one flat list, and she has two voices.

    Assigning the pick to `voice` unconditionally would point both slots at a Mandarin
    voice, which does not give her a Mandarin accent in English — it leaves Han text
    with nowhere to go, because the alt slot is what Chinese chunks are routed to.
    """
    latin, cjk = pick_voice(tmp_path, "auto", "zm_yunxi")
    assert (latin, cjk) == ("af_bella", "zm_yunxi")


def test_picking_an_english_voice_changes_her_english_voice(tmp_path):
    latin, cjk = pick_voice(tmp_path, "auto", "af_nicole")
    assert (latin, cjk) == ("af_nicole", "zf_xiaoxiao")


def test_it_works_from_the_chinese_side_too(tmp_path):
    # Started in `zh`, the primary slot is the Mandarin one and the alt is English.
    latin_slot, alt = pick_voice(tmp_path, "zh", "af_nicole")
    assert (latin_slot, alt) == ("zf_xiaoxiao", "af_nicole")


# --- detection is allowed two answers, not ninety-nine ------------------------
class FakeInfo:
    def __init__(self, language, ranked):
        self.language = language
        self.all_language_probs = ranked


class FakeSegment:
    no_speech_prob = 0.0

    def __init__(self, text):
        self.text = text


class FakeWhisper:
    """Answers with a language she does not speak first, then with what it is told."""

    def __init__(self, first, ranked, by_language):
        self.first, self.ranked, self.by_language = first, ranked, by_language
        self.asked: list = []

    def transcribe(self, audio, language=None, **kw):
        self.asked.append(language)
        if language is None:
            return [FakeSegment(self.by_language[self.first])], FakeInfo(
                self.first, self.ranked
            )
        return [FakeSegment(self.by_language[language])], FakeInfo(language, [])


def stt_with(model, **over):
    from aria.config import Config, apply_language
    from aria.stt.whisper import WhisperSTT

    cfg = Config()
    apply_language(cfg, "auto")
    for k, v in over.items():
        setattr(cfg.stt, k, v)
    stt = WhisperSTT.__new__(WhisperSTT)
    stt._cfg = cfg.stt
    stt._model = model
    stt._to_traditional = None
    return stt


def test_a_language_she_does_not_speak_is_redone():
    """Seen live, twice in five turns: Whisper answered Korean and she replied — in
    character, fluently — that she does not speak Korean and could he try Chinese.

    She has two languages and Whisper chooses from ninety-nine, so a third one is a
    misdetection by construction rather than a fact to act on.
    """
    model = FakeWhisper(
        first="ko",
        ranked=[("ko", 0.41), ("zh", 0.33), ("en", 0.11)],
        by_language={"ko": "우유", "zh": "有魚", "en": "oo yoo"},
    )
    heard = stt_with(model).transcribe(None)
    assert model.asked == [None, "zh"], "it did not fall back to her best language"
    assert (heard.language, heard.text) == ("zh", "有魚")


def test_it_falls_back_in_order_when_neither_ranks():
    model = FakeWhisper(
        first="ko", ranked=[("ko", 0.9)], by_language={"ko": "우유", "en": "oo yoo"}
    )
    assert stt_with(model).transcribe(None).language == "en"


def test_a_language_she_does_speak_is_left_alone():
    # The cost is one extra pass, so it must land only on the turns that were already
    # going to be wrong.
    model = FakeWhisper(first="zh", ranked=[("zh", 0.99)], by_language={"zh": "早安"})
    assert stt_with(model).transcribe(None).text == "早安"
    assert model.asked == [None], "a correct detection paid for a second pass"


def test_a_pinned_run_never_second_guesses():
    # `allowed` is empty for en and zh, because there is nothing to choose between.
    model = FakeWhisper(first="ko", ranked=[], by_language={"ko": "우유", "zh": "早安"})
    stt_with(model, language="zh", allowed=()).transcribe(None)
    assert model.asked == ["zh"], "a pinned run has nothing to choose between"
