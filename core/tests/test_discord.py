"""Who she answers on Discord, and what shape the answer comes out in.

These are the two ways this feature fails quietly. A gate that is too generous turns a
private assistant into one that anybody sharing a server can talk to — and can point at
a desktop. A gate that is too tight produces a bot that sits there online and says
nothing, which is indistinguishable from a broken one.

No token, no network, no discord.py objects: the whole gate is plain values by design,
which is what makes it testable at all.
"""

from __future__ import annotations

import pytest

from aria.chunker import clean_for_speech
from aria.config import (
    DISCORD_MOOD_EMOJI,
    DISCORD_STYLE,
    OWNER_IN_PUBLIC,
    PERSONA,
    PUBLIC_PERSONA,
    Config,
    DiscordConfig,
    _bool_env,
    _int_env,
)
from aria.discord_bot import Incoming, apply_mood_emoji, should_reply, split_message
from aria.emotion import extract
from aria.loop import VoiceLoop
from aria.memory import Memory

ME, STRANGER, HER = 111, 222, 999
CHANNEL, LISTENED = 555, 777


def ask(cfg: DiscordConfig, **over) -> tuple[bool, str]:
    base = dict(
        author_id=ME,
        is_self=False,
        is_bot=False,
        is_dm=True,
        channel_id=CHANNEL,
        mentioned=False,
        replied_to_her=False,
        has_text=True,
    )
    return should_reply(cfg, **{**base, **over})


# Every field these tests care about is passed explicitly, `public` included. The
# dataclass defaults read the environment, and `config._load_env` has already pulled in
# the real `core/.env` by the time pytest imports anything — so a bare `DiscordConfig()`
# here is whatever this machine happens to be configured as. Switching public mode on for
# real broke two tests that way, which is the suite reporting the developer's settings
# rather than the code.
@pytest.fixture
def owned() -> DiscordConfig:
    return DiscordConfig(token="x", owner_id=ME, channels=(LISTENED,), public=False)


@pytest.fixture
def open_to_all() -> DiscordConfig:
    return DiscordConfig(token="x", owner_id=0, public=False)


# --- who she answers ---------------------------------------------------------
def test_answers_the_owner_in_a_dm(owned):
    ok, why = ask(owned)
    assert ok and why == "dm"


def test_ignores_everyone_else_even_in_a_dm(owned):
    # The one that matters: anyone who shares a server with the bot can open a DM to
    # it, so "she only talks to me" is not something DMs give you for free.
    ok, why = ask(owned, author_id=STRANGER)
    assert not ok and "owner" in why


def test_never_answers_herself(owned):
    assert not ask(owned, is_self=True)[0], "a bot replying to itself is a loop"


def test_never_answers_another_bot(owned):
    assert not ask(owned, is_bot=True)[0]


def test_ignores_a_message_with_no_text(owned):
    # An image with no caption. There is nothing to answer and an empty prompt makes a
    # 9B invent a question to answer instead.
    assert not ask(owned, has_text=False)[0]


def test_stays_quiet_in_a_server_unless_spoken_to(owned):
    ok, why = ask(owned, is_dm=False)
    assert not ok and "not addressed" in why


@pytest.mark.parametrize("how", ["mentioned", "replied_to_her"])
def test_answers_when_addressed_in_a_server(owned, how):
    assert ask(owned, is_dm=False, **{how: True})[0]


def test_answers_everything_in_a_listening_channel(owned):
    assert ask(owned, is_dm=False, channel_id=LISTENED)[0]


def test_a_listening_channel_still_does_not_override_the_owner(owned):
    assert not ask(owned, is_dm=False, channel_id=LISTENED, author_id=STRANGER)[0]


def test_without_an_owner_she_answers_whoever_reaches_her(open_to_all):
    # Deliberate, and the reason it is safe is elsewhere: no owner means no screen
    # capture, which is the only thing here that leaks anything.
    assert ask(open_to_all, author_id=STRANGER)[0]


def test_every_refusal_says_why():
    # The reason strings are the entire diagnostic surface for "why is the bot
    # ignoring me". A blank one is a silent bot with no explanation.
    cfg = DiscordConfig(token="x", owner_id=ME)
    for over in ({"is_self": True}, {"is_bot": True}, {"has_text": False},
                 {"author_id": STRANGER}, {"is_dm": False}):
        ok, why = ask(cfg, **over)
        assert not ok and len(why) > 8, over


# --- opening her up to a server ----------------------------------------------
def test_public_lets_a_stranger_talk_in_a_server(owned):
    owned.public = True
    assert ask(owned, author_id=STRANGER, is_dm=False, mentioned=True)[0]


def test_public_does_not_open_dms(owned):
    # Anyone sharing a server with the bot can open a DM to it. A public DM would be a
    # private channel with a stranger in it, which is the opposite of opening up a
    # *server*, so `public` must not reach this door at all.
    owned.public = True
    ok, why = ask(owned, author_id=STRANGER, is_dm=True)
    assert not ok and "DMs are his alone" in why


def test_public_still_respects_where_she_may_speak(owned):
    # Public means "strangers may talk to her", not "she answers everything". A server
    # bot that replies to every message is one people mute.
    owned.public = True
    assert not ask(owned, author_id=STRANGER, is_dm=False)[0]


@pytest.mark.parametrize("raw,expected", [
    ("", False), ("0", False), ("false", False), ("no", False), ("off", False),
    ("1", True), ("true", True), ("YES", True), ("on", True),
])
def test_opening_her_up_takes_an_explicit_yes(monkeypatch, raw, expected):
    # Asserting `DiscordConfig().public is False` would have been the obvious test and
    # it tests this machine's .env, not the code. The contract worth pinning is that
    # anything short of a deliberate yes leaves her his alone.
    monkeypatch.setenv("ARIA_DISCORD_PUBLIC", raw)
    assert _bool_env("ARIA_DISCORD_PUBLIC") is expected


@pytest.mark.parametrize("raw,expected", [
    ("1234567890123456789", 1234567890123456789),
    ("  1234567890123456789  ", 1234567890123456789),
    ("<#1234567890123456789>", 1234567890123456789),   # pasted as a channel mention
    ("", 0),
    ("not-an-id", 0),
])
def test_an_id_survives_being_pasted_by_hand(monkeypatch, raw, expected):
    monkeypatch.setenv("ARIA_TEST_ID", raw)
    assert _int_env("ARIA_TEST_ID") == expected


def test_two_ids_run_together_are_refused_rather_than_guessed_at(monkeypatch, capsys):
    # This shipped, from a real .env: stripping every non-digit and gluing the rest
    # together turned two ids into a well-formed 38-digit number, which then failed as
    # "I can't find that voice channel" - Discord blamed for a typo in a file.
    #
    # Refused, not split. Snowflakes have no checksum and no fixed length, so there is
    # no honest way to find the join; a confidently wrong id is worse than none.
    monkeypatch.setenv("ARIA_TEST_ID", "12345678901234567899876543210987654321")
    assert _int_env("ARIA_TEST_ID") == 0
    assert "not a usable id" in capsys.readouterr().err


def test_two_separated_ids_take_the_first_and_say_so(monkeypatch, capsys):
    monkeypatch.setenv("ARIA_TEST_ID", "1234567890123456789, 9876543210987654321")
    assert _int_env("ARIA_TEST_ID") == 1234567890123456789
    assert "more than one id" in capsys.readouterr().err


def test_dms_can_be_switched_off(owned):
    owned.dms = False
    assert not ask(owned)[0]


def test_mentions_can_be_switched_off(owned):
    owned.mentions = False
    assert not ask(owned, is_dm=False, mentioned=True)[0]


# --- what comes out ----------------------------------------------------------
def test_a_normal_reply_is_left_alone():
    assert split_message("hey. yeah I saw that one.", 1900) == ["hey. yeah I saw that one."]


def test_nothing_is_sent_for_nothing():
    assert split_message("   ", 1900) == []


def test_a_long_reply_fits_discords_limit():
    text = " ".join(["word"] * 2000)
    parts = split_message(text, 100)
    assert parts and all(len(p) <= 100 for p in parts)


def test_it_breaks_where_a_reader_would():
    text = "First paragraph here.\n\nSecond paragraph, quite a bit longer than the first."
    parts = split_message(text, 40)
    assert parts[0] == "First paragraph here."


def test_it_never_splits_a_word_in_half():
    parts = split_message("supercalifragilistic " * 20, 60)
    rejoined = " ".join(parts).split()
    assert set(rejoined) == {"supercalifragilistic"}


def test_a_split_code_block_is_closed_and_reopened():
    # Otherwise the first half renders as an unterminated fence that swallows the rest
    # of the message, and the second half shows its own ``` as literal text.
    code = "\n".join(f"line_{i} = {i}" for i in range(40))
    parts = split_message(f"here:\n```python\n{code}\n```", 200)
    assert len(parts) > 1
    assert all(p.count("```") % 2 == 0 for p in parts), "every part must balance"
    assert parts[0].endswith("```") and parts[1].startswith("```")


# --- which rules she writes under --------------------------------------------
def _prompt_loop(tmp_path) -> VoiceLoop:
    # No setup(): that loads Whisper. Only prompt assembly is under test here.
    ln = VoiceLoop.__new__(VoiceLoop)
    ln.cfg = Config()
    ln._watching = False
    ln.server = None
    ln._history = []
    ln.memory = Memory(tmp_path / "memory.json")
    return ln


def test_a_typed_turn_gets_the_texting_rules(tmp_path):
    prompt = _prompt_loop(tmp_path)._system_prompt(discord=True)
    assert DISCORD_STYLE in prompt


def test_a_spoken_turn_does_not(tmp_path):
    # The read-aloud rules and the texting rules contradict each other on purpose.
    # Leaking the texting ones into a spoken turn puts markdown into the synthesiser.
    assert DISCORD_STYLE not in _prompt_loop(tmp_path)._system_prompt()


def test_the_texting_rules_sit_at_the_very_end(tmp_path):
    # Ollama caches the longest unchanged prefix. He moves between the desk and his
    # phone mid-conversation, so a block that changes with the medium has to be below
    # everything that does not — or every switch re-prefills the whole persona.
    prompt = _prompt_loop(tmp_path)._system_prompt(discord=True)
    tail = prompt[prompt.index(DISCORD_STYLE):]
    assert tail.startswith(DISCORD_STYLE), "nothing stable may sit below the style block"


def test_the_mood_vocabulary_is_offered_in_full(tmp_path):
    # Restated here rather than inherited from the character's emotion block: whether
    # her Live2D model has a `flirty` pose has nothing to do with whether 😏 is the
    # right emoji, and headless there is no character and no block at all.
    prompt = _prompt_loop(tmp_path)._system_prompt(discord=True)
    for mood in DISCORD_MOOD_EMOJI:
        assert f"[{mood}]" in prompt, mood


def test_a_spoken_turn_is_never_offered_the_mood_map(tmp_path):
    # The persona already forbids emoji out loud; what must not leak is the instruction
    # that a marker *becomes* one, which would put 😊 into the synthesiser.
    prompt = _prompt_loop(tmp_path)._system_prompt()
    assert "becomes an emoji" not in prompt
    assert all(glyph not in prompt for glyph in DISCORD_MOOD_EMOJI.values() if glyph)


# --- what a stranger can and cannot reach ------------------------------------
def _loop_with_secrets(tmp_path) -> VoiceLoop:
    ln = _prompt_loop(tmp_path)
    ln.memory.add("his sister is called Mei", source="you")
    ln.memory.add("has a demo on Friday")
    ln._history = [
        {"role": "user", "content": "I'm quitting my job on Monday, don't tell anyone"},
        {"role": "assistant", "content": "Okay. That's yours to tell."},
    ]
    return ln


@pytest.mark.parametrize("owner", [False, True])
def test_the_public_prompt_carries_none_of_his_life(tmp_path, owner):
    # The one that matters, and it has to hold for *him* as well. He gets his own Aria in
    # a shared channel so she doesn't read as a stranger — but the room still decides
    # what she can reach, and his notes are not in it either way. Otherwise "be yourself
    # with him in public" quietly becomes "read his private notes out in public".
    prompt = _loop_with_secrets(tmp_path)._public_system_prompt(owner=owner)
    for secret in ("Mei", "demo", "Friday", "quitting", "Monday"):
        assert secret not in prompt, f"{secret} leaked (owner={owner})"


def test_he_gets_his_own_aria_in_a_shared_channel(tmp_path):
    # Being handed the guest persona in his own server made her read as "just the model
    # itself", which is exactly what that persona was written to be — and it bought no
    # safety, because the persona contains nothing about him.
    prompt = _loop_with_secrets(tmp_path)._public_system_prompt(owner=True)
    assert PERSONA in prompt
    assert PUBLIC_PERSONA not in prompt
    assert OWNER_IN_PUBLIC in prompt


def test_he_is_told_the_room_is_shared_and_her_notes_are_not_here(tmp_path):
    # The two things she cannot work out from the prompt she is handed. Without the
    # second she agrees to remember something and writes nothing, which is the exact
    # failure the memory design exists to prevent, aimed at him this time.
    prompt = _loop_with_secrets(tmp_path)._public_system_prompt(owner=True)
    assert "Other people can read" in prompt
    assert "none of your notes here" in prompt


def test_a_stranger_still_gets_the_guest_persona(tmp_path):
    prompt = _loop_with_secrets(tmp_path)._public_system_prompt(owner=False)
    assert PUBLIC_PERSONA in prompt
    assert PERSONA not in prompt
    # No pet name reaches a stranger. `dude`/`mate` are the ones she uses with the
    # person she belongs to, and a guest hearing them would be claiming a closeness she
    # does not have. The guest persona names them once, to forbid them — that line is
    # scrubbed before the check rather than excluded from it, so a *second* mention
    # anywhere else still fails.
    named_to_forbid = "not dude, not mate"
    assert named_to_forbid in prompt.lower(), "the guest rule stopped naming what it bans"
    rest = prompt.lower().replace(named_to_forbid, "")
    for term in ("bro,", "dude", "mate", "babe", "girlfriend"):
        assert term not in rest, term


def test_his_own_prompt_still_has_everything(tmp_path):
    # The guard rails must not have cost him the thing they protect.
    prompt = _loop_with_secrets(tmp_path)._system_prompt(discord=True)
    assert "Mei" in prompt and "demo" in prompt


def test_a_server_channel_is_public_even_for_him(tmp_path):
    # The channel decides, not the person. He gets his own Aria in a DM; @mentioning
    # her in a room full of friends does not put his notes in her context.
    ln = _loop_with_secrets(tmp_path)
    ln.cfg.discord.owner_id = ME
    him_in_public = Incoming(text="hey", author_id=ME, author_name="him",
                             channel_id=CHANNEL, is_dm=False)
    him_in_dm = Incoming(text="hey", author_id=ME, author_name="him",
                         channel_id=CHANNEL, is_dm=True)
    assert not ln._is_private(him_in_public), "a public room is public, whoever is in it"
    assert ln._is_private(him_in_dm)


def test_nobody_is_him_when_no_owner_is_configured(tmp_path):
    ln = _loop_with_secrets(tmp_path)
    ln.cfg.discord.owner_id = 0
    msg = Incoming(text="hey", author_id=ME, author_name="him",
                   channel_id=CHANNEL, is_dm=True)
    assert not ln._is_private(msg), "unprovable identity must not unlock his conversation"


# --- mood, as an emoji -------------------------------------------------------
def test_the_marker_becomes_an_emoji_at_the_end_of_its_sentence():
    out = apply_mood_emoji("[happy] That actually worked.", DISCORD_MOOD_EMOJI)
    assert out == "[happy] That actually worked. 😊"


def test_it_lands_on_the_sentence_the_marker_opened_not_the_first():
    # The marker's position is the whole reason this runs before extraction. Appending
    # to the message instead would put the feeling on the wrong sentence.
    out = apply_mood_emoji(
        "Okay. So that broke. [worried] Are you alright?", DISCORD_MOOD_EMOJI
    )
    assert out.endswith("Are you alright? 😟")
    assert "So that broke. 😟" not in out


def test_the_emoji_goes_after_the_full_stop_not_before():
    out = apply_mood_emoji("[shy] Don't look at me.", DISCORD_MOOD_EMOJI)
    assert out.endswith(". 😳"), out


def test_a_sentence_with_no_terminator_still_gets_one():
    assert apply_mood_emoji("[happy] missed you", DISCORD_MOOD_EMOJI).endswith("missed you 😊")


def test_only_one_emoji_per_message():
    # She marks a mood per sentence for her face, where a change every few seconds looks
    # alive. The same rate in text reads as a keyboard with a sticky key.
    out = apply_mood_emoji("[happy] One. [sad] Two. [angry] Three.", DISCORD_MOOD_EMOJI)
    assert sum(out.count(e) for e in DISCORD_MOOD_EMOJI.values() if e) == 1


def test_neutral_asks_for_nothing_and_means_it():
    assert apply_mood_emoji("[neutral] It's Paris.", DISCORD_MOOD_EMOJI) == "[neutral] It's Paris."


def test_neutral_does_not_block_a_real_mood_later():
    out = apply_mood_emoji("[neutral] Sure. [happy] Nice one.", DISCORD_MOOD_EMOJI)
    assert out.endswith("Nice one. 😊")


def test_an_emoji_she_typed_herself_wins():
    # Observed: "Ugh, not the work again. 😩" — weary, which no marker covers. She is
    # told not to and mostly doesn't, but overruling her is worse than allowing it, and
    # stacking a second emoji next to it is worse than either.
    raw = "[sad] Ugh, not the work again. 😩"
    assert apply_mood_emoji(raw, DISCORD_MOOD_EMOJI) == raw


def test_an_unmapped_mood_is_skipped_not_guessed_at():
    # A name the map has never heard of. `smug` used to sit here and is mapped now, which
    # is the sort of thing that turns a real check into a tautology.
    assert apply_mood_emoji("[wistful] Told you.", DISCORD_MOOD_EMOJI) == "[wistful] Told you."


def test_text_with_no_markers_is_untouched():
    assert apply_mood_emoji("It's Paris.", DISCORD_MOOD_EMOJI) == "It's Paris."


def test_the_marker_still_comes_out_afterwards():
    # The two passes have to compose: emoji placed while the marker is visible, marker
    # removed after. A leftover [happy] posted to Discord is the whole failure.
    out = apply_mood_emoji("[happy] That worked.", DISCORD_MOOD_EMOJI)
    said, emotions = extract(out, list(DISCORD_MOOD_EMOJI))
    assert said == "That worked. 😊"
    assert emotions == ["happy"]


def test_every_emoji_would_be_stripped_before_speech():
    # `--speak-discord` reads a posted reply out loud, and it is handed the same text.
    # An emoji Kokoro can see is an emoji Kokoro pronounces.
    for glyph in DISCORD_MOOD_EMOJI.values():
        assert clean_for_speech(f"That worked. {glyph}") == "That worked.", glyph
