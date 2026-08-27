"""Aria on Discord — the same person, reachable from a phone.

Three things shape this file.

**She is one Aria, not two.** Discord is a second way in to the loop that already
exists, not a second assistant. Same memory file, same conversation history, same
notes: say something out loud at the desk and ask about it from your phone an hour
later, and she knows. The only thing that changes is the shape of a reply, because a
sentence written for a speech synthesiser makes a strange text message — see
`DISCORD_STYLE` in config.

**Nothing from discord.py crosses into the loop.** The bot hands over an `Incoming`
of plain values and a `Channel` to answer on, and that is the entire surface. So the
turn logic can be driven by a test with no gateway, no token and no network, which is
the only way any of this gets verified without a human sitting in a chat window.

**Silence is the failure mode to design against.** A bot that is online and simply does
not answer gives you nothing to go on. Every ignored message says why at debug level,
and the two errors that actually happen on a first run — a bad token, and a missing
Message Content intent — are caught and explained rather than left as a traceback out
of aiohttp.
"""

from __future__ import annotations

import contextlib
import logging
import re
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Mapping

import discord
from discord.ext import voice_recv

from .chunker import EMOJI
from .config import DiscordConfig
from .discord_voice import FRAME_SAMPLES_24K, Ears, dave_decrypt, to_discord_frame
from .emotion import MARKER

log = logging.getLogger(__name__)

#: View Channel + Send Messages + Read Message History + Connect + Speak, and nothing
#: else. She is attached to a desktop that can be screenshotted, so the invite asks for
#: the least that makes a conversation work.
#:
#: Connect and Speak were missing at first, which was correct for a text-only bot and
#: became wrong the moment voice existed. The symptom was not "permission denied": the
#: voice handshake simply never completed and surfaced forty seconds later as
#: `TimeoutError` from inside `voice_state._connect`, naming nothing. `join_voice`
#: checks the two permissions first now, so the answer arrives immediately and says
#: which one is missing.
INVITE_PERMISSIONS = 1024 | 2048 | 65536 | (1 << 20) | (1 << 21)  # 3214336


@dataclass(frozen=True)
class Incoming:
    """One message, as plain values the loop can reason about."""

    text: str
    author_id: int
    author_name: str
    channel_id: int
    is_dm: bool

    @property
    def where(self) -> str:
        return "dm" if self.is_dm else f"#{self.channel_id}"


def should_reply(
    cfg: DiscordConfig,
    *,
    author_id: int,
    is_self: bool,
    is_bot: bool,
    is_dm: bool,
    channel_id: int,
    mentioned: bool,
    replied_to_her: bool,
    has_text: bool,
) -> tuple[bool, str]:
    """Decide whether this message is for her, and say why either way.

    The reason is not decoration. Every question of the form "why isn't the bot
    answering me" is answered by exactly one of these strings, and without them the
    only observable behaviour is nothing happening.
    """
    if is_self:
        return False, "her own message"
    if is_bot:
        return False, "another bot"
    if not has_text:
        return False, "no text — attachment or embed only"
    stranger = bool(cfg.owner_id) and author_id != cfg.owner_id
    # A DM is never public, whatever `public` says. Anyone who shares a server with the
    # bot can open one, so "public DMs" would mean a private channel with a stranger in
    # it — the opposite of what opening her up to a *server* is asking for.
    if stranger and is_dm:
        return False, f"not the owner, and DMs are his alone (author {author_id})"
    if stranger and not cfg.public:
        return False, f"not the owner (author {author_id})"
    if is_dm:
        return (True, "dm") if cfg.dms else (False, "dms are switched off")
    if channel_id in cfg.channels:
        return True, "a channel she listens to"
    if mentioned or replied_to_her:
        if not cfg.mentions:
            return False, "mentions are switched off"
        return True, "mentioned" if mentioned else "replied to"
    return False, "server message, not addressed to her"


#: End of a sentence, including any closing quote or bracket that belongs to it. Same
#: shape as the chunker's, and for the same reason: an emoji before the full stop reads
#: as a typo.
_SENT_END = re.compile(r"""[.!?…]+["'")\]]*""")


def apply_mood_emoji(raw: str, emoji: Mapping[str, str]) -> str:
    """Turn the first usable mood marker into an emoji at the end of its sentence.

    Runs on the raw text, *before* the markers are stripped, because a marker's position
    is the entire point: it opens the sentence it colours, so that is the sentence the
    emoji belongs at the end of. Extract first and the position is gone, leaving a choice
    between putting every emoji at the end of the message or guessing.

    One per message. She marks a mood per sentence for her face, where an expression
    changing twice in four seconds looks alive — the same rate in text reads as a
    keyboard with a sticky key.

    If she typed one herself, hers wins and nothing is added. The prompt tells her not
    to and she mostly doesn't, but "mostly" is not a guarantee and she reaches past this
    vocabulary when she does — an observed 😩 for something no marker covers. Overruling
    that would be worse than allowing it; stacking a second emoji beside it is worse
    than both.
    """
    if EMOJI.search(raw):
        return raw
    for m in MARKER.finditer(raw):
        glyph = emoji.get(m.group(1).lower(), "")
        if not glyph:
            continue  # unknown, or `neutral` asking for nothing
        stop = _SENT_END.search(raw, m.end())
        at = stop.end() if stop else len(raw)
        return f"{raw[:at].rstrip()} {glyph}{raw[at:]}"
    return raw


def split_message(text: str, limit: int) -> list[str]:
    """Cut a reply into Discord-sized pieces, breaking where a reader would.

    Rare in practice — `num_predict` caps her well under the limit — but the one reply
    that does run long is the one containing a stack trace, so it is worth the twenty
    lines. Preference order is paragraph, line, sentence, word, and only then a hard
    cut, because a message split mid-word reads as a bug in the bot.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    parts: list[str] = []
    while len(text) > limit:
        window = text[:limit]
        cut = -1
        for sep in ("\n\n", "\n", ". ", "! ", "? ", " "):
            cut = window.rfind(sep)
            if cut > limit // 4:  # not so early that we emit a sliver
                cut += len(sep)
                break
            cut = -1
        if cut <= 0:
            cut = limit  # one unbroken 1900-character token; nothing to be done
        parts.append(text[:cut].strip())
        text = text[cut:].lstrip()
    if text:
        parts.append(text)
    return _balance_fences(parts)


def _balance_fences(parts: list[str]) -> list[str]:
    """Close and reopen a code block that a split landed inside.

    Without this the first half renders as an unterminated fence swallowing everything
    after it, and the second half shows its closing ``` as literal text — which looks
    exactly like she posted something malformed.
    """
    out: list[str] = []
    open_fence = False
    for part in parts:
        if open_fence:
            part = "```\n" + part  # language tag is lost; the highlighting is not worth
        if (part.count("```") % 2) == 1:  # a token of state threaded through the split
            part += "\n```"
            open_fence = True
        else:
            open_fence = False
        out.append(part)
    return out


class Channel:
    """Somewhere to answer. The loop is handed one of these and needs nothing else."""

    def __init__(self, channel: discord.abc.Messageable, cfg: DiscordConfig):
        self._channel = channel
        self._cfg = cfg

    def typing(self):
        """`async with channel.typing()` — the honest progress bar for a local model.

        Discord has no streaming, so the alternative to this is several seconds of
        nothing followed by a wall of text. Editing a message token by token would look
        closer to the voice loop but runs straight into the per-channel edit rate limit
        and turns one reply into thirty API calls.
        """
        return self._channel.typing()

    async def send(self, text: str) -> None:
        for part in split_message(text, self._cfg.max_message):
            await self._channel.send(part)


def _is_silence(packet) -> bool:
    check = getattr(packet, "is_silence", None)
    return bool(check()) if callable(check) else False


class _Sink(voice_recv.AudioSink):
    """Received voice, handed to the same VAD thread the microphone feeds.

    **Decoding happens here, not in the extension, and that is the whole point.**
    `wants_opus()` returning False looks obviously right — let the library decode, take
    the PCM — and it is a trap: the extension decodes on its router thread, and
    `router.run` catches any exception, logs it, then calls `stop_listening()` in its
    `finally`. A single undecodable packet therefore does not cost 20 ms of audio, it
    ends reception for the rest of the call, silently, after one line in the log.

    Claiming to want Opus means the router hands the packet over untouched and we decode
    where the failure is ours to contain. One bad packet is now one dropped frame.

    Called on discord.py's receive thread, once per packet per speaker. Everyone in the
    channel is mixed into one stream: correct while he is the only one who can join, and
    the place to start when that stops being true.
    """

    def __init__(self, ears: Ears):
        super().__init__()
        self._ears = ears
        #: One decoder per speaker. Opus is stateful across a stream, so feeding two
        #: people's packets through one decoder corrupts both.
        self._decoders: dict[int, discord.opus.Decoder] = {}
        self.decoded = 0
        #: Decoded packets that came off the wire, as opposed to the ones the extension
        #: invents. **These are two different questions and conflating them cost a
        #: silent call.** `SilenceGeneratorSink` fills transmission gaps with
        #: `SilencePacket`, whose `decrypted_data` is a canned `OPUS_SILENCE` frame —
        #: it has no end-to-end layer, so it always decrypts, and it is valid Opus, so
        #: it always decodes. Counting those as evidence of hearing meant one real
        #: packet arriving was enough to start the silence generator, whose output then
        #: announced "hearing you in the call" and permanently muted the complaint
        #: below. She reported herself as listening while every word he said failed to
        #: decrypt.
        self.heard = 0
        self.failed = 0
        self._why = ""
        self._reported_at = 0.0

    @property
    def why(self) -> str:
        return self._why or "no reason recorded"

    def wants_opus(self) -> bool:
        return True

    def _dave_session(self):
        state = getattr(self.voice_client, "_connection", None)
        return getattr(state, "dave_session", None)

    def write(self, user, data) -> None:
        packet = getattr(data, "packet", None)
        raw = getattr(packet, "decrypted_data", None)
        if not raw:
            return  # silence, or a synthesised gap the extension made up

        # `decrypted_data` has only had the *transport* layer removed. On a call with
        # end-to-end encryption there is another one underneath, which the extension
        # knows nothing about — this is where "corrupted stream" came from.
        # Locally generated silence never went over the wire, so it has no end-to-end
        # layer to peel. Decrypting it would fail and throw away the very frames the
        # endpointer is waiting for.
        silence = _is_silence(packet)
        session = None if silence else self._dave_session()
        if session is not None:
            uid = getattr(user, "id", None)
            if uid is None:
                uid = self.voice_client._get_id_from_ssrc(packet.ssrc)
            if uid is None:
                return  # unattributable packet; there is no key to try
            raw = dave_decrypt(session, uid, raw)
            if not raw:
                self._missed("the end-to-end layer would not decrypt")
                return

        try:
            decoder = self._decoders.get(packet.ssrc)
            if decoder is None:
                decoder = self._decoders[packet.ssrc] = discord.opus.Decoder()
            self._ears.feed(decoder.decode(raw, fec=False))
            self.decoded += 1
            if silence:
                # Still fed to the endpointer above, and deliberately: Discord stops
                # sending entirely while nobody is talking, so these generated frames
                # are the only thing that lets a turn end. They are just not evidence
                # of anyone speaking.
                return
            self.heard += 1
            if self.heard == 1:
                # The one moment worth announcing. Everything up to here can succeed
                # while she still hears nothing — she joins, packets arrive, and each
                # one fails to decrypt. This line is the difference between "connected"
                # and "actually listening", and it is otherwise invisible.
                print("\033[33m⦿ hearing you in the call\033[0m")
        except Exception as e:  # noqa: BLE001 — one frame, never the call
            self._missed(f"Opus would not decode it ({e})")

    def _missed(self, why: str) -> None:
        """One packet lost, and a periodic complaint while *every* packet is lost.

        Quiet when it is working and loud when it is not. The failure this exists for
        looks identical to a dead microphone from the outside — she joins, packets
        arrive, none of them survive, and she simply never answers. Reporting only on
        leave was not enough: that is the moment you have already given up.

        Both failure paths come through here. The first version only complained from the
        Opus branch, so an all-failing *decryption* — the likeliest fault by far —
        returned early and said nothing at all.

        The gate is `heard`, not `decoded`, and that distinction is the whole bug: the
        extension's generated silence decodes unconditionally, so gating on `decoded`
        meant the first filled gap silenced this warning for the rest of the call.
        """
        self.failed += 1
        self._why = why
        log.debug("voice packet dropped: %s", why)

        now = time.monotonic()
        if self.heard or now - self._reported_at < 5.0:
            return
        self._reported_at = now
        print(f"\033[33m⦿ {self.failed} voice packets have arrived and none have been "
              f"usable — {why}.\033[0m")

    def cleanup(self) -> None:
        self._decoders.clear()
        self._ears.reset()


class _Source(discord.AudioSource):
    """Her voice, pulled from the same buffer the speakers would have drained.

    Deliberately the *same* buffer rather than a copy. `Playback.seconds_played` is what
    decides which words go into history when he talks over her, and a second notion of
    "how much has actually been heard" is exactly how that starts quietly lying.
    """

    def __init__(self, playback):
        self._playback = playback

    def is_opus(self) -> bool:
        return False

    def read(self) -> bytes:
        chunk = self._playback.pull(FRAME_SAMPLES_24K)
        if chunk is None:
            return b""  # end of utterance; discord.py stops the player
        return to_discord_frame(chunk)


class DiscordBot(discord.Client):
    def __init__(
        self,
        cfg: DiscordConfig,
        on_message: Callable[[Incoming, Channel], Awaitable[None]],
        on_status: Callable[[bool, str], None] | None = None,
        on_voice: Callable[[bool], None] | None = None,
        describe: Callable[[], str] | None = None,
    ):
        # Only what is actually used. `members` and `presences` may well be switched on
        # in the developer portal — that is harmless — but asking the gateway for a
        # privileged intent this bot never reads would be sloppy, and every extra one
        # is another way for login to fail with a confusing error.
        intents = discord.Intents.none()
        intents.guilds = True          # channel/guild cache; message.guild is None without it
        intents.guild_messages = True
        intents.dm_messages = True
        intents.message_content = True  # privileged, and the whole point
        intents.voice_states = True     # not privileged; lets her follow him into a call
        super().__init__(intents=intents)
        self._cfg = cfg
        self._on_message = on_message
        self._on_status = on_status
        #: Called with True when a voice call starts and False when it ends, so core can
        #: move her ears and mouth onto the network and back.
        self._on_voice = on_voice
        #: One short line saying which model and which ears she came up with. Shown as
        #: her presence, so "is she up, and how" is answerable from the member list.
        self._describe = describe or (lambda: "")
        self._ears: Ears | None = None
        self._playback = None
        #: Held so the packet counters survive until the call ends, which is
        #: when they are worth reporting.
        self._sink: _Sink | None = None

    # --- lifecycle ------------------------------------------------------------
    def preflight(self) -> str | None:
        """Catch what can be caught before spending a connection on it."""
        token = self._cfg.token.strip()
        if not token:
            return None  # no token is "Discord is off", not a problem to report
        if token.count(".") != 2:
            return (
                "ARIA_DISCORD_TOKEN doesn't look like a bot token — they come in three\n"
                "  dot-separated parts. A client secret or an application id won't work.\n"
                "  Discord Developer Portal → your app → Bot → Reset Token."
            )
        return None

    async def run_forever(self) -> None:
        """Connect and stay connected, turning the two first-run failures into English.

        discord.py reconnects on its own, so the only paths out of here are a token
        that cannot log in, an intent that was never enabled, and cancellation.
        """
        try:
            await self.start(self._cfg.token)
        except discord.LoginFailure:
            self._complain(
                "Discord refused the token.",
                "Reset it in the Developer Portal → Bot → Reset Token, then put the new",
                "one in core/.env as ARIA_DISCORD_TOKEN. Everything else keeps working.",
            )
        except discord.PrivilegedIntentsRequired:
            self._complain(
                "Discord needs the Message Content intent switched on for this bot.",
                "Developer Portal → your app → Bot → Privileged Gateway Intents →",
                "MESSAGE CONTENT INTENT. Without it she receives empty messages.",
            )
        except Exception as e:  # noqa: BLE001 — Discord must never take the voice loop down
            log.warning("discord connection ended: %s: %s", type(e).__name__, e)
            self._complain(f"Discord disconnected: {type(e).__name__}: {e}")

    def _complain(self, *lines: str) -> None:
        yellow, reset = "\033[33m", "\033[0m"
        print(f"\n{yellow}{'─' * 68}{reset}")
        for line in lines:
            print(f"{yellow}{line}{reset}")
        print(f"{yellow}{'─' * 68}{reset}\n")
        if self._on_status:
            self._on_status(False, lines[0])

    def close_nowait(self) -> None:
        """Nothing to do, and the reason is worth writing down rather than rediscovering.

        `Client.close()` is a coroutine and `VoiceLoop.shutdown()` is not — and it runs
        while the event loop is already unwinding, so a task created here would be
        cancelled before it sent anything. Restructuring `main()` around its own loop
        to fix that would buy nothing: the process exits a moment later, the OS closes
        the gateway socket, and Discord treats a closed connection as an immediate
        disconnect rather than a heartbeat it has to time out. She goes offline either
        way. Cancelling the task in `shutdown()` is the whole of it.
        """
        log.debug("discord gateway left to close with the process")

    # --- voice ----------------------------------------------------------------
    def attach_audio(self, ears: Ears, playback) -> None:
        """Hand over the two ends of the audio path, once core has built them."""
        self._ears = ears
        self._playback = playback

    @property
    def voice(self):
        """The live voice client, or None. `is_connected` because discord.py leaves a
        stale object behind after a drop."""
        vc = self.voice_clients[0] if self.voice_clients else None
        return vc if vc and vc.is_connected() else None

    @property
    def in_voice_call(self) -> bool:
        return self.voice is not None

    async def join_voice(self, channel_id: int) -> str:
        """Join a voice channel and start listening. Returns what to tell him."""
        if self._ears is None or self._playback is None:
            return "I can't join a call — my audio isn't set up."
        channel = self.get_channel(channel_id)
        if not isinstance(channel, discord.VoiceChannel):
            log.warning("voice channel %s not found or not a voice channel", channel_id)
            return "I can't find that voice channel."
        if self.voice and self.voice.channel.id == channel_id:
            return "I'm already in there."

        # Before connecting, because Discord does not refuse — it just never completes
        # the handshake, and forty seconds later `_connect` raises `TimeoutError` from
        # somewhere in voice_state naming nothing. A missing tick box should not read
        # as a network fault.
        perms = channel.permissions_for(channel.guild.me)
        missing = [n for n, ok in (("Connect", perms.connect), ("Speak", perms.speak))
                   if not ok]
        if missing:
            log.error("cannot join %s: missing %s", channel.name, " and ".join(missing))
            print(f"\n\033[33mShe can see the voice channel but isn't allowed to "
                  f"{' or '.join(m.lower() for m in missing)} in it.\n"
                  f"  Server Settings -> Roles -> her role -> enable "
                  f"{' and '.join(missing)},\n"
                  f"  or right-click the channel -> Edit Channel -> Permissions.\033[0m\n")
            return (
                f"I'm not allowed to {' or '.join(m.lower() for m in missing)} in that "
                "channel. The terminal says how to fix it."
            )

        try:
            if self.voice:
                await self.voice.move_to(channel)
            else:
                # VoiceRecvClient, not the default: receiving is an extension, and
                # asking for it at connect time is the only place it can be chosen.
                await channel.connect(cls=voice_recv.VoiceRecvClient, self_deaf=False)
            self._ears.reset()
            self._sink = _Sink(self._ears)
            # Wrapped, and this is not optional. A Discord client stops transmitting
            # entirely while you are not speaking — sender-side voice detection — so the
            # endpointer never receives the trailing silence it needs to decide a turn
            # has ended. Measured before this: she answered, correctly and in character,
            # **37 seconds** after the question, because the turn only closed when the
            # next burst of packets arrived. The generator fills the gap with real Opus
            # silence frames so the VAD sees the pause that is actually happening.
            self.voice.listen(voice_recv.SilenceGeneratorSink(self._sink))
        except Exception as e:  # noqa: BLE001
            log.exception("could not join voice")
            return f"I couldn't get into the call — {type(e).__name__}."

        log.info("joined voice channel %s", channel.name)
        if self._on_voice:
            self._on_voice(True)
        return ""

    async def leave_voice(self) -> str:
        vc = self.voice
        if not vc:
            return "I'm not in a call."
        if self._sink is not None:
            frames = self._ears.frames if self._ears else 0
            # Four numbers, because they fail in four different places and the shape of
            # the failure is which one is zero: nothing arrived (she was never sent any
            # audio), nothing was *speech* (only the generated silence got through, so
            # encryption or Opus is eating his words), nothing decoded at all, or
            # nothing reached the VAD (the re-blocking).
            print(f"\033[2m  call ended — {self._sink.heard} packets of speech, "
                  f"{self._sink.decoded - self._sink.heard} generated silence, "
                  f"{self._sink.failed} unusable, {frames} frames to the VAD\033[0m")
            if self._sink.heard == 0 and self._sink.failed:
                print(f"\033[33mShe heard nothing that whole call: {self._sink.why}."
                      f"\033[0m")
            elif self._sink.heard == 0:
                print("\033[33mNo speech ever reached her in that call — nothing was "
                      "sent to her. Check that you were unmuted and that Discord was "
                      "transmitting.\033[0m")
            self._sink = None
        try:
            if vc.is_listening():
                vc.stop_listening()
            if vc.is_playing():
                vc.stop()
            await vc.disconnect()
        except Exception:  # noqa: BLE001
            log.exception("could not leave voice cleanly")
        if self._on_voice:
            self._on_voice(False)
        return ""

    def start_speaking(self) -> None:
        """Begin a voice packet stream for the reply that is about to be written.

        Per reply rather than a permanent stream of silence: discord.py raises the
        speaking flag for as long as a source is playing, and a bot lit up as talking
        every second of the day is both wrong and rude. Starting costs nothing — the UDP
        socket is already open.
        """
        vc = self.voice
        if vc and not vc.is_playing() and self._playback is not None:
            try:
                vc.play(_Source(self._playback))
            except Exception:  # noqa: BLE001 — losing a reply beats losing the loop
                log.exception("could not start voice playback")

    async def on_voice_state_update(self, member, before, after) -> None:
        """Follow him in and out of the configured channel.

        Only his own transitions, and only that one channel: a bot that joins whatever
        call it notices is a bot that turns up uninvited.
        """
        watched = self._cfg.voice_channel
        if not watched or not self._cfg.owner_id or member.id != self._cfg.owner_id:
            return
        here, gone = after.channel, before.channel
        if here and here.id == watched and (not gone or gone.id != watched):
            await self.join_voice(watched)
        elif gone and gone.id == watched and (not here or here.id != watched):
            await self.leave_voice()

    # --- events ---------------------------------------------------------------
    async def on_ready(self) -> None:
        dim, reset = "\033[2m", "\033[0m"
        print(f"{dim}Discord: connected as {self.user}{reset}")
        if not self.guilds:
            # Worth saying loudly, because it looks like a broken bot rather than a
            # missing step: you cannot open a DM to a bot you share no server with, so
            # a bot invited nowhere is unreachable by the exact route it exists for.
            print(f"{dim}  Not in any server yet — you can't DM a bot you don't share "
                  f"one with. Invite her:{reset}")
            print(f"{dim}  https://discord.com/oauth2/authorize?client_id={self.user.id}"
                  f"&scope=bot&permissions={INVITE_PERMISSIONS}{reset}")
        if not self._cfg.owner_id:
            print(f"{dim}  ARIA_DISCORD_OWNER is unset — she'll answer anyone who can "
                  f"reach her, and won't touch the screen for any of them.{reset}")
        if self._on_status:
            self._on_status(True, str(self.user))
        await self.refresh_presence()
        await self._announce(f"I'm up — {self._describe()}." if self._describe()
                             else "I'm up.")
        await self._join_if_he_is_already_there()

    # --- up and down ----------------------------------------------------------
    async def refresh_presence(self) -> None:
        """Say what she is, in the member list.

        This is the actual answer to "is Aria up". Discord maintains it for us and gets
        it right in the case a posted message never can — a crash, a kill, a machine
        going to sleep — because it is derived from whether the gateway connection
        exists rather than from anything she remembered to say. `Listening to …` rather
        than a custom status: bots have had that activity type forever, and it happens
        to be exactly true.
        """
        try:
            await self.change_presence(
                status=discord.Status.online,
                activity=discord.Activity(
                    type=discord.ActivityType.listening,
                    name=(self._describe() or "you")[:128],
                ),
            )
        except Exception:  # noqa: BLE001 — cosmetic; never worth failing a startup
            log.debug("could not set presence", exc_info=True)

    async def _announce(self, text: str) -> None:
        """Post to the status channel, if there is one. Never fatal."""
        if not self._cfg.status_channel:
            return
        channel = self.get_channel(self._cfg.status_channel)
        if channel is None:
            log.warning("ARIA_DISCORD_STATUS_CHANNEL %s not found",
                        self._cfg.status_channel)
            return
        try:
            await channel.send(text)
        except Exception:  # noqa: BLE001
            log.warning("could not post to the status channel", exc_info=True)

    def announce_offline_sync(self) -> None:
        """Say she is going, over plain HTTP rather than the gateway.

        Shutdown here is synchronous, and it runs while the event loop is already
        unwinding — an `await` at that point raises `CancelledError` immediately, so the
        obvious async version silently never posts. It took a while to accept that: the
        message has to be sent by something that does not need the loop at all.

        Posting a message needs only a REST call and a bot token, so that is what this
        does. Three seconds, fire and forget, and any failure is swallowed — a farewell
        must never be the reason she fails to exit.

        Her presence going offline is still the real indicator, because it is derived
        from the connection existing rather than from anything she remembered to say.
        This distinguishes *meant to go* from *fell over*, which is the part Discord
        cannot tell you.
        """
        channel = self._cfg.status_channel
        if not channel or not self._cfg.token:
            return
        try:
            import httpx

            httpx.post(
                f"https://discord.com/api/v10/channels/{channel}/messages",
                headers={"Authorization": f"Bot {self._cfg.token}"},
                json={"content": "Going offline."},
                timeout=3.0,
            )
        except Exception:  # noqa: BLE001
            log.debug("could not post the offline notice", exc_info=True)

    async def _join_if_he_is_already_there(self) -> None:
        """Catch up with a call that started before she did.

        `on_voice_state_update` fires on *transitions*, so following him in and out only
        works when he moves while she is already watching. Sit in the channel first — or
        restart her mid-call, which is exactly what happens while working on this — and
        no event ever arrives. She connects, reports herself ready, and simply never
        joins. From the channel that is indistinguishable from a microphone that does not
        work, which is precisely how it was reported.
        """
        watched = self._cfg.voice_channel
        if not watched or not self._cfg.owner_id:
            return
        channel = self.get_channel(watched)
        if not isinstance(channel, discord.VoiceChannel):
            return
        if not any(m.id == self._cfg.owner_id for m in channel.members):
            return
        log.info("owner is already in %s; joining", channel.name)
        if problem := await self.join_voice(watched):
            print(f"\033[33m{problem}\033[0m")

    async def on_message(self, message: discord.Message) -> None:
        # Cache-only, deliberately: resolving an uncached reply means an API call on
        # every message that happens to be one. The cost of getting it wrong is that a
        # reply to something she said before the last restart needs an @ instead.
        replied_to_her = (
            isinstance(message.reference, discord.MessageReference)
            and isinstance(message.reference.resolved, discord.Message)
            and message.reference.resolved.author.id == self.user.id
        )
        text = message.clean_content.strip()
        ok, why = should_reply(
            self._cfg,
            author_id=message.author.id,
            is_self=message.author.id == self.user.id,
            is_bot=message.author.bot,
            is_dm=message.guild is None,
            channel_id=message.channel.id,
            mentioned=self.user in message.mentions,
            replied_to_her=replied_to_her,
            has_text=bool(text),
        )
        if not ok:
            log.debug("ignored %s from %s: %s", message.id, message.author, why)
            return

        incoming = Incoming(
            text=text,
            author_id=message.author.id,
            author_name=message.author.display_name,
            channel_id=message.channel.id,
            is_dm=message.guild is None,
        )
        try:
            await self._on_message(incoming, Channel(message.channel, self._cfg))
        except Exception as e:  # noqa: BLE001 — one bad turn, not the end of the bot
            log.exception("discord turn failed")
            with contextlib.suppress(discord.HTTPException):
                await message.channel.send(
                    f"something broke on my end — {type(e).__name__}. it's in the terminal."
                )
