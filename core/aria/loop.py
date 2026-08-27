"""The voice loop: mic in, speaker out.

Concurrency shape:
  - audio capture and VAD run on their own threads (real-time, must not miss frames)
  - Whisper and Kokoro are blocking, so they run via asyncio.to_thread
  - the LLM streams natively async
  - orchestration is asyncio

The LLM and TTS are *pipelined*: while the first sentence is playing, the second is
already being synthesised. Without that, a four-sentence reply would stall between
every sentence.
"""

from __future__ import annotations

import asyncio
import logging
import random
import sys
import threading
import time

import numpy as np

from .audio import aec as aec_mod
from .audio.capture import VoiceListener
from .audio.playback import Playback
from .audio.vad import Utterance
from .chunker import SentenceChunker, clean_for_speech, is_cjk
from .config import (
    CJK_TTS_LANGS,
    MACHINE,
    DISCORD_MOOD,
    LANGUAGES,
    DISCORD_MOOD_EMOJI,
    DISCORD_STYLE,
    EMOTION_INSTRUCTIONS,
    LANGUAGE_NOW,
    NUDGE_EXAMPLES,
    NUDGE_PROMPT,
    OWNER_IN_PUBLIC,
    PUBLIC_PERSONA,
    SCREEN_ARMED,
    SCREEN_OFF,
    Config,
    emotion_example,
)
from .discord_bot import Channel, DiscordBot, Incoming, apply_mood_emoji
from .discord_voice import Ears
from .emotion import extract as extract_emotions
from .llm import background_for, build as build_llm
from .memory import EXTRACT_PROMPT, Memory, parse_extraction
from .memory_intent import as_note, memory_command
from .persona_store import PersonaStore
from .metrics import Session, TurnMetrics
from .server.overlay import OverlayServer
from .spoken import WrittenChunk, spoken_prefix
from .state import State, StateMachine
from .wake import WakeWord
from .stt.whisper import Transcript, WhisperSTT
from .tts.kokoro_backend import KokoroTTS
from .vision import describe as vision_backends
from .vision.capture import ScreenCapture, Screenshot
from .vision.intent import capture_command, wants_screen

log = logging.getLogger(__name__)

DIM, BOLD, CYAN, GREEN, YELLOW, RESET = (
    "\033[2m", "\033[1m", "\033[36m", "\033[32m", "\033[33m", "\033[0m",
)

#: Public messages waiting on the turn lock before further ones are dropped. Everything
#: shares one lock, so an unbounded queue lets a friend with a stuck enter key starve
#: his microphone. Three is enough for a conversation and short enough that a dropped
#: message is one nobody was still waiting on.
_MAX_PUBLIC_WAITING = 3


class VoiceLoop:
    #: Whisper's language for the most recent turn, in a bilingual run. None until
    #: something has been said, and in a pinned run it stays None — there is nothing to
    #: decide per turn when the language is a setting. See `_language_now`.
    #:
    #: A class attribute as well as an instance one because the prompt-assembly tests
    #: build a loop with `__new__` and no `__init__`, deliberately: those paths are pure
    #: string work and constructing a real one would mean a microphone and 3 GB of
    #: weights. Anything they touch has to have a value without the constructor running.
    _spoken_language: str | None = None

    def __init__(self, cfg: Config):
        cfg.validate()
        self.cfg = cfg
        self.state = StateMachine(self._on_state_change)
        self.session = Session()

        self._loop: asyncio.AbstractEventLoop | None = None
        self._history: list[dict] = []
        self._spoken_language = None
        self._turn = 0
        self._speaking_task: asyncio.Task | None = None
        self._spec_task: asyncio.Task[str] | None = None
        self._spec_frames = 0
        self._spec_hit = False
        self._speaking_since: float | None = None
        self._interrupted = False
        #: Monotonic, so a clock change can't make her think he vanished for a year.
        self._last_activity = time.monotonic()
        self._nudges = 0
        self._last_nudge = ""
        self._idle_task: asyncio.Task | None = None

        self.stt: WhisperSTT | None = None
        self.llm = None
        #: Where work nobody is waiting on goes — memory extraction, once per turn.
        #: Local whenever local exists, even in cloud mode. On a free tier that is the
        #: difference between 25 and 50 conversations a day, and it keeps the most
        #: personal thing in the system off the network.
        self.background_llm = None
        self.tts: KokoroTTS | None = None
        self.playback: Playback | None = None
        self.listener: VoiceListener | None = None
        self.server: OverlayServer | None = None
        self._viseme_task: asyncio.Task | None = None
        self.capture: ScreenCapture | None = None
        self.vision: vision_backends.VisionBackend | None = None
        #: Whether Aria is currently allowed to look at the screen. Starts off, and
        #: every change is announced — see `_set_watching`.
        self._watching = False

        self.discord: DiscordBot | None = None
        self._discord_task: asyncio.Task | None = None
        #: Conversations with everyone who is not him, one per channel, and never mixed
        #: into `_history`. Separate storage rather than a filter over one list: a filter
        #: is a thing that can be forgotten at one call site, and the cost of forgetting
        #: it here is his private conversation appearing in a public channel.
        self._public: dict[int, list[dict]] = {}
        self._public_waiting = 0
        #: One turn at a time, whichever door it came in through. Without this a
        #: Discord message that lands mid-utterance interleaves two turns into one
        #: `_history` and two writers into one state machine — and the symptom is not
        #: a crash, it is her answering the wrong half of the wrong question.
        self._turn_lock = asyncio.Lock()

        self.wake = (
            WakeWord(cfg.wake.word, cfg.wake.window_s) if cfg.wake.enabled else None
        )
        self.memory = Memory(cfg.memory.path, cfg.memory.max_notes).load()
        #: An edited persona overrides the built-in, and is loaded before the first turn
        #: so a change made yesterday is who she is today.
        self.persona = PersonaStore(cfg.memory.persona_path, cfg.persona)
        cfg.persona = self.persona.load()
        #: Background extraction tasks. Held so shutdown can wait for them rather than
        #: cancelling a half-written note.
        self._memory_tasks: set[asyncio.Task] = set()

    # --- lifecycle ------------------------------------------------------------
    async def setup(self, listen: bool = True) -> None:
        """Load and warm everything.

        `listen=False` skips opening the microphone, so the pipeline can be driven from
        synthesised audio in tests without hardware.
        """
        self._loop = asyncio.get_running_loop()
        t0 = time.perf_counter()

        print(f"{DIM}Hardware: {MACHINE.describe()}{RESET}")
        print(f"{DIM}Loading Whisper ({self.cfg.stt.model}, {self.cfg.stt.device}, "
              f"{self.cfg.stt.compute_type})…{RESET}")
        self.stt = await asyncio.to_thread(WhisperSTT, self.cfg.stt)

        print(f"{DIM}Loading Kokoro ({self.cfg.tts.voice})…{RESET}")
        self.tts = await asyncio.to_thread(KokoroTTS, self.cfg.tts)
        self._announce_language()

        self.llm = build_llm(self.cfg.llm)
        self.background_llm = background_for(self.cfg.llm, self.llm)
        if self.cfg.llm.is_cloud:
            print(f"{DIM}Conversation model: {self.llm.label}{RESET}")
            local = self.background_llm is not self.llm
            print(f"{DIM}  memory extraction stays "
                  f"{'local' if local else 'in the cloud — no local model available'}"
                  f"{RESET}")
            if caveat := self.cfg.llm.provider.caveat:
                print(f"{DIM}  {caveat}{RESET}")
        self.capture = ScreenCapture(self.cfg.vision)
        self.vision = vision_backends.build(self.cfg.vision)
        self._watching = self.cfg.vision.enabled

        # Only nag about the vision backend if screen watching is actually armed;
        # a headless voice-only run should not be told to pull a 6 GB model.
        if self._watching and (problem := await self.vision.preflight()):
            print(f"\n{YELLOW}{'─' * 68}{RESET}")
            for line in problem.splitlines():
                print(f"{YELLOW}{line}{RESET}")
            print(f"{YELLOW}{'─' * 68}{RESET}\n")

        problem = await self.llm.preflight()
        if problem:
            print(f"\n{YELLOW}{'─' * 68}{RESET}")
            for line in problem.splitlines():
                print(f"{YELLOW}{line}{RESET}")
            print(f"{YELLOW}{'─' * 68}{RESET}\n")

        if self.wake is not None:
            print(f"{DIM}Wake word: she only answers to \"{self.cfg.wake.word}\", then "
                  f"stays awake {self.cfg.wake.window_s:.0f}s{RESET}")

        self.memory.begin_session()
        if self.memory.notes:
            print(f"{DIM}Memory: {len(self.memory.notes)} notes, "
                  f"session {self.memory.continuity.sessions}{RESET}")

        print(f"{DIM}Warming up…{RESET}")
        await asyncio.gather(
            asyncio.to_thread(self.stt.warmup),
            asyncio.to_thread(self.tts.warmup),
            self.llm.warmup(),
        )

        self.playback = Playback(
            self._loop,
            sample_rate=self.tts.sample_rate,
            device=self.cfg.audio.output_device,
        )
        self.playback.start()

        if self.cfg.server.enabled:
            self.server = OverlayServer(
                self.cfg.server.host,
                self.cfg.server.port,
                on_command=self._on_command,
                # A panel that connects mid-session has no idea what anything is set
                # to. Push the whole picture the moment it announces itself.
                on_hello=self.push_settings,
            )
            await self.server.start()
            self._viseme_task = asyncio.create_task(self._stream_visemes())
            print(f"{DIM}Overlay server on "
                  f"ws://{self.cfg.server.host}:{self.cfg.server.port}{RESET}")

        self._utterances: asyncio.Queue[Utterance] = asyncio.Queue()
        if listen:
            self.listener = VoiceListener(
                self.cfg, self._loop, self._utterances,
                on_speech_start=self._on_speech_start,
                on_maybe_final=self.speculate,
                playback=self.playback,
                aec=self._build_aec(),
            )
            self.listener.start()
            if self.cfg.idle.enabled:
                # Only with a live microphone. Speaking first into a headless test run
                # would be talking to nobody, by construction.
                self._idle_task = asyncio.create_task(self._idle_watch())
            # Behind the same guard, for a sharper version of the same reason: every
            # e2e suite in this repo runs `setup(listen=False)`, and a test run that
            # connects to the real gateway would answer real messages from a fixture.
            # Speaking into an empty room is embarrassing; replying to someone out of a
            # test is worse.
            self._start_discord()

        print(f"{DIM}Ready in {time.perf_counter() - t0:.1f}s{RESET}\n")
        if listen:
            self._watch_stdin()
            barge = "on" if self.cfg.barge_in.enabled else "OFF"
            print(f"{BOLD}Speak when you're ready.{RESET} "
                  f"{DIM}Barge-in {barge} — talk over her, or press Enter. "
                  f"Ctrl-C to stop.{RESET}\n")

    async def run(self) -> None:
        await self.setup()
        try:
            while True:
                utterance = await self._utterances.get()
                await self._handle_turn(utterance)
        except asyncio.CancelledError:
            raise
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        # An in-flight note is dropped rather than waited for: shutdown is synchronous
        # and cannot await one. Losing the last thing she was learning is a fair price
        # for never writing the file while it is being replaced.
        for task in list(self._memory_tasks):
            if not task.done():
                task.cancel()
        self.memory.touch()
        self.memory.save()
        if self._idle_task and not self._idle_task.done():
            self._idle_task.cancel()
        if self.discord:
            # Before the task is cancelled, though it does not need the gateway — it is
            # a plain HTTP call precisely so that ordering cannot break it.
            self.discord.announce_offline_sync()
        if self._discord_task and not self._discord_task.done():
            self._discord_task.cancel()
        if self.discord:
            self.discord.close_nowait()
        if self._viseme_task and not self._viseme_task.done():
            self._viseme_task.cancel()
        if self.server:
            self.server.close_nowait()
        if self.listener:
            self.listener.stop()
        if self.playback:
            self.playback.close()

    # --- turn handling --------------------------------------------------------
    async def _handle_turn(self, utterance: Utterance) -> None:
        """A spoken turn, serialised against the text ones.

        The lock is uncontended in a voice-only run and costs nothing there. Barge-in
        is unaffected either way: `interrupt()` cancels `_speaking_task` directly and
        never waits on this.
        """
        async with self._turn_lock:
            await self._voice_turn(utterance)

    async def _voice_turn(self, utterance: Utterance) -> None:
        self._turn += 1
        m = TurnMetrics(
            turn=self._turn,
            speech_s=utterance.duration_s,
            endpoint_ms=utterance.endpoint_delay_ms,
        )
        self.state.to(State.THINKING)
        # He is here. Reset before the wake gate, not after — speech she wasn't
        # addressed in still means the room is occupied, and calling out to someone
        # who is audibly right there is the worst version of this feature.
        self._last_activity = time.monotonic()
        self._nudges = 0

        t0 = time.perf_counter()
        heard = await self._transcribe(utterance)
        text = heard.text
        m.stt_ms = (time.perf_counter() - t0) * 1000
        m.stt_speculative = self._spec_hit

        if not text:
            log.debug("empty transcription, ignoring")
            self.state.to(State.IDLE)
            return

        # The wake gate sits here — after STT, before the LLM. Everything above this
        # line already ran, and that is the deliberate trade: transcribing the room is
        # cheap on this machine, answering it is not.
        if self.wake is not None:
            answer, text = self.wake.should_answer(text)
            if not answer:
                print(f"{DIM}you ›{RESET} {DIM}{text}  (not addressed to her){RESET}")
                self.state.to(State.IDLE)
                return
            if not text:
                self.state.to(State.IDLE)
                return

        # The detected language is shown only when detection was actually doing work.
        # In a pinned run it is a constant, and a constant printed every turn is noise.
        detected = (
            f" {DIM}[{heard.language}]{RESET}"
            if self.cfg.stt.language is None and heard.language
            else ""
        )
        print(f"{CYAN}you ›{RESET} {text}{detected}")
        # What she has to answer *this* turn in. Whisper has already decided; until this
        # was recorded, the answer went no further than the line printed above.
        self._spoken_language = heard.language
        self._history.append({"role": "user", "content": text})
        if self.server:
            # Her side already goes out as `subtitle`, but that is a caption for the
            # character — it replaces itself and carries no author. A transcript needs
            # both voices, kept, in order.
            self.server.send(type="transcript", role="you", text=text, at=time.time())

        # "watch my screen" / "stop watching" never reach the LLM — a capability this
        # invasive should turn on and off exactly when asked, not when inferred.
        if (want := capture_command(text)) is not None:
            reply = await self._set_watching(want)
            self._speaking_task = asyncio.create_task(self._respond(m, scripted=reply))
        elif (reply := self._memory_command(text)) is not None:
            self._speaking_task = asyncio.create_task(self._respond(m, scripted=reply))
        elif (notice := self._disarmed_notice(text)) is not None:
            self._speaking_task = asyncio.create_task(self._respond(m, scripted=notice))
        else:
            screenshot = await self._maybe_capture(text)
            self._speaking_task = asyncio.create_task(
                self._respond(m, screenshot=screenshot)
            )
        try:
            await self._speaking_task
        except asyncio.CancelledError:
            log.debug("response cancelled")
        finally:
            self._speaking_task = None
            # Barge-in already moved us to LISTENING — the user is mid-sentence right
            # now. Stamping IDLE over that would make the character look asleep while
            # someone is talking to it.
            if self.state.current is not State.LISTENING:
                self.state.to(State.IDLE)

        self.session.add(m)
        print(f"{DIM}{m.line()}{RESET}\n")

        # After the reply is out and the metrics are printed — deliberately last, and
        # deliberately not awaited. Only what she actually spoke is worth a note; a
        # reply cut off by barge-in was never heard.
        self.memory.touch()
        said = next(
            (msg["content"] for msg in reversed(self._history) if msg["role"] == "assistant"),
            "",
        )
        if not m.interrupted:
            self._remember_turn(said)

    def _build_aec(self) -> "aec_mod.EchoCanceller | None":
        """Echo cancellation for speaker mode, or None and a clear reason why not.

        Refusing to start is the wrong response to a missing DLL: barge-in on
        headphones is unaffected, so the loop degrades to exactly the behaviour every
        milestone before M6 shipped with.
        """
        if not self.cfg.barge_in.speaker_mode:
            return None
        ok, why = aec_mod.available()
        if not ok:
            print(f"{YELLOW}Speaker mode needs echo cancellation, which is unavailable: "
                  f"{why}.{RESET}")
            print(f"{YELLOW}Falling back to headphone behaviour — barge-in will trigger "
                  f"on Aria's own voice if you use speakers.{RESET}")
            return None
        canceller = aec_mod.EchoCanceller(
            self.cfg.audio.frame_samples,
            self.cfg.audio.sample_rate,
            self.cfg.barge_in.aec_filter_ms,
            self.cfg.barge_in.aec_speech_margin,
        )
        print(f"{DIM}Speaker mode: echo cancellation on, "
              f"{self.cfg.barge_in.aec_delay_ms} ms delay + "
              f"{self.cfg.barge_in.aec_filter_ms} ms filter "
              f"({canceller.filter_taps} taps){RESET}")
        if self.cfg.barge_in.aec_delay_ms == 0:
            print(f"{YELLOW}Round trip is set to 0 ms. If your speakers are anything "
                  f"but instant, run tests/check_aec.py — an uncompensated delay past "
                  f"the filter means nothing is cancelled at all.{RESET}")
        return canceller

    # --- memory ---------------------------------------------------------------
    def _memory_command(self, text: str) -> str | None:
        """Handle "remember…" and "forget…" without involving the LLM.

        Left to the model these get a warm agreeable reply and nothing written down,
        which is the worst possible outcome: he walks away believing she knows.
        "What do you remember" is passed through instead — the notes are already in
        her prompt, so she can answer that one in her own words.
        """
        parsed = memory_command(text)
        if parsed is None:
            return None
        action, what = parsed

        if action == "recall":
            return None  # she has the notes; let her say it however she likes

        if action == "forget":
            gone = self.memory.forget(what)
            print(f"{YELLOW}⦿ forgot {len(gone)} note(s){RESET}")
            if not gone:
                return f"I don't think I had anything about {what}."
            return "Okay, forgotten." if len(gone) == 1 else f"Okay, that's {len(gone)} gone."

        note = as_note(what)
        added = self.memory.add(note, source="you")
        print(f"{YELLOW}⦿ remembered: {note}{RESET}")
        return "Got it, I'll remember." if added else "I already knew that one."

    def _remember_turn(self, said: str) -> None:
        """Kick off background extraction for the turn that just finished.

        Never awaited by the reply path. Extraction is another 9B call and nothing
        about remembering is worth making her slower to answer.
        """
        if not self.cfg.memory.auto_extract or not said:
            return
        user = next(
            (msg["content"] for msg in reversed(self._history) if msg["role"] == "user"), ""
        )
        if not user:
            return
        task = asyncio.create_task(self._extract(user, said))
        self._memory_tasks.add(task)
        task.add_done_callback(self._memory_tasks.discard)

    async def _extract(self, user: str, said: str) -> None:
        try:
            reply = ""
            # `background_llm`, not `llm`. In cloud mode this is the only call that
            # stays on the machine, and it is the one reading his conversation to decide
            # what is worth writing down about him.
            async for token in self.background_llm.stream([
                {"role": "system", "content": EXTRACT_PROMPT},
                {"role": "user", "content": f"Him: {user}\nYou: {said}"},
            ]):
                reply += token
            note = parse_extraction(reply)
            if note and self.memory.add(note):
                print(f"{DIM}  ⦿ noted: {note}{RESET}")
        except Exception as e:  # noqa: BLE001 — a failed note must not disturb the loop
            log.debug("memory extraction failed: %s", e)

    # --- discord --------------------------------------------------------------
    def _start_discord(self) -> None:
        """Bring the bot up, or say why not and carry on without it.

        Never fatal. Discord is a second way to reach her, and losing it should cost
        exactly that — the voice loop in front of you is the product.
        """
        if not self.cfg.discord.token:
            return
        self.discord = DiscordBot(
            self.cfg.discord, self._on_discord,
            on_status=self._discord_status,
            on_voice=self._on_voice_call,
            describe=self._describe_self,
        )
        if problem := self.discord.preflight():
            print(f"\n{YELLOW}{'─' * 68}{RESET}")
            for line in problem.splitlines():
                print(f"{YELLOW}{line}{RESET}")
            print(f"{YELLOW}{'─' * 68}{RESET}\n")
            self.discord = None
            return
        # Both ends of the voice path, handed over before the gateway is up so a call
        # cannot arrive before there is anywhere to put the audio.
        self.discord.attach_audio(
            Ears(self.cfg.audio.frame_samples, self._submit_voice_frame),
            self.playback,
        )
        self._discord_task = asyncio.create_task(self.discord.run_forever())
        print(f"{DIM}Discord: connecting…{RESET}")
        if self.cfg.discord.voice_channel:
            print(f"{DIM}  voice: she'll follow you into channel "
                  f"{self.cfg.discord.voice_channel}{RESET}")

    def _submit_voice_frame(self, frame) -> None:
        """One 16 kHz frame from the call, straight onto the VAD thread's queue.

        Runs on discord.py's receive thread. `VoiceListener.submit` is thread-safe by
        construction — it is the same queue the sound-card callback already posts to
        from its own real-time thread.
        """
        if self.listener is not None:
            self.listener.submit(frame)

    def _on_voice_call(self, joined: bool) -> None:
        """Move her ears and mouth onto the network, or back to the desk.

        Exclusive both ways. He is at one desk with one headset: two open ears would
        hear every sentence twice and endpoint on whichever copy arrived first, and two
        mouths would put her voice in the room and in the call a beat apart.

        Nothing else about a turn changes — same endpointer, same speculation, same
        barge-in, same handler. Only which side of the network the audio is on.
        """
        if self.listener is not None:
            self.listener.use_microphone(not joined)
        if self.playback is not None:
            self.playback.route_external(joined)
        where = "voice call" if joined else "the desk"
        print(f"{YELLOW}⦿ listening on {where}{RESET}")
        # Her presence says which ears she is using, so it has to move with them.
        if self.discord:
            asyncio.create_task(self.discord.refresh_presence())
        if self.server:
            self.server.send(
                type="notice", level="info",
                text="In a Discord call" if joined else "Back on the desktop mic",
            )

    def _describe_self(self) -> str:
        """One line: which model, and which ears. Shown as her Discord presence.

        The mode is the part that is genuinely hard to know from outside — "is she up"
        is in the member list either way, but "did she come up on the local model or the
        cloud one" is otherwise a guess, and it changes what she costs and how she sounds.
        """
        llm = self.cfg.llm
        model = llm.active_cloud_model if llm.is_cloud else llm.model.split("/")[-1]
        where = "voice call" if (self.discord and self.discord.in_voice_call) else "mic"
        return f"you on the {where} · {model}"

    def _discord_status(self, up: bool, detail: str) -> None:
        if self.server:
            self.server.send(
                type="notice",
                level="info" if up else "error",
                text=f"Discord connected as {detail}" if up else detail,
            )
        self.push_settings()

    def _is_private(self, msg: Incoming) -> bool:
        """Is this his Aria, or the one his friends get?

        **The channel decides, not the person.** A DM with him is private; everything
        else is public, including a channel he is standing in himself.

        Deciding per-person instead is the obvious version and it is wrong. He would get
        his own history and his own notes while @mentioning her in a room full of
        friends, and the first time someone asked "what's he been up to" she would have
        the honest answer sitting in her context. Making the room the unit means the
        leak cannot happen, rather than depending on a rule she remembers to follow.

        Private also requires a *provable* owner. With `owner_id` unset nobody is him,
        so nobody gets his conversation — which closes the same hole from the other
        side.
        """
        return msg.is_dm and self._is_owner(msg)

    async def _on_discord(self, msg: Incoming, channel: Channel) -> None:
        """One Discord message, start to finish.

        `typing()` wraps the lock as well as the work: a message that arrives while she
        is mid-sentence at the desk waits its turn, and during that wait the indicator
        is the only thing distinguishing "queued" from "ignored".
        """
        private = self._is_private(msg)
        if not private:
            # Everything shares one turn lock, so a busy channel queues behind his voice
            # and his voice queues behind it. That is right up to a point and a way to
            # starve him past it: four people chatting is fine, one person hammering
            # enter means his own microphone waits minutes. Dropping is better than
            # queueing here — a reply to a message from four minutes ago is noise
            # anyway, and the console says it happened.
            if self._public_waiting >= _MAX_PUBLIC_WAITING:
                log.info("dropped a message from %s: %d already waiting",
                         msg.author_name, self._public_waiting)
                print(f"{YELLOW}⦿ dropped a Discord message from {msg.author_name} "
                      f"— {self._public_waiting} already queued{RESET}")
                return
            self._public_waiting += 1
        try:
            async with channel.typing():
                async with self._turn_lock:
                    if private:
                        await self._text_turn(msg, channel)
                    else:
                        await self._public_turn(msg, channel)
        finally:
            if not private:
                self._public_waiting -= 1

    async def _public_turn(self, msg: Incoming, channel: Channel) -> None:
        """Answer someone who is not him, out of a conversation that is not his.

        Deliberately not a branch inside `_text_turn`. Almost every line differs, and
        the ones that matter differ by being *absent*: no memory read, no memory write,
        no screen, no continuity, no character on his desktop reacting to a stranger,
        and above all no `self._history`. A flag threaded through the private path would
        put all of that one missing `if` away from leaking.

        His terminal still gets every word. She is his assistant talking to his friends
        in his server; being able to read that without opening Discord is the point.
        """
        t0 = time.perf_counter()
        history = self._public.setdefault(msg.channel_id, [])
        # Prefixed, because a channel has more than one person in it and "who said that"
        # is otherwise unrecoverable from a flat list of user turns.
        history.append({"role": "user", "content": f"{msg.author_name}: {msg.text}"})

        print(f"{DIM}{msg.author_name} ›{RESET} {msg.text} {DIM}({msg.where}){RESET}")

        keep = self.cfg.llm.history_turns * 2
        reply = ""
        # Persona by *person*, everything else by *room*. He gets her; his friends get
        # the guest. Neither gets his notes, his continuity, his screen or his private
        # conversation, because that is decided by which channel this is.
        system = self._public_system_prompt(owner=self._is_owner(msg))
        async for token in self.llm.stream(
            [{"role": "system", "content": system}, *history[-keep:]],
            num_predict=self.cfg.discord.num_predict,
        ):
            reply += token

        reply = apply_mood_emoji(reply, DISCORD_MOOD_EMOJI)
        said, _ = extract_emotions(reply, list(DISCORD_MOOD_EMOJI))
        said = said.strip()
        if not said:
            log.warning("empty public reply to %s", msg.author_name)
            said = "…nothing useful to add to that one."

        await channel.send(said)
        history.append({"role": "assistant", "content": said})
        del history[: max(0, len(history) - keep)]

        print(f"{GREEN}aria ›{RESET} {said}")
        print(f"{DIM}  public turn in {time.perf_counter() - t0:.1f}s "
              f"({msg.author_name}){RESET}\n")
        # No state change, no expression, no transcript, no speaking aloud. His character
        # should not emote at a conversation he is not in, and `--speak-discord` must
        # never let someone else's message come out of his speakers.

    async def _text_turn(self, msg: Incoming, channel: Channel) -> None:
        t0 = time.perf_counter()
        self._turn += 1
        self.state.to(State.THINKING)
        # He is somewhere, even if it isn't this room — so this counts as activity and
        # pushes the idle nudges back. Calling out loud to a desk he has just texted
        # from is the same mistake as calling out while he is talking.
        self._last_activity = time.monotonic()
        self._nudges = 0

        print(f"{CYAN}you ›{RESET} {msg.text} {DIM}(discord, {msg.where}){RESET}")
        self._history.append({"role": "user", "content": msg.text})
        if self.server:
            self.server.send(
                type="transcript", role="you", text=msg.text, at=time.time(),
                via="discord",
            )

        try:
            raw = await self._text_reply(msg)
        finally:
            if self.state.current is not State.LISTENING:
                self.state.to(State.IDLE)

        # Emoji first, markers second — `apply_mood_emoji` needs to see where the marker
        # sits, and `extract_emotions` is what removes it.
        raw = apply_mood_emoji(raw, DISCORD_MOOD_EMOJI)
        said, emotions = extract_emotions(raw, self._mood_vocabulary())
        said = said.strip()
        if not said:
            # Never leave the message unanswered. A silent bot is indistinguishable
            # from a broken one, and this is a real outcome with a 9B — an empty
            # completion, or a reply that was nothing but a marker.
            log.warning("empty discord reply for %r", msg.text[:60])
            said = "…I've got nothing. Say that again?"

        await channel.send(said)
        print(f"{GREEN}aria ›{RESET} {said}")
        print(f"{DIM}  discord turn in {time.perf_counter() - t0:.1f}s{RESET}\n")

        self._history.append({"role": "assistant", "content": said})
        if self.server:
            self.server.send(
                type="transcript", role="aria", text=said, at=time.time(),
                via="discord",
            )

        if self.cfg.discord.speak_replies and self.tts:
            # The raw text, markers and all — `_respond` schedules expressions against
            # the audio timeline, which puts them on the right sentence instead of all
            # at once. It must not write history: that already happened above, and the
            # spoken version gets truncated by barge-in while the posted one does not.
            await self._speak_aside(raw)
        elif self.server:
            # No audio to schedule against, so the face just reacts. The first marker
            # rather than the last: it is the reaction to what he said, and the ones
            # after it belong to sentences nobody is hearing timed out loud.
            #
            # Filtered against what the character can actually show, which the emoji
            # vocabulary deliberately is not — whether her Live2D model has a `flirty`
            # pose has nothing to do with whether 😏 is the right emoji. Unfiltered, a
            # mood that exists only for the text asks the overlay for an expression file
            # that isn't there.
            available = self._available_emotions()
            for value in emotions:
                if not available or value in available:
                    self.server.send(type="expression", value=value)
                    break

        self.memory.touch()
        # Dictated notes are refused above; this is the background pass, and it needs the
        # same gate for a subtler reason. It writes notes *about him* — "has a demo on
        # Friday" — inferred from whoever was talking. Run it over a stranger's
        # conversation and she quietly learns things about him that were never his.
        if self._is_owner(msg):
            self._remember_turn(said)
        else:
            log.debug("skipped memory extraction: message was not provably his")

    async def _text_reply(self, msg: Incoming) -> str:
        """Produce the words, by the same routes a spoken turn takes.

        Commands, memory and screen questions all behave identically to voice — this is
        one Aria — with a single deliberate exception, which is who is allowed to point
        her at the desktop.
        """
        text = msg.text
        owner = self._is_owner(msg)

        if (want := capture_command(text)) is not None:
            return (
                await self._set_watching(want) if owner
                else self._not_the_owner("look at that desktop", "a screen request")
            )
        if memory_command(text) is not None:
            if not owner:
                return self._not_the_owner("keep notes", "a memory command")
            if (scripted := self._memory_command(text)) is not None:
                return scripted
        if wants_screen(text) and not owner:
            return self._not_the_owner("look at that desktop", "a screen request")
        if (notice := self._disarmed_notice(text)) is not None:
            return notice

        screenshot = await self._maybe_capture(text) if owner else None
        if screenshot is not None:
            source = self.vision.describe(text, screenshot)
        else:
            source = self.llm.stream(
                self._payload(discord=True), num_predict=self.cfg.discord.num_predict
            )
        reply = ""
        async for token in source:
            reply += token
        return reply

    def _mood_vocabulary(self) -> list[str]:
        """Every mood name a typed turn might legitimately produce.

        The union of what the character can show and what has an emoji, because the two
        sets answer different questions and a marker from either is a real one. Used
        only for repairing a marker that lost a bracket — the wider the true vocabulary,
        the more debris gets caught, and a name in neither set is left alone.
        """
        return sorted(set(self._available_emotions()) | set(DISCORD_MOOD_EMOJI))

    def _is_owner(self, msg: Incoming) -> bool:
        """Is this message provably from the person all of this belongs to?

        Provably is the word doing the work. A voice in the room is self-authenticating;
        a Discord message is a string from an account, and anyone who shares a server
        with the bot can open a DM to it. With `owner_id` unset there is nothing tying
        any message to him, so the answer is no for everybody rather than yes for
        everybody.
        """
        return bool(self.cfg.discord.owner_id) and msg.author_id == self.cfg.discord.owner_id

    def _not_the_owner(self, what: str, refused: str) -> str:
        """Decline something that only the owner should be able to reach.

        Two capabilities sit behind this, and they fail in opposite directions.

        **The screen** is M5's whole design: capture armed explicitly, by the person
        whose desktop it is. The damage is immediate and obvious — a picture of his desk
        arrives somewhere it should not.

        **Memory** is quieter and, by this project's own standard, worse. A note is
        written once and read back with total confidence weeks later, with no record of
        where it came from; `memory.py` calls a wrong note worse than no note for exactly
        that reason. Left ungated, a stranger could type "remember that he hates his job"
        into a DM and she would repeat it to him in a month as something she knew.

        Both refusals are in her own voice; the terminal carries the fix, the same split
        `_set_watching` uses — a reply is not the place for an environment variable name.
        """
        print(f"{YELLOW}⦿ refused {refused} over Discord: ARIA_DISCORD_OWNER is unset "
              f"(or does not match), so nothing identifies the sender as you.{RESET}")
        return (
            f"I only {what} for one person, and I can't tell it's you from here. "
            "The terminal says how to fix that."
        )

    async def _speak_aside(self, raw: str) -> None:
        """Say a Discord reply out loud too, without it counting as a spoken turn."""
        m = TurnMetrics(turn=self._turn, speech_s=0.0, endpoint_ms=0.0)
        self._speaking_task = asyncio.create_task(
            self._respond(m, scripted=raw, record=False)
        )
        try:
            await self._speaking_task
        except asyncio.CancelledError:
            log.debug("spoken discord reply interrupted")  # he is here; let him talk
        finally:
            self._speaking_task = None
            if self.state.current is not State.LISTENING:
                self.state.to(State.IDLE)

    # --- screen awareness -----------------------------------------------------
    async def _set_watching(self, on: bool) -> str:
        """Arm or disarm screen capture, and announce it. Returns what to say.

        Arming preflights the backend, because "Okay, I can see your screen now" is a
        promise. Startup only checks when watching begins armed, which it does not by
        default — so without this the first honest report of a missing model arrives
        a turn later, attached to a question it looks like it failed to answer.
        """
        if on and (problem := await self.vision.preflight()):
            log.warning("cannot arm screen watching: %s", problem)
            if self.server:
                self.server.send(type="notice", level="warn", text=problem)
            print(f"{YELLOW}{problem}{RESET}")
            return (
                "I can't turn on screen watching — my vision model isn't set up. "
                "Have a look at the terminal, it says what's missing."
            )

        self._watching = on
        if self.server:
            self.server.send(type="vision", watching=on, capturing=False)
            self.server.send(
                type="notice",
                level="info",
                text="Screen watching ON" if on else "Screen watching OFF",
            )
        print(f"{YELLOW}⦿ screen watching {'ON' if on else 'OFF'}{RESET}")
        return (
            "Okay, I can see your screen now."
            if on
            else "Okay, I've stopped watching your screen."
        )

    def _disarmed_notice(self, text: str) -> str | None:
        """Answer a screen question that arrived while watching is off.

        Without this the question goes to the chat model with no image and nothing
        saying why, and it improvises a denial — "I can't see your screen." Which
        sounds like a missing capability rather than a switch that is off, and never
        mentions the words that would turn it on. Saying so directly costs one
        scripted line and no LLM call.
        """
        if self._watching or not wants_screen(text):
            return None
        log.debug("screen-referential question but watching is off")
        return (
            "I'm not watching your screen right now. Say watch my screen, "
            "and then ask me again."
        )

    async def _maybe_capture(self, text: str) -> Screenshot | None:
        """Grab the screen only when the utterance is actually about it."""
        if not wants_screen(text):
            return None
        if not self._watching:
            log.debug("screen-referential question but watching is off")
            return None

        if self.server:
            self.server.send(type="vision", watching=True, capturing=True)
        try:
            shot = await asyncio.to_thread(self.capture.capture)
        except Exception as e:
            log.error("screen capture failed: %s", e)
            return None
        finally:
            if self.server:
                self.server.send(type="vision", watching=True, capturing=False)

        print(f"{DIM}  captured {shot.width}x{shot.height} "
              f"from monitor {shot.monitor} ({shot.kilobytes:.0f} KB){RESET}")
        return shot

    # --- speculative transcription -------------------------------------------
    def speculate(self, audio: np.ndarray, frames: int, voiced_s: float = 0.0) -> None:
        """Start transcribing during a pause, before the turn is known to be over.

        The endpoint hold is ~600 ms of silence during which the GPU does nothing and
        the user is already waiting. STT fits inside it almost exactly, so this removes
        essentially the whole STT term from perceived latency. The cost is a wasted
        transcription whenever a pause turns out to be mid-sentence.

        **That cost is not free, and it used to compound.** `task.cancel()` cancels the
        awaiting coroutine, not the thread — `asyncio.to_thread` has no way to stop work
        already running — so an abandoned pass keeps the GPU busy alongside its
        replacement. Measured: 331 ms alone, 626 ms with one abandoned pass still
        running. A hesitant sentence with three 250 ms pauses therefore had three
        transcriptions competing, and the turn that was supposed to be *faster* paid for
        all of them. One in flight at a time; the next pause starts a fresh one.
        """
        if self.state.current is State.SPEAKING:
            return  # our own audio; barge-in handling is M2's problem
        if self._spec_task is not None and not self._spec_task.done():
            return
        self._cancel_speculation()
        self._spec_frames = frames
        self._spec_task = asyncio.create_task(
            asyncio.to_thread(self.stt.transcribe, audio, voiced_s)
        )

    def _cancel_speculation(self) -> None:
        if self._spec_task and not self._spec_task.done():
            self._spec_task.cancel()
        self._spec_task = None
        self._spec_frames = 0

    async def _transcribe(self, utterance: Utterance) -> Transcript:
        """Use the speculative result if it covered all of this utterance's audio."""
        task, covered = self._spec_task, self._spec_frames
        self._spec_task, self._spec_frames = None, 0
        self._spec_hit = False

        if task is not None:
            if covered >= utterance.frames:
                try:
                    heard = await task
                    self._spec_hit = True
                    return heard
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    log.warning("speculative STT failed, redoing: %s", e)
            else:
                # The pause was mid-sentence and the user carried on. Throw it away.
                task.cancel()
        return await asyncio.to_thread(
            self.stt.transcribe, utterance.audio, utterance.voiced_s
        )

    def interrupt(self) -> None:
        """Cut off the current reply immediately.

        Order matters. Playback is flushed *first* so `seconds_played` is frozen before
        anything else unwinds — that number is what decides which words go into history.
        Then the task is cancelled, which unwinds TTS synthesis and the LLM stream with
        it.
        """
        if self._speaking_task and not self._speaking_task.done():
            self.playback.flush()
            self._interrupted = True
            self._speaking_task.cancel()

    def _past_grace(self) -> bool:
        if self._speaking_since is None:
            return False
        elapsed_ms = (time.perf_counter() - self._speaking_since) * 1000
        return elapsed_ms >= self.cfg.barge_in.grace_ms

    def _watch_stdin(self) -> None:
        """Enter interrupts, same path as voice barge-in.

        Lets the cancellation path be exercised deterministically without speaking,
        which is how it gets tested before the overlay's stop button exists in M3.
        """
        if not sys.stdin or not sys.stdin.isatty():
            return

        def watch() -> None:
            for _ in sys.stdin:
                self._loop.call_soon_threadsafe(self.interrupt)

        threading.Thread(target=watch, name="aria-stdin", daemon=True).start()

    async def _respond(
        self,
        m: TurnMetrics,
        screenshot: Screenshot | None = None,
        scripted: str | None = None,
        record: bool = True,
    ) -> None:
        text_q: asyncio.Queue[str | None] = asyncio.Queue()
        written: list[WrittenChunk] = []
        pending: list[tuple[float, str]] = []  # (audio offset, emotion), in order
        audio_end = 0.0
        t_start = time.perf_counter()
        t_first_token: float | None = None

        self._interrupted = False
        self._speaking_since = None
        # Decided once, before generation, so it can't flip midway through a reply.
        no_questions = self._asked_last_turn()

        def source():
            """Where this turn's words come from — a fixed line, the screen, or chat."""
            if scripted is not None:
                async def fixed():
                    yield scripted
                return fixed()
            if screenshot is not None:
                question = self._history[-1]["content"]
                return self.vision.describe(question, screenshot)
            return self.llm.stream(self._payload())

        async def produce() -> None:
            nonlocal t_first_token
            chunker = SentenceChunker()
            try:
                async for token in source():
                    if t_first_token is None:
                        t_first_token = time.perf_counter()
                        m.llm_ttft_ms = (t_first_token - t_start) * 1000
                    for chunk in chunker.push(token):
                        text_q.put_nowait(chunk)
                if tail := chunker.flush():
                    text_q.put_nowait(tail)
            finally:
                # Always terminate the consumer, even if generation blew up or was
                # cancelled — otherwise it waits on the queue forever. put_nowait
                # because awaiting inside a cancelling task can raise again.
                text_q.put_nowait(None)

        async def consume() -> None:
            nonlocal audio_end
            first = True
            self.playback.begin()
            # Per reply, not a permanent stream of silence: discord.py raises the
            # speaking flag for as long as a source is playing, and a bot lit up as
            # talking all day is both wrong and rude.
            if self.discord and self.discord.in_voice_call:
                self.discord.start_speaking()
            while (raw := await text_q.get()) is not None:
                # Markers out before anything else — TTS would read them aloud, and the
                # emotion belongs to this specific sentence. The vocabulary is passed in
                # so a marker that arrived missing a bracket is recovered rather than
                # spoken; see `emotion._with_debris`.
                chunk, emotions = extract_emotions(raw, self._available_emotions())
                if not chunk:
                    continue  # a marker on its own line carries no speech
                if not first and no_questions and chunk.rstrip().endswith("?"):
                    # Three rounds of prompt wording could not stop her ending every
                    # reply with a question — including one that says "no question
                    # mark anywhere", which she cheerfully ignored six turns running.
                    # So the second consecutive one is dropped here instead. Sentence
                    # by sentence is the only place this works: by the time the whole
                    # reply exists, the first half is already coming out of the
                    # speaker. Never the opening chunk — a reply that is only a
                    # question is better than silence.
                    log.debug("dropped a follow-up question: %r", chunk)
                    continue
                if first:
                    m.chunk_ms = (time.perf_counter() - (t_first_token or t_start)) * 1000
                t_synth = time.perf_counter()
                # The marker opens the sentence it belongs to, so it colours *this*
                # chunk's delivery — the same reason the face changes here and not at
                # the end of the reply. An unmarked chunk inherits nothing and is
                # spoken flat, which is correct: not every sentence carries a feeling.
                audio = await asyncio.to_thread(
                    self.tts.synth, chunk, emotions[0] if emotions else None
                )
                if len(audio) == 0:
                    continue
                if emotions:
                    # Queue against this chunk's *start* on the audio timeline, not now:
                    # chunk two is synthesised while chunk one is still playing, so
                    # firing on arrival would land the expression a sentence early.
                    self._queue_expression(pending, audio_end, emotions[0])
                if first:
                    m.tts_ms = (time.perf_counter() - t_synth) * 1000
                    self.state.to(State.SPEAKING)
                    self._speaking_since = time.perf_counter()
                    print(f"{GREEN}aria ›{RESET} ", end="", flush=True)
                    first = False
                print(chunk + " ", end="", flush=True)
                audio_end += len(audio) / self.tts.sample_rate
                written.append(WrittenChunk(chunk, audio_end))
                self.playback.write(audio)
                if self.server:
                    self.server.send(
                        type="subtitle",
                        text=" ".join(c.text for c in written),
                        final=False,
                    )
            self.playback.end_of_stream()
            await self.playback.wait_drained()

        expressions = asyncio.create_task(self._fire_expressions(pending))
        try:
            # return_exceptions so both halves always finish. Without it, a failure in
            # produce() propagates immediately while consume() is still draining audio
            # in the background, and history gets written underneath a reply that is
            # still playing. Barge-in still works: cancelling the outer task cancels
            # the gather regardless of this flag.
            results = await asyncio.gather(produce(), consume(), return_exceptions=True)
            failures = [
                r for r in results
                if isinstance(r, BaseException)
                and not isinstance(r, asyncio.CancelledError)
            ]
            if failures:
                m.failed = True
                self._report_turn_failure(failures[0])
        finally:
            expressions.cancel()
            self._speaking_since = None
            # Record what was *spoken*, not what was generated. Chunks queued behind
            # the cut never reached the ear, and storing them would leave Aria
            # convinced she answered a question the user never heard her answer.
            said = spoken_prefix(written, self.playback.seconds_played)
            m.interrupted = self._interrupted
            if self._interrupted:
                print(f" {YELLOW}⟨cut off⟩{RESET}")
                # The console shows text at *write* time, so it runs ahead of the
                # speaker. Only this is what actually reached the ear and went into
                # history.
                if self.cfg.verbose:
                    print(f"{DIM}  heard: {said!r}{RESET}")
            else:
                print()
            # `record=False` is a reply that has already been delivered somewhere else
            # and is only being read aloud as well — a Discord message with
            # `speak_replies` on. Writing it here would double it in history, and worse,
            # would replace the text he actually received with however much of the
            # spoken version survived barge-in.
            if said and record:
                self._history.append({"role": "assistant", "content": said})
                if self.server:
                    # What was *heard*, not what was generated — a reply cut off by
                    # barge-in should read in the transcript the way it sounded.
                    self.server.send(
                        type="transcript", role="aria", text=said, at=time.time()
                    )
            if self.server:
                # The overlay's caption should end up showing what was heard, not what
                # was generated — same reasoning as the history above.
                self.server.send(type="subtitle", text=said, final=True)

    def _queue_expression(
        self, pending: list[tuple[float, str]], at_s: float, emotion: str
    ) -> None:
        available = self._available_emotions()
        if available and emotion not in available:
            # Models differ wildly in what they can show, and the prompt lists only
            # what this one has — but a local 9B will still invent one occasionally.
            log.debug("ignoring unavailable emotion %r (have: %s)", emotion, available)
            return
        pending.append((at_s, emotion))

    async def _fire_expressions(self, pending: list[tuple[float, str]]) -> None:
        """Emit each queued expression when playback actually reaches its sentence.

        This is what makes expressions land on the right words rather than a sentence
        or two early — the whole point of driving them from inline markers.
        """
        try:
            while True:
                await asyncio.sleep(0.02)
                played = self.playback.seconds_played
                while pending and pending[0][0] <= played:
                    _, emotion = pending.pop(0)
                    if self.server:
                        self.server.send(type="expression", value=emotion)
                    if self.cfg.verbose:
                        print(f"{DIM}[{emotion}]{RESET}", end="", flush=True)
        except asyncio.CancelledError:
            pass

    def _report_turn_failure(self, error: BaseException) -> None:
        """A backend fell over mid-turn. Lose the turn, not the session.

        Before this, any LLM hiccup unwound all the way out of `run()` and killed the
        process — with a traceback whose top frame was somewhere in httpx, naming
        neither Ollama nor the thing the user should do about it.
        """
        name = type(error).__name__
        if "Connect" in name or "Timeout" in name:
            advice = "Lost the connection to Ollama. Is it still running? (ollama serve)"
        else:
            advice = f"{name}: {error}"

        log.error("turn failed: %s: %s", name, error, exc_info=self.cfg.verbose)
        print(f"\n{YELLOW}⚠ {advice}{RESET}")
        print(f"{DIM}  Still listening — say something to try again.{RESET}")
        if self.server:
            self.server.send(type="notice", level="error", text=advice)

    def _available_emotions(self) -> list[str]:
        """What the currently loaded character can actually show.

        Comes from the overlay's `hello`, so it tracks a hot-swap: the three installed
        models have completely different expressive ranges, and one of them names its
        expressions `f00`-`f07`. The overlay maps semantic names onto whatever it has,
        and reports the semantic ones here.
        """
        if not self.server:
            return []
        return list(self.server.capabilities.get("emotions", []))

    #: Appended for one turn after she ends a reply with a question. Two rounds of
    #: prompt wording moved the rate from 6-in-7 to 5-in-7 and then stalled — ending
    #: on a question is trained deep into small chat models, and a standing rule in
    #: the persona competes with everything else in there. A reminder that appears
    #: only on the turn it applies to is both louder and cheaper: it sits at the very
    #: end of the prompt, so the cached prefix above it still holds.
    _NO_QUESTION = (
        "\n\nYou asked him a question last turn. Do not ask one this turn — no "
        "question mark anywhere in your reply. React to what he said and stop."
    )

    def _asked_last_turn(self) -> bool:
        last = next(
            (m["content"] for m in reversed(self._history) if m["role"] == "assistant"),
            "",
        )
        return last.rstrip().endswith("?")

    def _system_prompt(self, discord: bool = False) -> str:
        """Assembled most-stable first, because the tail is what gets reprocessed.

        Ollama caches the longest unchanged prefix, so anything volatile placed high up
        invalidates everything below it. The continuity block carries a clock to the
        minute — put it near the top, as the first version did, and the whole prompt is
        re-prefilled on every turn that crosses a minute boundary. That measured as
        perceived latency going from ~1.4 s to 2.0 s, which is the entire cost of the
        feature showing up somewhere unrelated to the feature.

        So: persona, screen rules and emotion vocabulary first — those change only when
        the character does. Then her notes, which change a few times a session. Then the
        clock, then the one-turn reminder.
        """
        prompt = self.cfg.persona + (SCREEN_ARMED if self._watching else SCREEN_OFF)
        emotions = self._available_emotions()
        if emotions:
            prompt += "\n" + EMOTION_INSTRUCTIONS.format(
                emotions="  ".join(f"[{e}]" for e in emotions),
                example=emotion_example(emotions),
            )
        prompt += self.memory.notes_block()
        prompt += self.memory.continuity_block()
        # Constant for the process, so it sits inside the cached prefix rather than
        # below the blocks that change per turn — but after the persona, because it has
        # to outrank a page of English sample lines.
        prompt += self._language_prompt()
        # Last, with the other volatile blocks, and for the same reason: the medium can
        # change from one turn to the next when he moves between the desk and his
        # phone, so anything above this stays cached across the switch. It is also the
        # loudest position in the prompt, which a rule that contradicts the read-aloud
        # rules above it needs to be.
        if discord:
            prompt += DISCORD_STYLE
            prompt += DISCORD_MOOD.format(
                moods="  ".join(f"[{m}]" for m in DISCORD_MOOD_EMOJI)
            )
        if self._asked_last_turn():
            prompt += self._NO_QUESTION
        # Dead last, and later than the block that merely says she is bilingual. The
        # rules and the samples describe who she is across a conversation; this one is
        # about the sentence she is writing right now, and it only wins if nothing
        # follows it.
        prompt += self._language_now(self._last_user_text() if discord else None)
        return prompt

    def _public_system_prompt(self, owner: bool = False) -> str:
        """What she is working from in a shared channel.

        Built from scratch rather than by subtraction. `_system_prompt` assembles the
        persona, the screen rules, her notes and the continuity block, and every one of
        those is either about him or about a capability strangers do not get — so the
        safe version is the one where they were never added, not the one where they were
        removed and the removal has to keep being correct as the function grows.

        **Who she is and what she can reach are two different questions**, and conflating
        them was a real mistake. The room decides what she can reach, and that is the
        guarantee worth having: no notes, no continuity, no screen, and a conversation
        kept per channel, so nothing private can surface in a room his friends are in.
        The *person* decides who she is. Handing him the guest persona in his own server
        bought no safety at all — the persona contains nothing about him — and made her
        read as "just the model itself", which is precisely what it was written to be.
        """
        persona = (self.cfg.persona + OWNER_IN_PUBLIC) if owner else PUBLIC_PERSONA
        # Built from scratch, which is what makes it safe — and is also how it silently
        # missed the language block when that was added to `_system_prompt` alone. He
        # asked her 講中文 in a channel and got "I don't know what you're saying dude,
        # can you speak English": the persona was right, the language was not, and the
        # reason was that this path assembles its own prompt.
        return (
            persona
            # Only a stranger loses the samples. In his own server he is still himself,
            # and `OWNER_IN_PUBLIC` narrows what she can *reach*, never who she is.
            + self._language_prompt(guest=not owner)
            + DISCORD_STYLE
            + DISCORD_MOOD.format(moods="  ".join(f"[{m}]" for m in DISCORD_MOOD_EMOJI))
            # Same reasoning, and the same trap this path already fell into once with
            # the language block above: assembling from scratch is what makes it safe,
            # and is also what makes it silently miss anything added to `_system_prompt`
            # alone. These people are typing, so the script they typed in decides.
            + self._language_now(self._last_user_text())
        )

    def _announce_language(self) -> str:
        """Say which language she is in, because getting it wrong is silent otherwise.

        A run pinned to the wrong language does not error: Whisper told the audio is
        English hears Chinese as confident English nonsense, and the fault presents as
        a broken microphone. One line at startup is the cheapest possible defence.
        """
        if self.cfg.stt.language is None:
            cjk, latin = self.cfg.tts.alt_voice, self.cfg.tts.voice
            if self.cfg.tts.lang in CJK_TTS_LANGS:
                cjk, latin = latin, cjk
            line = (f"Language: whichever you speak — 繁體中文 ({cjk}) "
                    f"or English ({latin})")
        else:
            named = {"zh": "繁體中文 only", "en": "English only"}
            line = f"Language: {named.get(self.cfg.language, self.cfg.language)}"
        print(f"{DIM}{line}{RESET}")
        return line

    def _language_prompt(self, guest: bool = False) -> str:
        """The language block, for every path that builds a prompt.

        A function rather than a constant because there are four such paths and they do
        not share an assembly step. Anything that talks to the model has to call this or
        she answers in English, and only one of the four is the one people notice.

        `guest` takes the rule without the sample lines. Those samples are her voice
        with the person she belongs to — 老兄 and 兄弟 are 'dude' and 'mate' — and the
        guest persona bans precisely that. See `Language.guest_prompt`.
        """
        lang = LANGUAGES.get(self.cfg.language)
        if not lang:
            return ""
        return (lang.guest_prompt or lang.prompt) if guest else lang.prompt

    def _language_now(self, text: str | None = None) -> str:
        """One line naming the language *this* turn has to be answered in.

        Only in a bilingual run: pinned to `en` or `zh`, the language block already says
        the same thing unconditionally and repeating it per turn buys nothing.

        `text` is for the paths where nothing was spoken. A Discord message arrives as
        characters and its script is the only evidence there is — and it must not fall
        back to the last thing *said*, because moving between the desk and a phone is
        exactly when the two disagree.
        """
        if self.cfg.stt.language is not None:
            return ""
        code = "zh" if is_cjk(text) else "en" if text is not None else self._spoken_language
        return LANGUAGE_NOW.get(code or "", "")

    def _last_user_text(self) -> str:
        return next(
            (m["content"] for m in reversed(self._history) if m["role"] == "user"), ""
        )

    def _payload(self, discord: bool = False) -> list[dict]:
        keep = self.cfg.llm.history_turns * 2
        return [
            {"role": "system", "content": self._system_prompt(discord)},
            *self._history[-keep:],
        ]

    # --- callbacks ------------------------------------------------------------
    def _on_speech_start(self) -> None:
        if self.state.current is State.SPEAKING:
            if self.cfg.barge_in.enabled and self._past_grace():
                self.interrupt()
                self.state.to(State.LISTENING)
            return
        if self.state.current is State.IDLE:
            self.state.to(State.LISTENING)

    def _on_state_change(self, old: State, new: State) -> None:
        if self.cfg.verbose:
            print(f"{DIM}[{old} → {new}]{RESET}")
        if self.server:
            self.server.send(type="state", value=str(new))

    # --- speaking first -------------------------------------------------------
    def _may_nudge(self) -> bool:
        """Every reason to stay quiet, in one place.

        The timer is the easy part. What makes an unprompted voice tolerable rather
        than unnerving is the list of moments it must not arrive in — and each of
        these was a way to be genuinely unpleasant rather than a theoretical edge.
        """
        return not (
            self._speaking_task is not None            # mid-reply already
            or self.state.current is not State.IDLE    # listening or thinking
            # A turn is in flight through some other door — he is typing to her on
            # Discord right now. Calling out to an empty room while mid-conversation
            # somewhere else is the same mistake as talking over him.
            or self._turn_lock.locked()
            # Muted means he deliberately took her hearing away. Calling out to
            # someone who has silenced you, and who cannot answer, is a tantrum.
            or (self.listener and self.listener.is_muted)
            or self._nudges >= self.cfg.idle.max_nudges
        )

    async def _idle_watch(self) -> None:
        """Speak first once the room has been quiet long enough."""
        while True:
            wait = self.cfg.idle.after_s * (self.cfg.idle.backoff ** self._nudges)
            await asyncio.sleep(max(5.0, wait - (time.monotonic() - self._last_activity)))
            if time.monotonic() - self._last_activity < wait or not self._may_nudge():
                continue
            try:
                await self._nudge()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 — never take the loop down for this
                log.warning("idle nudge failed: %s", e)
                self._nudges += 1

    async def _nudge(self) -> None:
        """Generate one line and say it, as an ordinary turn.

        Takes the turn lock like any other turn. `_may_nudge` already checked it was
        free, but the check and the acquisition are not one operation — a Discord
        message can arrive in the gap, and this is the one turn where losing that race
        means talking over a conversation she is already having.
        """
        async with self._turn_lock:
            await self._nudge_locked()

    async def _nudge_locked(self) -> None:
        self._nudges += 1
        shown = random.sample(NUDGE_EXAMPLES, 3)
        prompt = NUDGE_PROMPT.format(examples="\n".join(f'  "{e}"' for e in shown))
        line = ""
        async for token in self.llm.stream([
            {"role": "system", "content": self._system_prompt() + prompt},
            {"role": "user", "content": "(silence)"},
        ]):
            line += token
        line = clean_for_speech(line).strip().strip('"')
        if not line or line == self._last_nudge:
            return  # saying the same thing twice running is worse than saying nothing
        self._last_nudge = line

        print(f"{YELLOW}⦿ speaking first{RESET}")
        self._turn += 1
        m = TurnMetrics(turn=self._turn, speech_s=0.0, endpoint_ms=0.0)
        self._speaking_task = asyncio.create_task(self._respond(m, scripted=line))
        try:
            await self._speaking_task
        except asyncio.CancelledError:
            log.debug("nudge interrupted")  # he answered, which is the point
        finally:
            self._speaking_task = None
            if self.state.current is not State.LISTENING:
                self.state.to(State.IDLE)
        self.session.add(m)
        # Deliberately not counted as activity: an unanswered nudge must not reset the
        # clock, or she calls out forever at a fixed interval into an empty room.

    # --- control panel --------------------------------------------------------
    def settings_snapshot(self) -> dict:
        """Everything the panel needs to render, in one message.

        The protocol is fire-and-forget by design — no request/response — so rather
        than let the panel ask questions, core states the whole picture whenever it
        changes. The panel is still a renderer; it just renders settings as well as a
        face.
        """
        return {
            "type": "settings",
            "voice": self.cfg.tts.voice,
            "voices": self.tts.voices() if self.tts else [],
            # The other script's voice, so the panel can show which one it just changed
            # rather than appearing to ignore the click. Empty in a one-language run.
            "alt_voice": self.cfg.tts.alt_voice,
            "language": self.cfg.language,
            "emotion_voice": self.cfg.tts.emotion_voice,
            "muted": bool(self.listener and self.listener.is_muted),
            "watching": self._watching,
            "vision_backend": self.cfg.vision.backend,
            "wake_enabled": self.wake is not None,
            "wake_word": self.cfg.wake.word,
            "speaker_mode": self.cfg.barge_in.speaker_mode,
            "barge_in": self.cfg.barge_in.enabled,
            #: Read-only for now — there is no panel control that would make sense.
            #: "Turn Discord off" from a phone you are holding is a way to lock
            #: yourself out of the thing you are holding it for.
            "discord": {
                "configured": bool(self.cfg.discord.token),
                "connected": bool(self.discord and self.discord.is_ready()),
                "user": str(self.discord.user) if self.discord and self.discord.user else "",
                "owner_set": bool(self.cfg.discord.owner_id),
                "speak_replies": self.cfg.discord.speak_replies,
            },
            "memory": [
                {"id": n.id, "text": n.text, "source": n.source,
                 "created_at": n.created_at}
                for n in sorted(self.memory.notes, key=lambda n: n.created_at)
            ],
            "sessions": self.memory.continuity.sessions,
            #: The editable half of who she is. The rest of the prompt is machinery.
            "persona": self.cfg.persona,
            "persona_is_custom": self.persona.is_custom,
            #: Read-only, and the most useful thing in the panel when she is behaving
            #: oddly: the whole assembly she actually receives, notes and clock and
            #: screen rules included. Built as a spoken turn, which is the common case —
            #: a typed one differs only by the block at the very end.
            "system_prompt": self._system_prompt(),
        }

    def push_settings(self) -> None:
        if self.server:
            self.server.send(**self.settings_snapshot())

    def _notify(self, text: str, level: str = "info") -> None:
        """Say something back to whoever pressed the button.

        A save that silently does nothing is the failure this whole panel is built to
        avoid — the same reason nothing is drawn optimistically. An edit that was
        refused has to say so.
        """
        if self.server:
            self.server.send(type="notice", level=level, text=text)

    def _on_command(self, msg: dict) -> None:
        """Commands from the overlay's strip, panel, tray menu or hotkeys.

        Every branch ends by pushing a fresh snapshot rather than trusting the panel
        to predict the result. A toggle that renders optimistically and then diverges
        from the thing it controls is worse than no toggle.
        """
        name = msg.get("name", "")
        value = msg.get("value")

        if name == "stop":
            self.interrupt()  # same path as voice barge-in
            return
        elif name in ("mute", "unmute") and self.listener:
            self.listener.set_muted(name == "mute")
            self.server.send(type="notice", level="info", text=f"Microphone {name}d")
        elif name == "set_voice" and value in (self.tts.voices() if self.tts else []):
            # Hot swap: Kokoro takes the voice per call, so nothing needs reloading and
            # the change lands on her next sentence.
            #
            # She has two voices when she speaks two languages, and the picker shows one
            # flat list of 54. So the chosen voice replaces the one for *its own* script:
            # picking `zm_yunxi` from a list while she is answering in English should
            # change her Chinese voice, not silence her Chinese by pointing both slots at
            # a Mandarin voice and leaving Han text with nowhere to go.
            slot_is_cjk = value.startswith(("zf_", "zm_"))
            if self.cfg.tts.alt_voice and slot_is_cjk != (self.cfg.tts.lang in CJK_TTS_LANGS):
                self.cfg.tts.alt_voice = value
            else:
                self.cfg.tts.voice = value
            self.server.send(type="notice", level="info", text=f"Voice: {value}")
        elif name == "set_emotion_voice":
            self.cfg.tts.emotion_voice = bool(value)
        elif name == "set_watching":
            asyncio.create_task(self._set_watching_from_panel(bool(value)))
            return  # that path pushes its own snapshot when the preflight finishes
        elif name == "set_wake":
            self.wake = WakeWord(self.cfg.wake.word, self.cfg.wake.window_s) if value else None
            self.cfg.wake.enabled = bool(value)
        elif name == "set_barge_in":
            self.cfg.barge_in.enabled = bool(value)
        elif name == "forget" and value:
            gone = self.memory.forget(str(value))
            log.info("panel forgot %d note(s)", len(gone))
        # Memory editing from the panel works by id, never by text. `forget` matches
        # loosely on purpose — said out loud it should catch a note however it happens
        # to be worded — but a button next to one specific line must take that line and
        # nothing else.
        elif name == "add_note" and value:
            added = self.memory.add(str(value), source="you")
            self._notify("Noted." if added else "She already knew that one.")
        elif name == "edit_note" and isinstance(value, dict):
            ok = self.memory.update(str(value.get("id", "")), str(value.get("text", "")))
            if not ok:
                self._notify("Couldn't save that note — too short, too long, or gone.",
                             level="warn")
        elif name == "delete_note" and value:
            self.memory.remove(str(value))
        elif name == "set_persona" and isinstance(value, str):
            ok, message = self.persona.save(value)
            if ok:
                # Takes effect on the next turn by construction: the system prompt is
                # assembled per turn from `cfg.persona`. Nothing reloads and the
                # conversation is kept.
                self.cfg.persona = self.persona.load()
                print(f"{YELLOW}⦿ persona updated from the panel{RESET}")
            self._notify(message, level="info" if ok else "warn")
        elif name == "reset_persona":
            ok, message = self.persona.reset()
            if ok:
                self.cfg.persona = self.persona.load()
                print(f"{YELLOW}⦿ persona reset to the built-in{RESET}")
            self._notify(message, level="info" if ok else "warn")
        elif name == "settings":
            pass  # a plain refresh request
        else:
            log.debug("unhandled overlay command: %s", name)

        self.push_settings()

    async def _set_watching_from_panel(self, on: bool) -> None:
        """Arming from the panel goes through the same preflight as arming by voice.

        Skipping it would let a button promise something the first screenshot fails to
        deliver — the exact failure M5 exists to prevent, reintroduced through a
        different door.
        """
        reply = await self._set_watching(on)
        self.server.send(type="notice", level="info", text=reply)
        self.push_settings()

    async def _stream_visemes(self) -> None:
        """Drive the mouth from the audio actually being played.

        Amplitude, not text: it costs nothing, stays in sync by construction, and looks
        convincing. Phoneme-accurate visemes would need timings Kokoro does not expose.
        """
        interval = 1.0 / self.cfg.server.viseme_hz
        gain = self.cfg.server.viseme_gain
        smoothed = 0.0
        last_sent = -1.0
        try:
            while True:
                await asyncio.sleep(interval)
                raw = (
                    self.playback.level
                    if self.state.current is State.SPEAKING
                    else 0.0
                )
                target = min(1.0, raw * gain)
                # Fast attack, slower release: mouths open quickly and close smoothly,
                # and it stops the jaw chattering on frame-to-frame RMS jitter.
                smoothed = target if target > smoothed else smoothed * 0.55 + target * 0.45
                # The release is exponential, so it approaches zero without ever
                # arriving. Left alone the mouth rests very slightly open forever.
                if smoothed < 0.015:
                    smoothed = 0.0
                value = round(smoothed, 3)
                if abs(value - last_sent) >= 0.02 or (value == 0.0 and last_sent != 0.0):
                    self.server.send(type="viseme", open=value)
                    last_sent = value
        except asyncio.CancelledError:
            pass
