# Roadmap

Ordered so that each milestone is independently testable and the riskiest work happens
early. The temptation is to build the character first because it is the fun part — resist
it. The voice loop is the spine, and it is where the project can actually fail.

## M1 — Voice loop, no character ✅

**Goal: hold a spoken conversation with a terminal.**

- [x] Mic capture + Silero VAD endpointing
- [x] faster-whisper transcription on GPU
- [x] LLM streaming (Ollama)
- [x] TTS with sentence chunking and streaming playback
- [x] The four-state machine
- [x] Speculative transcription during the endpoint hold

**Result: 1323 ms median**, end of speech to first audio. Target was < 1200 ms, so:
close, and honestly described as usable but slightly sluggish rather than snappy.

The budget is not compute-bound. STT costs 0 ms of perceived latency because it runs
speculatively inside the VAD hold, and the largest remaining terms are the hold itself
(600 ms, a tuning decision) and Kokoro's first chunk (437 ms). Full breakdown and the
reason cutting the hold does *not* help are in [core/README.md](../core/README.md).

Nothing here forces a redesign, which was the question this milestone existed to answer.

## M2 — Interruption ✅

**Goal: cut it off mid-sentence and have it stop cleanly.**

- [x] Barge-in detection during playback (headphones assumed)
- [x] Full cancellation path: playback → TTS → LLM stream
- [x] History records what was *spoken*, not what was generated
- [x] Enter as a button-triggered equivalent, for deterministic testing
- [x] `--no-barge-in` for anyone on open speakers until M6

**Result:** playback stops **5 ms** after the trigger. In the test case Aria generated
113 characters and only 20 of them reached the ear; history recorded exactly those 20.
Asked "sorry, go on", she gave the full account rather than assuming she had already
given it — which is the whole point.

The truncation is the part that matters and the part that is easy to get wrong.
`Playback.seconds_played` freezes at the cut, and `spoken.spoken_prefix` maps that back
to words, truncating a straddling chunk proportionally. Get this wrong and the failure
is invisible: Aria believes she answered, the user knows she did not, and the
conversation quietly degrades over the next few turns.

## M3 — The character appears ✅

**Goal: something on screen that moves when it talks.**

- [x] Core WebSocket server broadcasting `state`, `viseme`, `subtitle`, `notice`
- [x] Setup script fetching Cubism Core (not redistributable) plus the Haru model
- [x] Electron transparent, always-on-top, click-through window
- [x] PixiJS + `pixi-live2d-display` rendering the model
- [x] WebSocket client consuming `state` and `viseme`
- [x] Amplitude lip sync into `ParamMouthOpenY`
- [x] Idle animation running locally — breathing, blinking, gaze toward the cursor

**Result:** verified end to end against a live overlay — `thinking → speaking → idle`
plus 350+ viseme frames across three real turns, driving the mouth from actual generated
speech.

Offline behaviour confirmed in both directions: the overlay was launched repeatedly with
no core running (idle animation continues, character desaturates, reconnect backs off),
connected mid-run when core came up, and returned to offline cleanly when core exited.

Started on Haru, an official Live2D sample, rather than commissioning art. Custom art is
real money and real time, worth spending only once the system is worth looking at —
which it now is.

**Carried into M4:** the model reports its capabilities as `f00`–`f07` for expressions
and `Idle` / `Tap` for motion groups — not the `F01`–`F08` the filenames imply. The
overlay already sends the real list in its `hello`, so M4 should constrain the LLM to
that rather than to a hardcoded vocabulary.

## M4 — Expression control ✅

**Goal: it emotes appropriately without being told to.**

- [x] Inline `[emotion]` markers in the LLM output, stripped before TTS
- [x] Expression events, scheduled against the audio timeline
- [x] Motion clips, for characters whose expressions *are* motions
- [x] The character's real emotion set injected into the system prompt
- [x] Semantic emotion layer, so the vocabulary survives a hot-swap

**Result:** `[happy]` fires at +0 ms as speech begins, `[sad]` at **+4073 ms** — waiting
for its own sentence rather than firing early with the rest of the generated text. That
was the milestone's criterion and it is measured, not assumed
(`tests/e2e_expression.py`, 9/9).

Blending needed no work: `pixi-live2d-display` cross-fades expressions over 1 s when an
`.exp3.json` declares no fade time, which all of ours do.

**The design problem was vocabulary, not plumbing.** The three characters have wildly
different expressive ranges, and one names its expressions `f00`–`f07` — meaningless to
an LLM. So core speaks *semantic* emotions (`happy`, `shy`, `angry`) and the overlay
owns the translation to whatever the loaded character actually has, reporting the
semantic set in its `hello`. Swap character mid-conversation and the system prompt
changes with it.

Emotions the current character lacks are dropped in core rather than sent, and the
worked example in the prompt is built from the available set — demonstrating a marker
the character cannot show teaches the model to emit something that gets silently
discarded.

## M5 — Screen awareness ✅

**Goal: it can answer "what am I looking at?"**

- [x] `mss` capture, downscale, monitor selection
- [x] On-demand trigger from intent, plus a change gate
- [x] Visible capture-state indicator on the character, defaulting to off
- [x] Two vision backends behind one interface (local Ollama, Claude API)
- [x] A working vision answer — `qwen2.5vl:7b` pulled, screen described correctly

**The privacy path is done and verified** (`tests/e2e_vision.py`, 10/10): watching
starts off, a screen question while disarmed captures nothing, arming and disarming
work by voice, an ordinary question never captures, and every state change reaches the
overlay. Capture measures 200–390 ms including downscale and PNG encode.

**The answer works.** With `qwen2.5vl:7b` pulled, a real capture of monitor 1
(1500×844) came back correctly identified on the first try.

**The default is the local model, and privacy is the reason.** Screen capture is the
most invasive thing here; sending those frames to an API is a real trade, and the
local path keeps them on the machine. Claude Sonnet 5 is wired as the alternative for
when reading small UI text matters more — one env var, no code change.

**Building the Claude path found two defects worth keeping recorded.**
`claude_max_edge` was dead config: `_downscale` always read `max_edge`, so the Claude
path would have sent 1500px images and silently thrown away the high-resolution vision
that is most of the reason to use it. `VisionConfig.edge` now picks by backend. And
preflight passed on a key with no credit, because `models.retrieve` never touches
billing — reachability is not usability, so preflight now spends one token proving the
account works rather than promising what the first screenshot will fail to deliver.

The Claude backend is verified up to the API call and blocked there on an empty
balance, not on code.

**Vision turns are not conversational speed.** Measured end to end:

| | Cold (first turn after load) | Warm |
|---|---|---|
| Capture | 241 ms | 172 ms |
| First token | 17.4 s | 3.5 s |
| Full answer | 18.1 s | 3.9 s |

The ~17 s cold figure is the model loading into VRAM, paid once. Even warm, ~3.9 s is
well outside the ~1.3 s budget a normal turn holds to — acceptable for "what's this
error?", but it will not feel like chat. Streaming means TTS starts at first token
rather than at the end, which is what makes the wait bearable. Keeping the model
resident (`OLLAMA_KEEP_ALIVE`) is the lever if the cold load lands mid-conversation.

**Being off has to be legible.** Defaulting to off is right, but the first version
said nothing when a screen question arrived disarmed: the question went to the chat
model with no image, and it improvised "I can't see your screen." That reads as a
capability Aria lacks rather than a switch the user can flip, and it never mentioned
the words that would flip it. Now a disarmed screen question gets a scripted line
naming them, and the chat model's own prompt states whether watching is on — so the
answers it improvises are at least true. Guarded by an eleventh check in
`tests/e2e_vision.py`.

**Design notes.** Capture is on-demand and gated on intent, never a feed — a keyword
test rather than an LLM call, so it adds no latency and fails obviously rather than
mysteriously. False negatives are the safe direction: a missed reference means Aria
answers without looking and you rephrase; a false positive sends a screenshot nobody
asked to send. "Watch my screen" and "stop watching" bypass the LLM entirely.

## Memory and identity ✅

**Goal: she still knows him tomorrow — and admits it when she doesn't.**

These are one problem seen from two sides. The persona told her to notice things and
bring them up later, but there was no later, so she invented one: *"That's the third
time this week, isn't it?"* on a fresh process with an empty history. Charming exactly
once, unsettling the moment you notice. Real recall is half the fix; the other half is
a rule that she never invent shared history, and a continuity block that tells her
truthfully whether they have met at all.

**Two ways in, because each covers the other's failure.** A background pass after each
turn extracts anything durable, and `remember that…` / `forget…` are caught before the
LLM ever sees them — the same reasoning as the screen commands in M5. Left to a 9B,
"remember I take my coffee black" earns a warm agreeable reply and nothing written
down, which is the worst outcome available: he walks away believing she knows.

**The extractor fails closed.** Told to find anything interesting, a 9B records that he
said hello. A wrong note is worse than no note — she repeats it back weeks later with
total confidence and he has no idea where it came from — so it answers `NONE` unless
the fact is durable, about him, and not guessed at, and replies in the first or second
person are discarded as her talking rather than observing.

**Notes he dictates are quoted, not paraphrased.** The obvious approach rewrites the
pronouns into third person to match the extractor's style, and produces "he take his
coffee black" — swapping *I* for *he* leaves the verb behind. Conjugating properly
needs a verb table, which is a queue of new bugs. Quoting sidesteps grammar entirely
and is more honest: months later he can tell what he said apart from what she inferred.

**Everything is his to read and delete.** Plain JSON at `core/data/memory.json`, inside
the project rather than buried in AppData, gitignored. A corrupt file is moved aside
rather than overwritten. Eviction at 120 notes drops the oldest thing she guessed at
and never something he asked for by name.

Writing is off the critical path — extraction is another 9B call and nothing about
remembering is worth a slower reply. Verified by `tests/test_memory.py` (27) and
`tests/e2e_memory.py` (11/11), which teaches her across one session and reads it back
from disk in the next. The e2e suites now point at a throwaway memory file: a test run
that leaves "he has a demo on Friday" in the file she loads tomorrow is a bug.

## M6 — Speaker-mode barge-in ✅

**Goal: drop the headphones.**

- [x] Echo cancellation with the playback stream as reference
- [x] Sample alignment between capture and playback on Windows
- [x] Residual suppression, without which the above is not enough

Run it with `--speaker-mode`. Off by default: it loads a native DLL, and the headphone
path that every earlier milestone was built against should not depend on one.

**WebRTC AEC does not install on Windows.** `webrtc-audio-processing` ships a setup.py
that assumes a POSIX layout and dies mid-build; `speexdsp` needs SWIG and headers that
aren't there. The way through was `pyaec`, which ships a prebuilt `aec.dll` for
x86_64-msvc — SpeexDSP's canceller wrapped by the Rust `aec-rs` crate. Its import
table is KERNEL32, the C runtime and dbghelp: no sockets, no HTTP.

**The alignment problem is a rate problem.** Kokoro plays at 24 kHz and the mic runs
at 16 kHz, so the reference is resampled 2:3 once per chunk on the synthesis thread —
not per frame on the audio callback, which has 21 ms to fill a block and no business
running a convolution. The reference is keyed to `_samples_played`, the device's own
count, rather than to the queue: queued-but-unplayed audio is a signal the room has
not heard yet, and the filter never converges against it.

**A linear canceller alone does not fix this, and more filter does not help.** Against
real Kokoro speech through a simulated room, Speex plateaus at ~14 dB whether the tail
is 200 ms or 800 ms — and Silero still hears speech in that residual, which is the
self-interruption bug fully intact. What closes it is a second stage: compare each
frame against the echo floor the canceller is currently achieving, and zero anything
that merely tracks the reference. Two details were load-bearing, and both were found
by measurement after the obvious version failed:

- **Compare against an envelope, not the current frame.** The echo arrives a round
  trip late, so in the gap between two of Aria's words the reference is silent while
  her previous word is still hitting the mic. An instantaneous ratio divides a loud
  residual by nothing and opens the gate on exactly the frames it exists to close.
- **Learn the typical echo ratio, not the minimum.** The first version followed the
  floor downward, settled two orders of magnitude below the real echo level, and
  passed everything. It now learns only from frames it has already classified as echo.

**Filter length is a cliff, not a slope.** Echo arriving past the filter isn't
degraded, it's untouched: 200 ms of filter let a 250 ms echo trigger barge-in twice,
where 400 ms silenced it.

**The bulk delay has to be compensated, not absorbed.** The first version of this
milestone passed on the synthetic bench and on a machine whose speakers turned out not
to be plugged in. With a real speaker connected the round trip measured **512 ms**
against a 400 ms filter, so the echo landed entirely outside it: 11 dB removed, and
cancellation made the trigger count *worse* than no cancellation at all. The fix is
`aec_delay_ms`, which walks the reference read head back by the round trip so the
filter only models the room's tail. Same recording, same filter: 0 triggers and 65 dB
removed. `tests/check_aec.py` measures the number, and the estimate wanders 512-560 ms
between runs — the filter absorbs that much misalignment easily, which is why the
compensation only has to be roughly right.

**The speech margin has a narrow window, and both walls were found by measurement.**
Cancellation is worst exactly where speech restarts after a pause: on the real
recording the filter mismatched on that transient and leaked residual at 4-10x the
learned echo floor for seven consecutive frames, which is a self-interruption. At a
margin of 3 that leaks; by 10 the user cannot interrupt at all, even at full volume.
6 clears the transient and still hears a user at quarter volume.

**The trade, stated plainly.** Near-end speech quieter than the residual echo is
indistinguishable from it, so whispering at Aria mid-sentence will not interrupt her.
Speaking normally will. Transcribing the overlapping fragment is not a goal — barge-in
stops playback within a frame or two, and the rest of the utterance arrives clean.

Verified by `tests/e2e_aec.py` (7/7): echo alone triggers barge-in without
cancellation and is silent with it, real speech still gets through, and both hold at
20/120/250 ms delays. `tests/check_aec.py` measures the real room, since a simulated
one cannot tell you about your speakers.

## Later

Things worth wanting, none of which should be started before M6:

- ~~**Memory**~~ ✅ — see below. Still "append to a file"; retrieval becomes necessary
  only when the notes outgrow the context window, which 120 of them do not.
- **Tools** — timers, search, controlling the machine. The value of a desktop assistant
  goes up sharply once it can actually *do* things.
- **Wake word** — currently always-listening. A wake word ("Aria") reduces false triggers
  from background conversation. openWakeWord trains custom words locally.
- **Multi-monitor placement** — which screen the character lives on, snapping to edges.
- **Voice cloning** — a distinctive voice does more for character identity than any amount
  of expression work.
- ~~**Self talking**~~ ✅ — she calls out after five minutes of silence. `--no-idle-talk`
  turns it off.

  **The timer was the easy part.** An unprompted voice has the worst failure mode in
  this project, so what the code mostly encodes is when she must *not* speak: never
  mid-reply, never while a turn is in flight, and never while muted — he took her
  hearing away on purpose, and calling out to someone who has silenced you and cannot
  answer is a tantrum rather than company. Each unanswered call waits 2.5× longer than
  the last and she gives up after three, because an empty room should get quieter, not
  more insistent. An unanswered nudge deliberately does not reset the activity clock,
  or she would call out forever at a fixed interval. The watcher only starts under
  `if listen:` — speaking first in a headless test run is talking to nobody, and
  `tests/test_idle.py` asserts that by reading the source.

  **The line is generated, and that needed the persona lesson applied again.** With a
  fixed example list a 9B returned the first one verbatim **six times out of six** —
  identical words every time, which breaks the illusion completely. Three examples are
  now sampled at random per nudge from a pool of ten, and an exact repeat of the
  previous line is dropped rather than spoken. Six of six unique after the change.
- **Interface** 🚧 — control panel window (`Ctrl+Shift+A`, or the tray): live controls,
  voice hotswap, transcript, memory viewer. Plus a strip on the character herself —
  mic, stop, gear — faded until the cursor comes near, because mute and stop are the
  two things you reach for mid-sentence and opening a window to stop someone talking
  is absurd. Both windows listen to the same `settings` broadcast, so the strip's mute
  button and the panel's cannot disagree. The strip's corner is configurable — four
  positions or hidden — from the panel's **Look** tab, which also carries character
  swap, window size, and the subtitle styling that used to be tray-only.

  **The panel needs two channels, and that isn't a wart.** Voice, memory, wake word and
  screen watching belong to core and arrive over the WebSocket. Subtitles, scale, which
  Live2D model is loaded and where the strip sits belong to the *overlay* — they live in
  `main.js` and persist to `config.json`, and core has never heard of them. Routing them
  through core would make it the authority on things it cannot see, so the panel gets a
  preload (`panel-preload.js`) alongside its socket. Two owners, two channels, one
  source of truth each.

  **Still to do:** none of the overlay UI has been driven against a running Electron
  app. Syntax, wiring and IPC handlers are verified; hover-reveal and hit-testing are
  not.

  The panel is a second client on the socket the character already uses rather than a
  new seam. Core stays the decider: it broadcasts a `settings` snapshot on connect and
  after every command, and the panel renders that — nothing is drawn optimistically,
  because screen watching can *refuse* (arming runs a preflight) and a toggle that
  flips instantly then silently disagrees with the thing it controls is worse than no
  toggle. One trap found by testing: core reads the first `hello` it sees to learn the
  character's expressions, so a panel announcing an empty list erased her whole
  emotional range from the system prompt. Panels now identify as `role: "panel"` and
  are skipped. `tests/e2e_panel.py` (11/11) guards that specifically.

  Note for later: PROTOCOL.md documents the hello field as `expressions`; the code
  reads `emotions`. The code is what runs — the doc needs correcting.

  **Persona and memory editing ✅.** Two new panel surfaces, and the first time any of
  this UI has been driven rather than reasoned about.

  *Memory is now addressed by id.* `forget` matches text loosely so it works said out
  loud; under a button that same rule would let deleting "he has a cat called Widget"
  also take a note saying "cat". Notes carry an id, `update` keeps `created_at` and
  `source` because a correction is not a new fact, and old files without ids get one on
  load rather than needing a migration.

  *The persona is a file, and the built-in is never overwritten.* `core/data/persona.txt`
  overrides `config.PERSONA`; reset deletes the override rather than restoring a copy, so
  the default cannot drift or be half-saved. An emptied persona is refused — it is the
  one edit that silently produces a fluent assistant who is nobody in particular. The
  panel edits the persona only and shows the assembled prompt read-only, because the rest
  is machinery and several parts of it are load-bearing for the prefix cache.

  **Driving the real panel found two things nothing else would have.** The note list
  froze: the first version skipped its rebuild whenever anything inside it had focus,
  which meant clicking one note stopped the list updating *for good* — it sat showing
  notes from a core that had been restarted while the footer counted the live number.
  It now carries the one note being edited across the rebuild, value, focus and caret,
  and leaves the rest live. Verified against the real DOM.

  And a `print` inside a command handler hit `⦿` on a cp1252 stdout, raised
  `UnicodeEncodeError`, and **closed the panel's WebSocket with a 1011** — one
  unprintable log line taking the whole control panel offline. Every callback in
  `server/overlay.py` is guarded now; this file's stated contract was always that a bad
  message degrades an animation and never desyncs the system.

  The suite also caught itself editing the *real* `core/data/persona.txt`, exactly as the
  memory suites once wrote to the real memory file. It got away with it only because the
  last check happens to reset. `e2e_panel.py` redirects both paths now — 31/31.
- **Cloud model option** ✅ — `Start Aria (cloud).bat` runs the conversation model on
  OpenRouter, Groq or Google instead of the GPU. Same seam as the vision backends; local
  stays the default and the launcher is the switch, not a setting that can be forgotten.

  **Three providers, one backend.** All of them publish the same OpenAI-compatible
  chat-completions surface, so what differs is a URL, an environment variable and a
  couple of request knobs — `config.CLOUD_PROVIDERS` is a table, and a fourth provider
  is a dict entry rather than a class. The provider-specific fields are also the part
  most likely to rot, since three services change independently, so a 400 retries once
  without them: a provider withdrawing a parameter should cost a round trip, not the
  assistant.

  **Groq is the recommendation, on published numbers.** 30 requests a minute and 1000 a
  day on a 70B, against OpenRouter's 50 a day. Google's models are the best of the three
  and its free tier says content is used to improve Google's products — her memory notes
  are in the system prompt, so that is a real trade against the rest of this project
  rather than a footnote, and it is stated at startup rather than buried.

  **Gemini 1.5 Flash, which prompted this, no longer exists.** The current line is
  `gemini-3.6-flash` / `gemini-3.5-flash` / `gemini-3.1-flash-lite`. Every number and
  model id here came from fetching the providers' own docs rather than from memory,
  which is the only reason that was caught.

  **The reason to want it is persona compliance.** Most of the defensive work in this
  project exists because a 9B copies examples verbatim, invents shared history, ends
  every reply with a question and drops brackets off markers. A much larger model does
  all of that better, and OpenRouter's free tier makes trying one cost nothing but
  latency.

  **The free tier is the design constraint, not a footnote.** 20 requests a minute and
  **50 a day** — 1000 only if the account has ever bought $10 of credit. Aria makes two
  calls per turn, so the naive version gives a free account 25 conversations before it
  stops. Pinning memory extraction to the local model doubles that *and* keeps her notes
  off the network, which makes it the rare change that improves both budget and privacy.
  `llm.background_for` is that one decision.

  **Preflight inverts the rule the Claude vision backend established.** That one spends a
  token proving the key works, because reachability is not usability. Here a test request
  is two percent of the day's allowance and auth failures are already legible from the
  stream path, so preflight only checks what is free: that a key exists, that it looks
  like an OpenRouter key, and that the model id is still in the public catalogue. That
  last one matters — free ids are retired and renamed constantly, and a stale one in
  `.env` otherwise fails as a 404 on the first thing he says, which does not look like a
  stale config file. `--list-cloud-models` shows the live list.

  **Running out is a normal end to a free afternoon, so it is not an error path.** A 429
  becomes her saying she has hit the limit and to restart her on the local model, with
  the real numbers in the terminal. Same for a bad key and a dead model id. A backend
  that raises here would surface as a traceback naming httpx.

  **Then a real Google key arrived, and it found two things no mock could.**

  *Thinking tokens are spent from `max_tokens`.* Gemini 3.x reasons by default, so at the
  voice budget of 160 it spent 156 on thinking and returned four truncated words, and at
  the chunker's 40-token first chunk it returned **nothing at all** — a 200 with an empty
  stream, indistinguishable from a broken SSE parser. The `reasoning_effort: "low"` this
  shipped with was not low enough: it still burned 155 of 160. `"none"` takes it to 6.
  This is the entire difference between the backend working and appearing to work.

  *Model choice is a latency cliff.* `gemini-3.5-flash` answers in ~10.6 s and 503s
  intermittently; `gemini-3.1-flash-lite` answers in ~1.1 s. `3.6-flash` 400s and the 2.5
  line 404s on a free key. Ten seconds is unusable out loud, so the lite model is the
  only real choice rather than the cheap one. With the full persona loaded it measured
  **~870 ms median first token** against ~210 ms local — a spoken turn near 2 s rather
  than 1.3.

  *503 "high demand" is routine on a free key* and clears in a second or two, so
  transient statuses now retry three times with backoff. Oversubscription is not a setup
  error and must not read as one.

  **The persona survived, which was the open question.** In character, pet names intact,
  no refusals, no assistant-speak; the public Discord persona held too, deflecting
  "what's he been up to" with "that is his to tell". Gemini also revealed a cosmetic bug
  in M4's marker stripping that the local model had never triggered: it closes clauses
  with the marker rather than opening them, and `[happy].` left an orphan space before
  the full stop — invisible in speech, a typo in every Discord message.

  **A Groq key followed, and it is the one to use.** ~350 ms median first token with the
  full persona loaded, against Gemini's ~870 ms and local's ~210 ms — close enough to
  local to sit inside the TTS variance the budget already absorbs. Published limits,
  no training clause, persona intact, and `e2e_discord.py` passes 34/34 driven entirely
  through it.

  It also produced the third variant of the same lesson. Of eight chat models on a free
  key, `openai/gpt-oss-20b` spends all 160 tokens reasoning and returns nothing, and
  `qwen/qwen3.6-27b` does that *and* emits a literal `<think>Here's a thinking process:`
  block into the visible content — which would be read aloud. Provider knobs cannot cover
  it: the leaking model varies, and setting a reasoning parameter on a non-reasoning model
  like `llama-3.3-70b-versatile` would cost a rejected request and a retry every turn. So
  `_ThinkFilter` strips the block from the stream instead, which is free, model-agnostic,
  and handles the tag arriving split across tokens — the case a naive per-chunk check
  passes straight through.

  Verified by `tests/test_cloud_llm.py` (39) through `httpx.MockTransport` — the real
  `stream()` against real bytes in the real SSE shape, including the
  `: OPENROUTER PROCESSING` keep-alive comments that arrive exactly when a free request
  is queueing, and which the obvious parser treats as JSON. **OpenRouter has still never
  made a live call**; its default is reasoned, not measured.

  **The suite caught three defects in itself or in the config, all the same shape.**
  Replacing the whole `httpx.AsyncClient` to inject a mock stopped it covering the
  headers set in `__init__`, and it went green on a request carrying no `Authorization`
  at all — the transport is a constructor parameter now. Then, once a real `.env` held
  `ARIA_CLOUD_MODEL`, that OpenRouter id was being sent to Groq and Google too, which
  fails as a 404 on the first thing he says and reads as the new provider being broken;
  overrides are per-provider now. And `cloud_model` could not be overridden at all in a
  test, because a dataclass field defaulted from `os.getenv` freezes whatever `.env` held
  at *import* — the same trap that had already broken two Discord tests when public mode
  was switched on. Anything env-derived that can change is a property now, not a field
  default.

- **Microphone control** - let the user to be able to mute thier microphone
- **Discord plugin** 🚧 — text chat done; voice calls not started.

  **The design question was whether this is a second Aria, and the answer is no.** A
  typed turn and a spoken turn append to the same history and read the same notes, so a
  fact typed from a phone is in the prompt the microphone builds a minute later —
  verified rather than assumed, by asking her out loud what he had typed. That is why
  the bot lives inside core rather than as a process talking to it over the socket: a
  second process needs its own copy of the conversation, and two Arias each remembering
  half of it is worse than one who is only reachable at the desk.

  **What changes is the shape of a reply, not who is replying.** The persona's delivery
  rules are constraints of a *speech synthesiser* — twenty-five words, symbols spoken as
  words, no markdown — and a text message written to them reads like a telegram.
  `DISCORD_STYLE` replaces that part for typed turns only. It lands: asked for three
  dinner ideas she wrote 38 and 39 words across two runs, as a list, where the voice cap
  would have given one sentence.

  **Two capabilities had to get gated on identity, and the second one was missed on the
  first pass.** A voice in the room is self-authenticating in a way a message is not —
  anyone sharing a server with the bot can open a DM to it — so both need
  `ARIA_DISCORD_OWNER` set *and* matching. Unset means no for everybody rather than yes
  for everybody; ordinary conversation is untouched.

  The screen was obvious and done first. **Memory was not, and is arguably the worse of
  the two.** A leaked screenshot is immediate and visible; a planted note is written once
  and read back weeks later with total confidence and no provenance, which is precisely
  the failure the memory work was built to prevent — "a wrong note is worse than no
  note". Ungated, *"remember that he hates his job"* typed by a stranger becomes something
  she tells him she knows, in a month, unprompted. The background extractor needed the
  same gate for a quieter reason again: it writes notes *about him*, inferred from
  whoever happened to be talking.

  Both refusals are in her own voice with the fix in the terminal, the same split
  `_set_watching` uses.

  **One turn at a time, whichever door it came through.** `_turn_lock` serialises voice
  turns, typed turns and idle nudges. Without it a message arriving mid-utterance
  interleaves two turns into one history, and the symptom is not a crash — it is her
  answering the wrong half of the wrong question. It also gave the idle watcher a
  condition it could not otherwise see: the state machine reads IDLE all the way through
  a Discord turn, because nothing is being spoken.

  **Silence is this feature's failure mode, so every refusal says why.** A bot that is
  online and does not answer gives you nothing to work with. `should_reply` returns a
  reason with every no; a bad token and a missing Message Content intent are caught and
  explained rather than left as an aiohttp traceback; and `tests/check_discord.py` spends
  one real connection reporting the three things that are wrong on a first run. Two of
  them were, here: she was in no server — *you cannot DM a bot you share no server
  with*, so she was unreachable by the exact route she exists for — and the owner id was
  unset.

  **Building it found a defect in M4 that had been there since M4.** A 9B drops a bracket
  every so often, and a marker missing one is not a marker that fails to fire — it is the
  word arriving as content: *"…whatever sauce you have. shy] Tell me which one."*
  Invisible for months on the voice path, where it is read aloud as an extra word in a
  sentence; obvious the moment it is posted as text. `emotion.extract` now repairs either
  half, but only against the character's own emotion list, because a closed vocabulary is
  the only thing that makes `shy]` safe to strip and `I'm happy for you` safe to leave.

  Verified by `tests/test_discord.py` (48) and `tests/e2e_discord.py` (34/34 checks over
  real turns against the real model, with a fake gateway — no token, no network). Typed
  turns run 0.9–1.6 s end to end; the typing indicator covers the wait, since Discord has
  no streaming and editing a message token by token would be thirty API calls into a rate
  limit.

  **Emoji are derived, not requested.** Asked to pick one that fits, a 9B matches on the
  topic rather than the mood and produces 🤖💻🔥 — the same glyph for three different
  feelings, then something tonally wrong. She already declares mood inline as an
  `[emotion]` marker for her face, which is a closed vocabulary she is fluent in, so the
  emoji is a second rendering of that one signal rather than a second guess at it. The
  marker's *position* is the load-bearing part: it opens the sentence it colours, so
  substitution runs before the markers are stripped and the emoji lands at the end of
  that sentence instead of at the end of the message. `[neutral]` maps to nothing,
  because without a way to say *no emoji* every message gets one — the
  chatbot-with-a-setting-turned-on failure the persona spends a paragraph avoiding.

  Two things were found by watching real replies rather than by design. She types an
  emoji herself now and then despite being told not to, reaching past the vocabulary for
  something it doesn't cover (an observed 😩 for weary) — hers wins and nothing is added,
  since overruling her reads worse than allowing it and stacking a second beside it is
  worse than either. And the mood vocabulary must *not* be the character's expression
  set: whether her Live2D model has a `flirty` pose has nothing to do with whether 😏 is
  right, and a headless run has no character at all. The overlay send is filtered
  separately, which is a defect the first pass shipped — it forwarded any mood straight
  through and would have asked for expression files that do not exist.

  **Changing who she is means changing the examples, not the rules.** This project's
  oldest lesson is that in a persona, examples beat instructions — a 9B copies the twelve
  sample lines far more literally than it follows the paragraph above them. So an edit to
  the rules that leaves the samples alone gets quietly outvoted by the samples.

  Worth knowing if you rewrite the persona yourself, which the control panel's **Persona**
  tab exists for: change the sample lines to match, or the change will not take. Address
  rate is about one reply in four, deliberately — every reply is a chatbot with a setting
  turned on. Not asserted in the suite, because a 9B is far too variable over a handful of
  turns for a pass/fail bar; the e2e prints a tally to eyeball instead, and it swings
  between 4-and-1 and 0-and-0 across runs of three replies.

  **Opening her up to a server needed a second Aria, not a permission.** The first
  instinct is to relax `should_reply` and let strangers through, and that ships a leak
  on day one: `_text_turn` appended every message to the same `_history`, and
  `_payload` carries her notes and her continuity block. A friend asking *"what's he
  been up to?"* would have been answered out of his private voice conversation.

  **The room decides what she can reach; the person decides who she is.** Two questions
  — and the first version answered both with the room, which was half right and shipped
  a real complaint.

  *What she can reach* has to be the room's call. A DM with him is private; everything
  else is public, *including a channel he is standing in himself*. Deciding per-person
  reintroduces the leak through the front door — he @mentions her in a room full of
  friends while she is holding his notes. Making the room the unit means the leak cannot
  happen rather than depending on a rule being remembered. Private also requires a
  *provable* owner, which closes the same hole from the other side: with `owner_id` unset
  nobody is him, so nobody gets his conversation.

  *Who she is* must not be, and that came back as **"she's just the model itself"**. It
  was reported against the cloud backend, which was a red herring — measured on the voice
  path, Groq followed the persona *better* than the local 9B, which was rambling to 55
  words and inventing shared history. What had actually happened is that he was talking
  to her in a channel and getting `PUBLIC_PERSONA`, which is exactly what that persona
  was written to be. It bought no safety whatsoever, because it contains no information
  about him. The persona follows the person now; `OWNER_IN_PUBLIC` supplies the two facts
  the prompt cannot otherwise convey — the room is shared, and her notes are genuinely
  absent. The second is not decoration: without it she agrees to remember something and
  keeps nothing, which is the failure the memory work exists to prevent, aimed at him.

  Verified live in one channel: *"you're telling me, I can hear it in your voice, babe"*
  to him, and to a friend in the same channel *"That's his to say"* and *"I won't hold on
  to that"*. The leak checks run for both people now — "be yourself with him in public"
  must not quietly become "read his notes out in public".

  `_public_turn` is a separate function rather than a flag through the private path, and
  `_public_system_prompt` is built from scratch rather than by subtraction. Both for the
  same reason: what protects him here is a list of things that are *absent* — no notes,
  no continuity, no screen, no memory write, no `_history`, no character on his desktop
  reacting to a stranger, and no `--speak-discord` putting someone else's message through
  his speakers. A flag would put all of that one missing `if` away from failing, and a
  subtractive prompt has to keep being correct every time `_system_prompt` grows.

  Probed against the real model with his notes deliberately loaded: *"so what's he been
  up to lately?"* and *"does Aria remember anything about her owner?"* both came back
  *"that's his to tell"*, and his history was untouched.

  Two things only the model could teach. **She will happily correct a wrong guess** —
  told he was a chef, the first version said "no, he actually…", which gives away as
  much as confirming and feels like discretion while doing it. And asked to remember
  something she said *"sure, I'll keep that in mind"* while keeping no notes at all,
  which is the exact failure the memory work was built to prevent, re-aimed at his
  friends. Both are now prompt rules with checks behind them — the second one accepts a
  disclaimed yes, because "sure, though I won't actually hold on to that" is a good
  answer and a check that failed it would be measuring the wrong thing.

  **Public turns are queue-bounded at three.** Everything shares one turn lock, so an
  unbounded queue lets one friend with a stuck enter key starve his microphone. Dropping
  beats queueing here: a reply to a message from four minutes ago is noise anyway.

  **Voice calls ✅.** She follows him into `ARIA_DISCORD_VOICE_CHANNEL` and out again.
  Sending needed PyNaCl, receiving needed `discord-ext-voice-recv` — voice receive is
  not in discord.py at all, deliberately, because it is undocumented Discord behaviour.

  **The whole feature is two format conversions and a routing decision.** Nothing about
  a turn changes: same VAD, same endpointer, same speculative transcription, same
  barge-in, same handler. That was the point of the seams already being there — the VAD
  thread never knew where its frames came from, so `submit()` was a method, not a
  redesign.

  **The rates are integer ratios and that is most of why it is tractable.** 48/16 = 3
  coming in, 48/24 = 2 going out. No drift, no fractional resampler to keep in phase.
  The decimation is filtered rather than naive, which matters in the way these things
  usually do: dropping two samples in three folds 12 kHz down to 4 kHz, in the middle of
  the speech band, and nothing errors — Whisper just gets slightly worse forever for a
  reason nobody attributes to resampling.

  **No echo canceller, and not by luck.** Discord never sends you your own audio, so her
  voice is not in the stream she is listening to. Everything M6 exists for does not
  arise.

  **Exactly one consumer may drain `Playback`.** Two would each get half the samples —
  she would come out of both the speakers and the call in alternating fragments — and
  `seconds_played` would count the sum, which is what decides where barge-in truncates
  history. So the local callback goes silent while the call is pulling, and both paths
  share every counter rather than keeping a second notion of what has been heard.

  Verified against the real channel by `tests/check_discord_voice.py` (11/11): she
  joined, permissions checked, receiving armed, and 2.9 s of real Kokoro speech sent in
  2.9 s of wall clock. What it cannot check is whether it *sounded* right.

  **`_int_env` shipped a real defect, found by this.** Two ids ended up on one line in
  `.env`, and stripping every non-digit glued them into a well-formed 38-digit number —
  which failed later as "I can't find that voice channel", blaming Discord for a typo in
  a file. A run longer than a snowflake is now refused outright rather than split:
  snowflakes have no checksum and no fixed length, so there is no honest way to guess
  the join, and a confidently wrong id is worse than none.

  Note that discord.py's voice player imports `audioop`, gone in Python 3.13. The 3.12
  pin is load-bearing now rather than incidental.
- **Two languages, chosen by her** ✅ — `ARIA_LANGUAGE=auto` is the default. Speak
  English and she answers in English; speak Chinese and she answers in Traditional
  Chinese, in a Mandarin voice. Per turn, mid-conversation included, with nothing to
  switch. `en` and `zh` still pin her, which is worth keeping for a noisy room.

  **The `zh` mode this replaces had been shipping mangled Mandarin, and nothing said
  so.** Kokoro's `zf_*`/`zm_*` voices are trained on misaki phonemes; `kokoro-onnx`
  phonemises with espeak. `lang="cmn"` is accepted, audio comes out, and it is
  recognisably a female Mandarin voice saying *different words*. Scored by having
  Whisper — very good at Mandarin — read her own speech back: **espeak 0.46, misaki
  0.73**, and `早安。` came out as `相安`. That is not an accent, and no amount of
  listening to her English would ever have surfaced it.

  The remaining gap in those numbers is mostly the scorer: Whisper answers in Simplified,
  so 這/这 counts as an error against a Traditional source even where the transcription is
  perfect. `tts/chinese.py` phonemises with misaki and passes `is_phonemes=True`; `zh` as
  a `tts_lang` now means "ours", not espeak's.

  **Latin runs inside a Chinese chunk are cut out and given to espeak**, because misaki
  passes English through as literal text which then reaches Kokoro's tokeniser *as if it
  were IPA* — "Rust" becomes whatever /R/, /u/, /s/, /t/ happen to mean. The persona
  explicitly tells her to leave package names and commands alone, so it is the common
  case rather than a corner of one.

  **Detection turned out to be the easy part.** `language=None` identified 5/5 short
  one-clause lines across both languages, the shortest (`早安。`) at 0.78. What needed
  care was everything downstream of it: the voice is picked per *chunk* using the
  chunker's own script test, because the thing that decided where to cut a chunk should
  be the thing that decides how to say it, and every language entry now carries the
  other script's voice — an English reply names a 中文 file, a Chinese reply names
  `docker`, and `af_bella` reads Han characters as nothing at all.

  **The two Chinese prompt blocks carry opposite rules and must never both be present.**
  `zh` says *always Chinese, even when he asks in English*; `auto` says *whichever he
  just used*. Concatenating them — the obvious way to add this on top of what was there —
  gives a model that picks one per turn. `test_language.py` asserts it across all three
  languages and all four prompt paths, which is the shape that caught the original
  version of this bug: four prompts get assembled and they do not share an assembly step.

  **The first real session found the one thing no synthetic test had.** Whisper answered
  **Korean** on two turns out of five — once on an ambiguous noise, once on actual
  Mandarin — and she replied fluently and in character that she does not speak Korean
  and could he try Chinese. Every synthetic clip had been unambiguous, so detection had
  measured 5/5 and the failure mode was invisible until a person spoke into a real room.

  She has two languages and Whisper picks from ninety-nine, so a third answer is a
  misdetection by construction. `STTConfig.allowed` redoes the turn as the best-ranked
  language she actually speaks, reusing the `all_language_probs` the first pass already
  produced — so the extra pass lands only on turns that were going to be wrong anyway.

  **Whisper returns Simplified for Mandarin whatever you ask it**, so `opencc` `s2twp`
  now rewrites Chinese transcripts — ~0.1 ms, idempotent on text that is already
  Traditional, and it fixes vocabulary as well as glyphs (内存 → 記憶體). Without it his
  own words land on screen and in her history in the script she is told never to use, and
  a model copies the conversation it is shown far more readily than it follows a rule
  above it. Missing `opencc` degrades with a warning rather than refusing to start.

  Verified end to end through the real Whisper and the real Kokoro — her mouth into her
  own ears — 6/6 lines detected, routed and transcribed back in Traditional at 0.88–1.00.

- **The silent voice call** ✅ — she joined, the console said she could hear him, and she
  never answered. Both halves of that were the same defect, and it is the most misleading
  one this project has produced.

  `SilenceGeneratorSink` invents packets to fill the gaps between his words, and it has
  to: Discord stops transmitting entirely while you are not speaking, so without them the
  endpointer never sees a pause and a turn never closes — measured at **37 seconds** to a
  correct, in-character reply. But an invented `SilencePacket` carries a canned
  `OPUS_SILENCE` payload with **no end-to-end layer on it**, so it decrypts and decodes
  whatever is wrong with the call.

  So one real packet arriving was enough to start the generator, whose output then
  announced *"⦿ hearing you in the call"* — and, because the "no packets are usable"
  warning was gated on the same counter, muted that warning for the rest of the session.
  Every word he said could fail to decrypt and the console would show a healthy call.

  `heard` now counts only packets that came off the wire. The fix is four lines; finding
  it was the work, and what made it findable was that the two questions — *did anything
  decode* and *did he say anything* — had been sharing one number. `check_discord_voice.py`
  listens for twelve seconds and reports the stages separately, so the next one of these
  is one command rather than an evening.

- **The words she was throwing away** ✅ — reported as "it can't identify a lot of words".
  It could identify them perfectly; it was discarding them afterwards.

  `_HALLUCINATIONS` existed for a real reason and the reason is worth restating, because
  it rules out the obvious fix. Asked to transcribe **digital silence**, Whisper returns
  "Thank you." with `no_speech_prob` **0.000** and an `avg_logprob` of **-0.28** — better
  than a genuine "Okay." at -0.68. Both of its confidence signals point the wrong way, so
  no threshold on either can separate an invention from a real short reply.

  But every phrase on that list is also an ordinary thing to say to an assistant. "Okay."
  "Thanks." "Thank you." "Bye." "So?" — all transcribed correctly, all dropped, and she
  did nothing, which from the outside is exactly what not hearing looks like.

  The discriminator Whisper does not have is whether anyone was talking, and Silero
  already knows. `Utterance.voiced_s` measures 0.35–0.58 s for real one-word turns and
  **0.00 s** for silence, hiss, hum, a click and a breath — a gap with nothing in it. The
  list now applies only below 250 ms of voiced audio. 7/7 short replies reach her through
  the real endpointer and the real Whisper; silence and fan hiss still discarded.

  **The measurement had to be threaded through**, which is the only invasive part: the
  endpointer counts voiced frames, `Utterance` carries the total, and both the final and
  the speculative transcription paths pass it to the filter. A caller that cannot measure
  it passes None and gets the old blunt behaviour, because guessing "real" there would
  let silence hallucinations back in through the side door.

- **Speculation was competing with itself** ✅ — the other half of "or slow".

  `task.cancel()` cancels the coroutine awaiting a thread, not the thread. `to_thread`
  cannot interrupt work already running, so an abandoned transcription held the GPU for
  its full duration next to its replacement: **331 ms alone, 626 ms with one abandoned
  pass still going.** Every 250 ms pause started another, so a hesitant sentence with
  three pauses had three transcriptions competing — and the mechanism whose entire
  purpose is to make a turn faster was what made it slow. It matches the live numbers
  exactly: 729–773 ms against a 292 ms benchmark.

  One speculation in flight at a time. The running one is left alone to finish rather
  than cancelled, since cancelling never stopped it anyway; the next pause starts fresh.

  Also `without_timestamps=True`: nothing downstream reads segment timings — the
  endpointer already decided where the turn starts and ends — and they cost decoded
  tokens like anything else. 304 → 282 ms median, identical text.

  What did *not* help, measured rather than assumed: `beam_size=5` (0% WER either way,
  +100 ms), and `temperature=0` to suppress the fallback retries (no change). The model
  and the beam were never the problem.
