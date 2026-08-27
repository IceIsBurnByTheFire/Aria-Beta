# Architecture

## The shape of it

Two processes, one WebSocket.

```
┌─────────────────────── core (Python) ───────────────────────┐
│                                                              │
│  mic ──► VAD ──► STT ──► ┌─────┐ ──► TTS ──► speaker        │
│           │              │ LLM │      │                      │
│           │              └─────┘      │                      │
│           │              ▲    ▲       │                      │
│           │   screen capture  │       │                      │
│           │                   │       │                      │
│           └── barge-in: cancel┼───────┘                      │
│                               │                              │
│              Discord ─────────┘  text in, text out           │
│                                                              │
└──────────────────────────┬───────────────────────────────────┘
                           │  WebSocket (localhost)
                           │  state + expression + viseme events
┌──────────────────────────▼───────────────────────────────────┐
│              overlay (Electron) — transparent window          │
│              renders + animates the character                 │
└───────────────────────────────────────────────────────────────┘
```

Core owns all timing and state. The overlay is a **dumb renderer** — it receives events
and plays them. It never decides anything. That keeps the interesting logic in one
language and makes the character swappable.

## Why two processes

Python has the ML ecosystem (Whisper, Silero, torch, every TTS worth using). Electron has
transparent click-through always-on-top windows and a real GPU canvas. Trying to do
character rendering from Python means fighting Qt/OpenGL for a worse result; trying to do
Whisper from Node means ONNX gymnastics and no CUDA story worth having.

Cost of the split: one IPC hop, ~1ms locally. Worth it.

## The event loop

The whole system is a state machine over four states:

```
IDLE ──user speaks──► LISTENING ──silence──► THINKING ──first token──► SPEAKING
 ▲                         │                                              │
 │                         │                                              │
 └─────────────────────────┴──────────── barge-in / done ─────────────────┘
```

Every state transition is broadcast to the overlay so the character can react — perk up
when listening, tilt its head when thinking, animate when speaking.

## Components

### Audio capture and VAD

Continuous mic capture at 16 kHz mono, fed into **Silero VAD** frame by frame. Silero is
small, runs on CPU in well under real time, and is far more robust than energy-threshold
VAD against keyboard clatter and fan noise.

VAD does two jobs:

1. **Endpointing** — decide when the user finished a thought, so STT can run. Needs a
   trailing-silence window (~500-700ms). Too short and it cuts people off mid-sentence;
   too long and the assistant feels sluggish.
2. **Barge-in detection** — see below. This is the harder job.

### STT

**faster-whisper** (CTranslate2 backend) with `large-v3-turbo` on the GPU. On a 5080 this
transcribes a few seconds of speech in a couple hundred milliseconds, which is well inside
the budget.

Run it on the segment VAD hands over, not on a rolling buffer. Streaming partial
transcripts are a nice-to-have for showing live captions, not needed for the core loop.

**Transcription is speculative.** The endpoint hold is ~600 ms of silence during which
the GPU is idle and the user is already waiting, and STT fits inside it almost exactly.
So transcription starts 250 ms into a pause, before the turn is known to be over. If the
pause turns out to be mid-sentence, the work is discarded — `Utterance.frames` records
how much audio the committed turn covers, and a speculative result is adopted only if it
covered at least that much.

Measured effect: STT drops from 292 ms of perceived latency to zero. This is the single
largest win available in the loop, and it costs nothing but wasted GPU work on pauses.

### LLM

Pluggable behind one interface, because this is the piece most likely to change:

```python
async def stream(messages, images=None) -> AsyncIterator[str]
```

Two backends, routed by task:

- **Local (Ollama)** — drives conversation. Free, private, offline, no rate limits. Already
  installed here with a 9B model, which handles chat fine.
- **Cloud (Claude API)** — drives the screen-reading path, where the quality gap over a
  local model is largest and the call volume is lowest.

The persona lives in the system prompt, not in the model choice. Swapping backends should
not change who the character is.

VRAM budget: the local model shares 16 GB with Whisper and Kokoro. If it gets tight, shrink
Whisper first — a `distil` variant frees real headroom and `large-v3-turbo` is already
faster than the loop needs.

### TTS — the latency-critical piece

This is where perceived responsiveness is won or lost. Two rules:

**Chunk by sentence.** Do not wait for the full LLM response. As tokens stream in, split
on sentence boundaries and start synthesizing the first sentence while the rest is still
generating. First audio out should land ~300-500ms after the first token.

**Stream the audio out.** Synthesize into a queue the playback thread drains, so a long
reply starts playing immediately rather than after full synthesis.

Engine is **Kokoro-82M** — small, fast, good quality for its size. Same pluggable-interface
treatment as the LLM, because voice is the component most likely to get swapped once you
have actually listened to it for a while.

### Barge-in — the genuinely hard part

Everything else here is assembly. This is the part that needs real engineering.

The problem: the mic hears the assistant's own voice through the speakers. Naive VAD sees
that as the user talking and cancels the reply the instant it starts — the assistant
interrupts itself into permanent silence.

Three approaches, in increasing order of difficulty:

1. **Headphones.** No echo path exists. Works perfectly, zero code, and is what most
   people building this actually ship first. Good enough to validate the whole loop.
2. **Half-duplex gating.** Suppress VAD while playing audio, with a raised threshold so
   only loud speech breaks through. Cheap, but interruption feels unreliable — you have
   to half-shout.
3. **Acoustic echo cancellation.** Feed the playback signal as the reference into WebRTC's
   audio processing module, which subtracts it from the mic input. This is the real
   answer for speaker use. It needs the playback and capture streams sample-aligned,
   which is the fiddly part on Windows.

Built on (1); (3) is M6. (2) was deliberately skipped — it produces a bad experience and
teaches you nothing you need for the real fix. `--no-barge-in` covers speaker users in
the meantime by disabling the trigger entirely, which is honest about the limitation
rather than pretending to work.

**Cancellation order is load-bearing.** Playback is flushed *first*, freezing
`seconds_played` before anything else unwinds, then the response task is cancelled —
which tears down TTS synthesis and the LLM stream with it. Flushing after cancelling
would race the audio callback and make the played-sample count unreliable, which
corrupts the history truncation below.

**History records what was heard, not what was generated.** This is the subtle half of
barge-in. When the cut lands, chunks are already sitting in the playback queue that will
never reach the ear. `spoken.spoken_prefix` maps `seconds_played` back to text,
truncating a straddling chunk proportionally by word count. Skip this and Aria believes
she answered a question the user never heard her answer — and since nothing errors, the
conversation just quietly stops making sense two or three turns later.

Measured: playback stops 5 ms after the trigger.

A **grace window** (300 ms) suppresses the voice trigger immediately after playback
starts, so a trailing breath or the tail of the user's own previous utterance does not
kill the reply before it begins. The manual trigger has no grace — a deliberate press
means stop now.

Cancellation itself must be genuinely immediate. When barge-in fires: stop playback, flush
the audio queue, cancel the TTS task, cancel the LLM stream, and record what was *actually
spoken* into history rather than what was generated. That last detail matters — if the
assistant thinks it said three sentences it never got to say, the conversation drifts.

### Vision / screen capture

Capture with `mss` (fast, cross-platform, grabs a monitor or region).

**Do not stream frames continuously.** It is expensive on cloud models and mostly useless —
the screen is static 95% of the time. Instead:

- **On demand** — the user asks something screen-referential ("what's this error?").
- **Throttled + change-gated** — sample at a low rate, and only send when the frame differs
  meaningfully from the last one sent.

Downscale before sending; a 4K screenshot is a lot of tokens for no gain. Long edge around
1500px is plenty for reading UI text.

Privacy: screen capture is a genuinely invasive capability. It needs a clear on/off state
the user can see from the overlay, and it should default to off.

### Character control

The character is a **Live2D** model rendered with `pixi-live2d-display` on PixiJS. Cubism
exposes the model as a set of named parameters, and control means writing to those
parameters each frame. The ones that matter:

| Parameter | Drives |
|-----------|--------|
| `ParamMouthOpenY` | Lip sync |
| `ParamAngleX/Y/Z` | Head turn and tilt |
| `ParamEyeBallX/Y` | Gaze direction |
| `ParamEyeLOpen`, `ParamEyeROpen` | Blinking |
| `ParamBodyAngleX` | Body sway |
| `ParamBreath` | Idle breathing |

Named expressions (`.exp3.json`) and motions (`.motion3.json`) ship with the model and are
triggered by name rather than by parameter writes.

Three independent channels drive all this:

**Expression** — the LLM emits inline markers like `[happy]` in its output stream. Core
strips them before the text reaches TTS and forwards them to the overlay, which maps them
onto the model's expression names. Inline markers beat a structured side-channel because
they arrive *in time order with the speech*, so the expression lands on the right sentence.

The mapping matters: the LLM's emotional vocabulary and the model's expression files will
not match. Keep an explicit map in the overlay and send the model's real expression list to
core in `hello`, so the system prompt can constrain what the LLM asks for.

**Lip sync** — derived from the TTS audio, not from text. Compute RMS amplitude per frame
during playback, smooth it, and write it to `ParamMouthOpenY`. Cheap and surprisingly
convincing. Phoneme-accurate visemes are an upgrade path, not a v1 requirement.

**Idle motion** — the overlay handles this locally: breathing, blinking, small head sway,
gaze tracking toward the cursor. It needs no input from core and must keep running
regardless of connection state, so the character never looks dead.

Layering is the thing to get right. Idle motion, expression, and lip sync all write to
overlapping parameters, and whichever runs last wins unless you blend deliberately. Decide
the precedence order once, up front: lip sync owns the mouth, expressions own the eyes and
brows, idle owns everything else and yields on conflict.

### The overlay window

Electron with `transparent: true`, `frame: false`, `alwaysOnTop: true`.

The important trick is **click-through**: `setIgnoreMouseEvents(true, { forward: true })`
makes the transparent region pass clicks to whatever is underneath, while still delivering
mouseover events so you can detect when the cursor is over the character and re-enable
interaction. Without this the character blocks a rectangle of your desktop and becomes
intolerable within a day.

### Discord — a second door into the same loop

`discord_bot.py` is a gateway client and nothing more. It decides whose messages become
turns, hands the loop an `Incoming` of plain values plus a `Channel` to answer on, and
that is the entire surface: no discord.py type ever reaches `loop.py`. The reason is
testability. A turn driven by a fake channel exercises the real prompt assembly, the real
memory writes and the real screen gate, with no token and no network — `e2e_discord.py`
runs offline in seconds and is the only way any of this is verified without a human
sitting in a chat window.

**One history, one memory, one Aria.** A typed turn and a spoken turn append to the same
`_history` and read the same notes, so a fact typed from a phone is in the prompt the
microphone builds a minute later. That is the whole reason Discord lives inside core
rather than as a process talking to it over the socket — a second process would need its
own copy of the conversation, and two Arias who each remember half of it is worse than
one who is only reachable at the desk.

**What differs is the shape of a reply, not who is replying.** `DISCORD_STYLE` is
appended to the system prompt for typed turns only, and it exists because the persona's
delivery rules are constraints of a *speech synthesiser* — twenty-five words, symbols
spoken as words, no markdown. Applied to a text message those produce something visibly
odd. The block sits at the very end of the prompt with the other volatile pieces: he
moves between the desk and his phone mid-conversation, and anything above the switch
point stays in Ollama's prefix cache across it.

**Emoji are derived from the emotion markers, not requested from the model.** She already
declares mood inline for her face; `DISCORD_MOOD_EMOJI` maps that closed vocabulary onto
glyphs, so text and face are two renderings of one signal rather than two guesses at it.
Substitution runs *before* the markers are stripped, because a marker's position is what
says which sentence the feeling belongs to — extract first and the only options left are
"end of message" or a guess. The mood vocabulary is deliberately wider than the loaded
character's expression set and independent of it: `flirty` needs an emoji whether or not
the Live2D model has a pose for it, and a headless run has no model at all. What reaches
the overlay is filtered back down to what the character can actually show.

**The room decides what she can reach; the person decides who she is.** Two questions,
and the first version answered both with the room.

*What she can reach* has to be the room's call. A DM with him is private; everything else
is public, **including a channel he is standing in himself**. Deciding that per-person
reintroduces the leak through the front door — he @mentions her in a room full of friends
while she is holding his notes, and the first "what's he been up to?" has an honest answer
sitting in her context. Making the room the unit means the leak cannot happen rather than
depending on a rule being remembered. Private also requires a *provable* owner, which
closes the same hole from the other side: with `owner_id` unset nobody is him, so nobody
gets his conversation.

*Who she is* must not be. Handing him the guest persona in his own server made her read as
"just the model itself" — which is exactly what `PUBLIC_PERSONA` was written to be — and
bought nothing, because that persona contains no information about him at all. The persona
follows the person now, and `OWNER_IN_PUBLIC` tells her the two things the prompt cannot
otherwise convey: that the room is shared, and that her notes are genuinely absent. The
second matters more than it looks — without it she agrees to remember something and keeps
nothing, which is the failure the memory design exists to prevent, aimed at him this time.

**One turn at a time, whichever door it came through.** `_turn_lock` serialises voice
turns, typed turns and idle nudges. Without it a message arriving mid-utterance
interleaves two turns into one history and two writers into one state machine, and the
symptom is not a crash — it is her answering the wrong half of the wrong question. It
also gives the idle watcher a condition it could not otherwise see: the state machine
reads IDLE during a Discord turn, because nothing is being spoken.

**Two capabilities are gated on provable identity, and it is a lower bar than it looks.**
A voice in the room is self-authenticating; a Discord message is a string from an account,
and anyone who shares a server with the bot can open a DM to it. So `_is_owner` requires
`owner_id` to be set *and* to match — unset means no for everybody rather than yes for
everybody. Ordinary conversation is untouched.

- **The screen**, because M5's whole design rests on capture being armed by the person
  whose desktop it is. The damage is immediate and obvious.
- **Memory** — dictated notes *and* the background extractor — for a quieter reason that
  is arguably worse. A note is written once and read back weeks later with total
  confidence and no provenance; `memory.py` calls a wrong note worse than no note for
  exactly that reason. Ungated, "remember that he hates his job" from a stranger becomes
  something she tells him she knows, in a month, unprompted. The extractor needs the same
  gate because it writes notes *about him*, inferred from whoever was talking.

Both refusals are in her own voice and send him to the terminal for the fix, the same
split `_set_watching` uses — a reply is not the place for an environment variable name.

The bot is never fatal. A bad token or a missing intent prints what to do and the voice
loop in front of you carries on; that is the product, and Discord is a convenience
attached to it.

## Known environment gotchas

All of these were hit for real while building M1, and every one of them fails silently
or with a misleading error:

- **Half of Aria running looks exactly like all of Aria running.** The overlay is built
  to survive core disconnecting — it keeps breathing, blinking and reconnecting — which
  is correct, and means an orphaned character is indistinguishable at a glance from a
  working one. She is on screen, so she looks fine; she has no mic, no model and no
  Discord, so she answers nothing. It reads as "Aria is broken" when it is "Aria is not
  running", and those have completely different fixes. Hit for real on 2026-08-05 after
  a reboot. `Start Aria.bat` now enrols itself in a Windows job object with
  `KILL_ON_JOB_CLOSE`, so both halves die together however the launcher dies, and sweeps
  up anything a previous run left behind. **Which half orphans depends on how it died**
  — closing the window with the X takes the console down with core and spares the hidden
  overlay; a hard kill does the opposite, because terminating a process does not
  terminate its children. Fixing only the visible direction creates the other one; that
  was found by testing, not by thinking about it.
- **`setPosition` grows a frameless window on a fractionally-scaled display.** Measured
  on a 150% monitor: 400 calls to `win.setPosition` took a 400x600 window to 800x1000 —
  exactly one pixel per call, in both axes. Each move round-trips the window's size
  DIP → physical → DIP and the rounding only ever goes one way. One drag is hundreds of
  mousemove events, so the character visibly grew while being dragged, and `layout()`
  faithfully refit the model to the larger window. Drag with `setBounds` and the size
  recomputed from the intended scale, never `setPosition`: measured drift then is zero,
  because the size stops being a value that carries between calls. Reported as "her
  model will size up when just dragging her around", and it never survived a restart,
  which makes it easy to dismiss as imagination.
- **Never make the character wait on something it does not need.** The launcher waits
  for Ollama before running core, and the first version did that *before* starting the
  overlay — so with Ollama down you got 40 seconds of console and an empty desktop. That
  reads as a broken launcher and gets closed long before it would have worked. The
  overlay is deliberately independent of core: it breathes, blinks and reconnects on a
  backoff with nothing behind it. So it goes up first and the wait happens behind a
  character that is already on screen. Reported as "no overlay shows", which is exactly
  what it looked like.
- **Launching at login turns two harmless races into real ones.** Ollama is a Startup
  item too, and two of those begin at the same instant — Aria loses about half the time
  and boots with a yellow "cannot reach Ollama" block, mute until you repeat yourself.
  And once she starts by herself, double-clicking the launcher out of habit is a *second*
  core, which cannot bind 8765 and dies with an OSError out of `OverlayServer.start()`
  that reads as a crash rather than as "she is already open". The launcher now waits for
  the backend (bounded, non-fatal — core's own message is better than ours) and refuses
  to start a second instance. The refusal must return **before** the orphan sweep, or
  the second launch cheerfully kills the running Aria's overlay on its way to telling
  you it did nothing.
- **PowerShell 5.1 reads a BOM-less file as cp1252, and an em-dash becomes a quote.**
  `—` is `E2 80 94` in UTF-8, and byte `0x94` in cp1252 is `"` — a smart closing quote,
  which PowerShell accepts as a real string delimiter. An em-dash inside `"..."` ends
  the string early, the rest of the line parses as code, and the script dies with
  "missing closing '}'" twenty lines further down, pointing at something innocent.
  Harmless in comments, fatal in strings. Keep `.ps1` string literals ASCII, or give the
  file a BOM.

- **Silero v5's ONNX graph needs 64 samples of context prepended to each 512-sample
  frame.** Feed it a bare frame and it accepts the input, runs, and returns ~0.0 for
  everything including obvious speech. The VAD simply never fires — no exception, no
  warning. Guarded by `core/tests/test_vad.py`.
- **Reasoning models need `think=False`.** Qwen3.5 routes its entire output to a
  separate `thinking` field and the spoken `content` never arrives inside any usable
  token budget. The symptom is a stream that yields nothing at all.
- **int8 compute types fail on Blackwell** with `CUBLAS_STATUS_NOT_SUPPORTED`. float16
  works. This shows up only at kernel launch, well after the model loads happily.
- **The first CUDA call costs ~16 s** of PTX JIT for sm_120, then ~0.5 s once the driver
  cache is warm. Warm up at startup or the first turn looks catastrophically broken.
- **Whisper hallucinates confidently on near-silence** — "Thank you.", "you", "Thanks
  for watching!" — so probability thresholds do not catch it. Needs an explicit
  denylist.

- **RTX 5080 is Blackwell (sm_120).** It needs CUDA 12.8+ builds. Older PyTorch wheels will
  install fine, detect the GPU, and then fail at kernel launch with no useful message.
  Install torch from the cu128 (or newer) index.
- **`python` on this machine is 3.14.** The ML stack's wheel coverage lags new releases —
  build `core/` against the 3.12 interpreter already installed (`uv` will pin it), not the
  default one.
- **Live2D's Cubism Core runtime cannot be committed.** `pixi-live2d-display` is a wrapper
  around Live2D's own `live2dcubismcore.js`, which is not redistributable. A setup script
  fetches it and `.gitignore` keeps it out of the repo. Anyone cloning this runs setup
  first or gets a blank canvas with no useful error.
- **Pin `pixi-live2d-display` against your PixiJS version.** The original library targets
  Pixi 6; community forks track newer Pixi and add lip-sync helpers. Version drift here is
  a known source of wasted afternoons.
