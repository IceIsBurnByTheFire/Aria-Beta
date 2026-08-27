# aria-core — the voice loop (M1) + interruption (M2)

A spoken conversation with a terminal, which you can talk over. Mic in, speaker out, no
character yet.

## Setup

Commands use `uv run --directory .` so they work from anywhere and need no `cd`. On
Windows PowerShell 5.1 there is no `&&`, so run them one at a time.

Install dependencies:

```bash
uv sync --directory .
```

Fetch Silero VAD, Kokoro and its voice pack (~340 MB):

```bash
uv run --directory . python -m aria.setup_models
```

Whisper pulls itself from HuggingFace on first run. Ollama must be running with the
target model.

## Run

```bash
uv run --directory . python -m aria
```

Speak; it replies. Talk over it and it stops. Enter does the same thing from the
keyboard. Ctrl-C prints the latency breakdown.

**On open speakers, turn barge-in off.** The mic hears Aria's own voice, the VAD reads
it as you talking, and she interrupts herself into silence the instant she starts.
Headphones have no echo path and need nothing. Real echo cancellation is M6.

```bash
uv run --directory . python -m aria --no-barge-in
```

```bash
uv run --directory . python -m aria --list-devices
```

```bash
uv run --directory . python -m aria --list-voices
```

```bash
uv run --directory . python -m aria --voice am_michael --end-silence 500
```

## Verify

```bash
uv run --directory . pytest tests/ -q
```

Two harnesses need models and a GPU, so they are scripts rather than tests.

Live capture plumbing — frame rate, drops, VAD triggering. Say nothing and it still
tells you whether the mic path is healthy:

```bash
uv run --directory . python tests/check_mic.py 5
```

Drives the whole pipeline from synthesised speech instead of a microphone: Kokoro makes
a question, the real Silero endpointer consumes it frame by frame, and the real turn
handler answers it. Covers everything except the mic driver:

```bash
uv run --directory . python tests/e2e_offline.py
```

Cuts a reply off mid-sentence and checks what survives — that playback stops, that
history records only what was heard, and that the next turn still makes sense:

```bash
uv run --directory . python tests/e2e_bargein.py
```

## Interruption

Playback stops **5 ms** after the trigger. Flush order is load-bearing: playback is
flushed first, freezing `seconds_played`, and only then is the response task cancelled —
which unwinds TTS synthesis and the LLM stream with it.

The half that is easy to get wrong is what goes into history. At the moment of the cut,
chunks are already queued that will never be heard. In the test case Aria generated 113
characters and 20 reached the ear; history recorded those 20. Asked "sorry, go on", she
gave the full account instead of assuming she had already given it.

Record the generated text instead and nothing errors — Aria simply believes she answered,
and the conversation stops making sense two or three turns later for no visible reason.

`spoken.spoken_prefix` does the mapping, truncating a straddling chunk proportionally by
word count. It is a pure function with its own tests because the failure mode is silent.

A 300 ms grace window after playback starts stops a trailing breath or the tail of your
own previous sentence from killing the reply before it begins. Pressing Enter has no
grace — a deliberate stop means now.

One caveat: `seconds_played` counts samples handed to the audio device, so up to one
block (~43 ms) may not have physically reached the ear yet. Small enough not to matter
for word-level truncation.

## Measured on the dev machine

RTX 5080 Laptop, Core Ultra 9 275HX. Median over three turns, `large-v3-turbo` +
Qwen3.5 9B + Kokoro:

| Stage | Median | Notes |
|-------|--------|-------|
| VAD endpoint hold | 600 ms | Config, not compute — see below |
| STT | **0 ms** | ~292 ms of work, hidden inside the hold |
| LLM first token | 210–235 ms | |
| LLM to first chunk | 40–90 ms | Generating enough text to speak |
| TTS first chunk | 310–440 ms | The variable one |
| **Perceived** | **1170–1330 ms** | User stops talking → hears audio |

Against the < 1200 ms target: it lands either side of the line depending on the run. The
variance is almost entirely Kokoro — synthesis cost tracks first-chunk length, and how
long that chunk is depends on where the LLM's wording puts the first clause boundary.
Same question, same settings, 309 ms one run and 437 ms the next.

Call it *about 1.2 seconds, and conversational*. Not snappy.

## Why STT is free

The endpoint hold is ~600 ms of silence in which the GPU does nothing and the user is
already waiting. So transcription starts speculatively 250 ms into a pause, before the
turn is known to be over. It finishes with time to spare, and by the time the endpointer
commits, the transcript is already sitting there.

If the pause turns out to be mid-sentence the work is thrown away — `Utterance.frames`
records how much audio the final turn covers, and a speculative result is only adopted
if it covered at least that much.

**Throwing it away does not stop it, and that used to compound.** `task.cancel()`
cancels the coroutine awaiting the thread; `asyncio.to_thread` has no way to interrupt
work already running, so an abandoned transcription keeps the GPU for its full duration
alongside its replacement. Measured on the same model:

| | median |
|---|---|
| a transcription on its own | 331 ms |
| the same one with an abandoned pass still running | 626 ms |

Every 250 ms pause used to start another one, so a hesitant sentence with three pauses
had three transcriptions competing and the mechanism that exists to make a turn *faster*
was what made it slow — matching the 729–773 ms STT times seen in a real session against
a 292 ms benchmark. Only one speculation is in flight at a time now; the next pause
starts a fresh one once it has finished.

Segment timestamps are also switched off (`without_timestamps=True`). Nothing downstream
reads them — the endpointer already decided where the turn begins and ends — and they
are decoded tokens like any other: 304 → 282 ms median for identical text.

## The remaining latency, and why the obvious lever doesn't work

The obvious move is to cut `end_silence_ms`. It mostly doesn't help, because shrinking
the hold also shrinks the window that hides STT:

```
perceived = end_silence + max(0, stt − (end_silence − speculate_after))
                        + llm_ttft + gen + tts
```

Once `end_silence` drops below `speculate_after + stt` (≈ 540 ms here), every
millisecond saved on the hold is paid straight back as exposed STT. The floor is:

```
speculate_after + stt + llm + gen + tts  =  250 + 292 + 723  =  1265 ms
```

So 600 → 450 ms buys ~60 ms and costs real usability, because a shorter hold cuts people
off when they pause to think. The levers that actually move the number:

1. **TTS first chunk (437 ms)** — the largest term after the hold. Kokoro costs roughly
   160 ms fixed plus 0.14 ms per ms of audio produced, so this is mostly a function of
   first-chunk length. It is already clamped to 40 characters; going shorter trades
   prosody for latency and should be decided by ear.
2. **`speculate_after` (250 ms)** — lowering it lowers the floor directly. The limit is
   how often it fires on ordinary inter-word gaps and wastes a transcription.
3. **LLM first token (233 ms)** — a smaller model, or a draft model for the opening few
   tokens.

## Expressions

The LLM marks emotion inline — `[happy] That actually worked.` — and core strips the
markers before the text reaches TTS, then emits each one **when its sentence actually
plays**. That last part is the whole point: chunk two is synthesised while chunk one is
still playing, so firing on arrival would land every expression a sentence early.

The available emotions come from the overlay's `hello`, not from a fixed list, because
the three installed characters have different ranges and one names its expressions
`f00`–`f07`. Anything the current character cannot show is dropped here rather than
sent.

`[3]` and `[see below]` are left alone — only letter-and-underscore markers of 2–24
characters are treated as emotions, so ordinary brackets still reach the speech.

**A 9B drops a bracket every so often**, and the result is not a marker that fails to
fire — it is the word arriving as content. Caught posting to Discord: *"…whatever sauce
you have. shy] Tell me which one."* On the voice path the same thing is read aloud
mid-sentence. `emotion.extract` repairs a marker missing either bracket, but only against
the character's own emotion list: a closed vocabulary is what makes `shy]` safe to strip
while `I'm happy for you` is left alone. With no character attached there is no
vocabulary to be sure about, and it does nothing.

## Screen awareness

Off by default. Arm it with `--watch-screen`, or out loud: *"watch my screen"* /
*"stop watching my screen"* — those bypass the LLM entirely, because a capability this
invasive should turn on exactly when asked rather than when inferred. While armed, the
overlay shows a persistent red dot.

```bash
uv run --directory . python -m aria --list-monitors
```

```bash
uv run --directory . python -m aria --watch-screen --monitor 2
```

Capture is on-demand and change-gated, never a feed. A keyword test decides whether an
utterance is about the screen, so it costs no latency and is wrong in obvious ways
rather than mysterious ones.

Two backends, `ARIA_VISION_BACKEND=ollama|claude`:

| | Needs | Notes |
|---|---|---|
| `ollama` (default) | `ollama pull qwen2.5vl:7b` (~6 GB) | Free, private, offline; weaker on small text |
| `claude` | `ANTHROPIC_API_KEY` or `ant auth login` | Sonnet 5; much better at reading UI text; costs per call |

**The backend sets the capture resolution, and that is most of the quality difference.**
The local model gets 1500px; Sonnet 5 is the first Sonnet-tier model with high-resolution
vision and takes a 2576px long edge, so a 1920×1080 monitor would reach it whole, with no
downscale at all. `VisionConfig.edge` picks per backend — reading either `max_edge` or
`claude_max_edge` directly is the bug that silently sends Claude a downscaled image.

**Running local is the default, and the reason is privacy, not cost.** On the `claude`
backend screenshots leave the machine. That is a real trade against everything else in
this project, and it is the one thing the local model is unambiguously better at.

`qwen2.5vl:7b` is installed and works offline: ~3.9 s warm, ~18 s on the first turn after
the model loads, of which capture is ~200 ms. Switching backends is one env var and no
code change. The Claude path is wired and tested up to the API call; the key on this
machine has no credit, so preflight declines it and says so.

## Wake word

```bash
uv run --directory . python -m aria --wake-word
```

Off by default — alone at a desk, always-listening is nicer, and it's what everything
up to here was tuned against. With it on she ignores anything she isn't named in.

Saying her name opens a **30-second window**, so follow-ups don't need it again; every
exchange resets the clock. Her name is stripped before the model sees it, or every turn
arrives with her being addressed by name and she starts treating that as remarkable.

The gate sits **after STT, not before it** — Whisper already transcribes every
utterance and is far better at hearing "Aria" than a small wake model would be, with no
extra dependency and no custom model to train. The cost is that background speech still
reaches the GPU; what it no longer reaches is the LLM, the TTS, or her mouth.

Matching is fuzzy because Whisper writes her name as "area", "arya" or "ariya" — but
fuzzy on a four-letter word needs guarding: **"Maria" scores 0.89 against "aria"**
because it contains the whole word, and "aerial" scores exactly 0.80. A fuzzy match
needs the same first letter and a length within one of the real word.

## Memory

```bash
uv run --directory . python -m aria --memory
```

Shows everything she remembers and where the file is. The file itself is
`core/data/memory.json` — plain JSON, gitignored, **written the first time she runs**,
so it won't exist until then. `ARIA_MEMORY` points it somewhere else.

She keeps notes two ways: a background pass after each turn extracts anything durable,
and `remember that…` / `forget…` are caught before the LLM ever sees them. That second
path matters — left to a 9B, "remember I take my coffee black" earns a warm agreeable
reply and nothing written down, which is the worst outcome available, because you walk
away believing she knows.

Notes you dictate are stored as your own words in quotes. Rewriting the pronouns into
third person to match the extractor produces "he take his coffee black" — swapping *I*
for *he* leaves the verb behind, and conjugating properly needs a verb table nobody
wants to own.

The extractor fails closed: told to find anything interesting, a 9B records that you
said hello. It answers `NONE` unless the fact is durable, about you, and not guessed
at. If she's missing things you'd want kept, loosen `EXTRACT_PROMPT` — much easier than
cleaning out a file full of junk.

## Which model, and where

Two launchers, one Aria. `Start Aria.bat` runs the conversation model on the GPU;
`Start Aria (cloud).bat` runs it on OpenRouter. Everything else — voice, memory,
character, Discord, screen reading — is identical.

```bash
uv run --directory . python -m aria --llm-backend openrouter
```

| | Local (default) | Cloud |
|---|---|---|
| Cost | free | free tier, see the table below |
| Rate | unmetered | 20–30/minute |
| First token | ~210 ms measured | a round trip, plus free-tier queueing |
| Privacy | nothing leaves the machine | every message, **including her notes about you** |
| Offline | yes | no |
| Persona compliance | the weak spot | the reason to bother |

Three providers, one backend — they all publish the same OpenAI-compatible endpoint, so
they are a table in `config.CLOUD_PROVIDERS` rather than three classes. Pick with
`ARIA_LLM_BACKEND` in `core/.env`; the cloud launcher passes no provider of its own, so
switching is one line.

| | Free tier | Default model | Measured first token |
|---|---|---|---|
| **`groq`** | 30/min, 1000/day (14,400/day on Llama 3.1 8B) | `llama-3.3-70b-versatile` | **~350 ms** |
| `google` | not published — check AI Studio | `gemini-3.1-flash-lite` | ~870 ms. Free tier content is used to improve Google's products |
| `openrouter` | 20/min, 50/day (1000 with $10 lifetime credit) | `nvidia/nemotron-3-super-120b-a12b:free` | untested — no key |

**Groq is the recommendation and it is not close.** Against local's ~210 ms it costs
about 140 ms, which is inside the noise of the TTS variance the latency budget already
absorbs. It is 2.5× faster than Gemini, has published limits rather than "check your
dashboard", and no training clause on the free tier. Both were measured over four real
turns with the full persona loaded.

`openrouter` has never made a live call — its transport and failure handling are covered
against a mock, but its default model is reasoned rather than measured.

### Which Groq models are usable

Eight chat models are visible on a free key. Only three are worth pointing Aria at, and
the reason the others fail is the same one Gemini had — reasoning spent from the reply
budget. Two calls each, "say hi in three words", 160 tokens:

| Model | Result |
|---|---|
| `llama-3.3-70b-versatile` | **511 ms**, clean. The default |
| `llama-3.1-8b-instant` | 334 ms, clean — but 8B, so no persona gain over local |
| `openai/gpt-oss-120b` | 771 ms, answers, ~82 tokens spent reasoning |
| `openai/gpt-oss-20b` | spent all 160 on reasoning, returned **nothing** |
| `qwen/qwen3.6-27b` | spent all 160, and leaks a literal `<think>` block into the reply |

`qwen` is why `_ThinkFilter` exists. Reasoning is supposed to stay out of `content` and
usually does — OpenRouter takes `reasoning: {exclude}`, Gemini takes `reasoning_effort` —
but qwen on Groq emits `<think>Here's a thinking process:…` as visible text, which
reaches the synthesiser and gets read aloud. A provider knob cannot fix it, because
setting a reasoning parameter on a *non*-reasoning model like `llama-3.3-70b-versatile`
would cost a rejected request and a retry on every turn. Filtering the stream is free and
model-agnostic, and it is streaming-safe: the tag arrives split across tokens.

### What a live Gemini key actually taught

**Thinking tokens are spent from `max_tokens`, and that breaks everything quietly.**
Gemini 3.x reasons by default. Asking for three words with a 160-token budget:

| | thinking | spoken | finish |
|---|---|---|---|
| no knob | 156 | 4 | `length` — truncated mid-sentence |
| `reasoning_effort: low` | 155 | 5 | `stop` |
| `reasoning_effort: none` | 6 | 5 | `stop` |

At the chunker's 40-token first chunk the unknobbed version returns **nothing at all** —
a 200 with an empty stream, which looks exactly like a broken SSE parser. `low` is not
low enough; the provider table sets `none`.

**Model choice is a latency cliff, not a slope.** Two calls each, same prompt:

| Model | Result |
|---|---|
| `gemini-3.5-flash` | works, **~10.6 s**, intermittent 503 |
| `gemini-3.1-flash-lite` | works, **~1.1 s** |
| `gemini-3.6-flash` | 400 on this key |
| `gemini-2.5-flash`, `-lite` | 404 on this key |
| `gemini-2.0-flash` | 429 |

Ten seconds is unusable out loud, so the lite model is the only real option rather than
the budget one. With the full persona loaded it measured **~870 ms median first token**
over four turns — against ~210 ms local, so a spoken turn lands near 2 s rather than 1.3.

**503 "experiencing high demand" is common on a free key** and clears in a second or two,
so transient statuses are retried three times with backoff before she says anything. It
is oversubscription, not a setup error, and her line says so without sending anyone to
the terminal.

**The persona survives Google's safety training.** This was the main risk and it did not
materialise: in character, using pet names, no refusals, no assistant-speak. The public
Discord persona also held — asked what her owner had been up to, she answered *"That is
his to tell."*

**The privacy trade is real and it is the reason to consider `groq` instead.** Google's
free tier states content is used to improve their products; paid tiers are excluded. Her
memory notes are in the system prompt. Memory *extraction* stays local either way, so
what she writes down is never sent — but what she is told is.

Model ids are **per-provider**: a Llama id sent to Gemini is a 404. `ARIA_CLOUD_MODEL_GROQ`,
`ARIA_CLOUD_MODEL_GOOGLE` and `ARIA_CLOUD_MODEL_OPENROUTER` can all be set at once so
switching provider does not drag the wrong id along.

**Local stays the default and that is not just inertia.** The entire latency budget was
measured against it, screen capture already keeps frames on the machine for privacy
reasons, and her memory notes sit in the system prompt — sending those to a third party
is a real trade, not a config change.

**The reason to want cloud is persona compliance.** Most of the defensive work in this
project exists because a 9B copies persona examples verbatim, invents shared history,
ends every reply with a question, and drops brackets off emotion markers. A much larger
model does all of that better.

**Memory extraction stays local either way.** Aria makes two calls per turn — the reply,
and a background pass deciding whether anything is worth remembering. Sending the second
one to the cloud would halve a 50-a-day allowance to pay a large model for a small
classification, and ship the most personal thing in the system over the network. See
`llm.background_for`. Cloud mode therefore still wants Ollama running; without it she
falls back and says so.

### Picking a model

```bash
uv run --directory . python -m aria --list-cloud-models
```

That prints every provider's free tier, which key it wants, whether you have one, and
OpenRouter's live catalogue. Free ids are retired and renamed regularly, so the live list
beats anything written here — a stale id fails as a 404 on the first thing you say, which
does not look like a stale config file.

**`groq` and `openrouter` have never made a live call.** Their transport, parsing,
retries and failure messages are covered against a mock, and their defaults are reasoned
rather than measured. Only `google` has been run for real.

**If a cloud Aria sounds like a helpful assistant, that is the model, not a prompt bug.**
Instruction-tuned models pull hard toward assistant register, and how hard varies by
provider — Gemini held the persona fine, Groq's Llama 3.3 70B held it well enough to pass
the whole Discord e2e. If yours keeps saying "I'd be happy to help", try a different model
id before rewriting the persona; a local model is generally the most compliant.

Running out is a normal end to a free day, not a malfunction — she says so in her own
words and the terminal gives the numbers.

## Discord

She can be reached as a Discord bot — same memory, same conversation, same notes. Type to
her from a phone and the microphone knows about it a minute later.

Three steps, and the last two are the ones people skip:

1. **Token.** Developer Portal → your app → Bot → Reset Token. Put it in `core/.env` as
   `ARIA_DISCORD_TOKEN` (copy `core/.env.example`). Switch on **Message Content Intent**
   on the same page — without it every message arrives empty. Having a token is what
   turns Discord on; there is no second flag.
2. **Invite her to a server.** *You cannot DM a bot you share no server with*, so a bot
   invited nowhere is unreachable by the exact route it exists for. Your own empty server
   is fine.
3. **`ARIA_DISCORD_OWNER`** — your user id (right-click yourself → Copy User ID, after
   Settings → Advanced → Developer Mode). See below for exactly what it gates.

### What `ARIA_DISCORD_OWNER` controls

Anyone who shares a server with the bot can open a DM to it. A voice in the room is
self-authenticating; a Discord message is a string from an account. This variable is the
only thing that ties a message to you, so three capabilities hang off it:

| | Owner set, message from you | Owner set, anyone else | Owner unset |
|---|---|---|---|
| Gets a reply at all | yes | **ignored entirely** | yes, anyone |
| Screen — `watch my screen`, *"what's on my screen"* | yes | no | **no, for everybody** |
| Memory — `remember that…`, `forget…`, and the background note-taker | yes | no | **no, for everybody** |

Ordinary conversation is never gated — unset, she is simply a chatbot anyone can talk to
who cannot see your desktop or write anything down.

**Memory is gated for a subtler reason than the screen, and a worse one.** A screenshot
leaking is immediate and obvious. A note is written once and read back weeks later with
total confidence and no record of where it came from — `memory.py` calls a wrong note
worse than no note for exactly that reason. Ungated, *"remember that he hates his job"*
typed by a stranger becomes something she tells you she knows, in a month, unprompted.
The background extractor needs the same gate: it writes notes *about you*, inferred from
whoever happened to be talking.

If the id is wrong rather than missing, she ignores every message including yours —
`check_discord.py` resolves it against Discord and says so.

Check all three at once:

```bash
uv run --directory . python tests/check_discord.py
```

It connects, names her, lists her servers, resolves the owner id, and prints the invite
URL if she is in none. Every failure it can detect says what to do about it.

In a server she stays quiet unless @mentioned or replied to; `DiscordConfig.channels`
lists channel ids where she answers everything. Replies are posted, not spoken — you are
usually on Discord because you are *not* at the desk. `--speak-discord` says them out loud
as well, for when you are.

```bash
uv run --directory . python -m aria --no-discord
```

**A typed reply is written to different rules than a spoken one.** The persona's delivery
constraints — twenty-five words, symbols as words, no markdown — belong to the speech
synthesiser, and a message written to them reads oddly. `DISCORD_STYLE` replaces that part
for typed turns and leaves everything about who she is alone. It sits at the very end of
the system prompt so moving between the desk and your phone mid-conversation doesn't
invalidate Ollama's cached prefix.

### Letting other people talk to her

`ARIA_DISCORD_PUBLIC=1` opens her up to everyone in your servers. They get a **different
Aria**, and the separation is structural rather than a rule she is asked to follow.

**The room decides what she can reach. The person decides who she is.** Those are two
different questions, and conflating them was a real mistake — see below.

|  | You, in a DM | **You, in a server** | Anyone else |
|---|---|---|---|
| Persona | `PERSONA` | **`PERSONA`** + `OWNER_IN_PUBLIC` | `PUBLIC_PERSONA` — a guest |
| Conversation | `_history`, shared with your voice turns | its own, per channel | its own, per channel |
| Her notes about you | in the prompt | **never** | never |
| Screen, memory writes | yes | **no** | no |
| Drives the character on your desktop | yes | no | no |
| `--speak-discord` | yes | never — nobody else's message comes out of your speakers | never |

**Why the room still decides what she can reach.** Deciding that per-person is the
obvious version and it is wrong: you would be @mentioning her in a room full of friends
while she holds your notes and your voice conversation, and the first *"so what's he been
up to?"* has an honest answer sitting in her context. Making the room the unit means the
leak cannot happen, rather than depending on a rule she remembers to follow.

**Why the person decides who she is.** The first version let the room decide that too,
and it was wrong in the other direction: talking to her in your own server got you the
guest persona, which reads as *"she's just the model itself"* — precisely what that
persona was written to be. It bought no safety at all, because the persona contains
nothing about you. She is herself to you anywhere now; `OWNER_IN_PUBLIC` tells her the
two things she cannot work out from the prompt she has been handed — that others can read
it, and that her notes genuinely are not there, without which she agrees to remember
things and keeps nothing.

Her public prompt is built from scratch rather than by removing things from yours. A
subtractive version has to keep being correct every time `_system_prompt` grows; this one
was never given anything to leak.

**Your DMs are never public**, whatever the setting. Anyone who shares a server with the
bot can open one, so a public DM would be a private channel with a stranger in it.

She stays quiet in servers unless @mentioned or replied to. `ARIA_DISCORD_CHANNELS` is a
comma-separated list of channels where she answers everything.

Three behaviours worth knowing, all found by pointing the real model at them:

- **She won't correct a wrong guess about you either.** Denying gives away as much as
  confirming, and it is the version a model volunteers happily because it feels like
  discretion. *"That's his to say"* is the whole answer.
- **She won't promise to remember anything.** She keeps no notes in a server, and
  agreeing warmly while writing nothing is the failure the memory design exists to
  prevent — now aimed at your friends instead of you.
- **Public turns are queue-bounded** at three waiting. Everything shares one turn lock,
  so an unbounded queue lets one person with a stuck enter key starve your microphone.
  Beyond that they are dropped, and the console says so.

Every word she says to anyone is printed in your terminal. She is your assistant talking
to your friends; being able to read that without opening Discord is the point.

### Voice calls

Set `ARIA_DISCORD_VOICE_CHANNEL` to a channel id and she follows you in and out of it.
Join, and she joins; leave, and she leaves.

**She needs Connect and Speak on that channel**, which the original text-only invite did
not ask for. Server Settings → Roles → her role → enable both, or re-invite her with the
current `INVITE_PERMISSIONS`. Without them Discord does not refuse the connection — it
simply never completes the handshake, and forty seconds later `voice_state._connect`
raises `TimeoutError` naming nothing at all. `join_voice` checks both first now, so the
answer is immediate and says which one is missing.

**End-to-end encryption has to be implemented, not avoided.** discord.py advertises DAVE
whenever the optional `davey` package is present, and `discord-ext-voice-recv` knows
nothing about it — so every packet arrives still wrapped, Opus reports `corrupted stream`,
and it is telling the truth. The obvious fix is to stop advertising support so the call
downgrades, and Discord answers the voice handshake with close code **4017, "This channel
requires a client supporting E2EE via the DAVE Protocol"**. A channel can mandate it. So
`discord_voice.dave_decrypt` peels the layer off each packet using the session discord.py
already holds but never wired to the receive path.

**Decoding happens in our sink, not in the extension.** `wants_opus()` returns True so
the router hands packets over untouched, because a decode failure on its thread ends
reception permanently while one on ours costs a single 20 ms frame.

**The sink counts speech and generated silence separately, and that distinction is a bug
fix rather than a nicety.** `SilenceGeneratorSink` fills the gaps between his words with
invented `SilencePacket`s — necessary, because Discord stops transmitting entirely while
you are not speaking and the endpointer would otherwise never see a pause end a turn
(measured before it: a correct, in-character reply **37 seconds** late). But those packets
carry a canned `OPUS_SILENCE` payload with no end-to-end layer on it, so they decrypt and
decode *whatever is wrong with the call*.

Counting them as evidence of hearing produced the worst failure this feature has had. One
real packet arriving is enough to start the generator; its output then printed **"⦿ hearing
you in the call"** and, because the "no packets are usable" warning was gated on the same
counter, silenced that warning permanently. The console showed a call that was working and
she never answered a word. `heard` counts only packets that came off the wire; `decoded`
counts everything, and the difference is reported when the call ends.

```bash
uv run --directory . python tests/check_discord_voice.py
```

That spends one real connection: joins the channel, checks she is allowed to connect and
speak, plays a real Kokoro line, then **listens for twelve seconds while you say
something** and reports which stage the audio stopped at. Which number is zero is the
diagnosis:

| symptom | where it broke |
|---|---|
| nothing arrived | you were muted, or Discord was not transmitting |
| arrived, none decoded | the end-to-end layer, or Opus |
| only silence decoded | your speech specifically is failing to decrypt |
| decoded, no frames | the re-blocking between the sink and the VAD |
| frames, but silent | the audio is arriving as digital silence |

It still cannot tell you it *sounded* right — sit in the channel for that.

**Nothing about a turn changes.** Same VAD, same endpointer, same speculative
transcription, same barge-in, same handler. Only where the frames come from and where
the audio goes.

**The rates are integer ratios, which is most of why this is tractable.** Discord speaks
48 kHz stereo both ways; Silero and Whisper want 16 kHz mono (48/16 = 3), and Kokoro
produces 24 kHz mono (48/24 = 2). No drift to accumulate, no fractional resampler to keep
in phase. Coming in, the decimation is filtered rather than naive — dropping two samples
in three folds 12 kHz down to 4 kHz, which does not error, it just makes Whisper quietly
worse.

**No echo canceller is needed, and that is not luck.** Discord never sends you your own
audio, so her voice is not in the stream she is listening to. The entire problem M6 exists
to solve does not arise, and barge-in works untouched.

**It is exclusive, not additive.** In a call the sound card is closed and the desktop
speakers are silent. You are at one desk with one headset: two open ears would hear every
sentence twice and endpoint on whichever copy arrived first, and two mouths would put her
voice in the room and in the call a beat apart. `Playback` enforces this — exactly one
consumer may drain the buffer, because two would each get half the samples and
`seconds_played` would count the sum, which is what decides where a barge-in truncates
history.

**Multiple speakers are mixed into one stream.** Correct while you are the only one who
can join that channel, and the obvious place to start when that stops being true.

### Emoji

Her emoji come from the `[emotion]` marker she already emits for her face, not from asking
her to pick one. `DISCORD_MOOD_EMOJI` in `config.py` is the map — edit it there.

Asked to "use an emoji that fits", a 9B produces 🤖💻🔥: the same glyph for three different
feelings one turn and something tonally wrong the next, because it is matching on the
topic rather than reporting a mood. The markers are a closed vocabulary she is already
fluent in, so her face and her text end up two renderings of one signal instead of two
guesses at it. Same reasoning as the screen and memory commands — decide it in code where
code can.

The marker's **position** is the information: it opens the sentence it colours, so the
emoji lands at the end of *that* sentence, which is why substitution runs before the
markers are stripped. One per message; `[neutral]` maps to nothing, because without a way
to say *no emoji* every message gets one.

She still types one herself occasionally despite being told not to — an observed 😩 for
weary, which no marker covers. Hers wins and nothing is added, since overruling her reads
worse than allowing it and stacking a second beside it is worse than either.

The mood vocabulary is deliberately **not** the character's expression set: whether her
Live2D model has a `flirty` pose has nothing to do with whether 😏 is the right emoji, and
a headless run has no character at all. Anything outside what the loaded character can
show is dropped before it reaches the overlay.

Offline verification — real turns, real model, fake gateway, no token:

```bash
uv run --directory . python tests/e2e_discord.py
```

## Editing her persona and her memory

Both live in the control panel (`Ctrl+Shift+A`, the gear on her strip, or the tray).

**Memory** — add a note, click any note to correct it, `×` to delete. Anything you type
counts as something you told her, so it is never evicted to make room. Notes carry an
`id` and the panel addresses them by it: `forget` matches text loosely so it works said
out loud, and that same rule under a button would let deleting *"he has a cat called
Widget"* also take a note that just said *"cat"*.

**Persona** — the whole of who she is, in a text box. Saved to `core/data/persona.txt`
beside `memory.json`, gitignored for the same reason: what she remembers and who she is
are both yours, and a file you cannot find is one you cannot correct. Changes apply to
her next reply; nothing reloads and the conversation is kept.

The panel edits the *persona*, not the prompt. What she receives is assembled per turn —
persona, then screen rules, the character's emotion vocabulary, her notes, the clock, and
a one-turn reminder if she just asked a question. The panel shows that assembly read-only
underneath, which is what you actually want when the question is "why did she say that".

Two guards, both for silent failures. An **empty persona is refused**: with no character
text she keeps answering, fluently and as nobody, and it reads as the model having changed
rather than a box having been cleared. And **reset deletes the override rather than
restoring a copy** — the built-in lives in code, so it cannot drift or be half-written.

```bash
uv run --directory . python tests/e2e_panel.py
```

## Language

```bash
uv run --directory . python -m aria --language auto
```

`auto` is the default, and it is the answer for anyone who speaks two languages: **she
follows you.** Say something in English and she answers in English; say something in
Chinese and she answers in Traditional Chinese, in a Mandarin voice. Decided per turn,
including mid-conversation, with nothing to switch. `en` and `zh` still pin her to one —
worth doing in a noisy room, because it removes the one decision detection can get wrong.

`ARIA_LANGUAGE` in `core/.env` sets the default for every launcher.

**No new model was needed for any of it.** Kokoro's voice pack already ships eight
Mandarin voices — `zf_xiaoxiao zf_xiaoni zf_xiaobei zf_xiaoyi` and `zm_yunxi zm_yunjian
zm_yunxia zm_yunyang` — and both Llama 3.3 70B and the local Qwen write Chinese fine.
What was missing was configuration and, it turned out, a phonemiser.

**Three things have to agree, and only one of them fails loudly.** A wrong voice sounds
wrong instantly. A wrong phonemiser refuses outright — espeak rejects both `zh` and
`zh-cn` with *"not supported by the espeak backend"* — or, worse, accepts and lies; see
below. The quiet failure is `stt.language`: Whisper is *told* the audio is English, so
Chinese speech comes back as confident English-sounding nonsense and the fault looks like
the microphone. `Language` in `config.py` keeps all of it in one place for that reason.

**Detection is per turn and it measures well.** `language=None` identified 5/5 short,
one-clause lines across both languages, the shortest (`早安。`) at 0.78 confidence. The
detected language is printed next to the transcript, but only in an `auto` run — in a
pinned one it is a constant, and a constant printed every turn is noise.

**It is also restricted to the two languages she has, and that is not paranoia.** In the
first real session after this shipped, Whisper answered **Korean** on two turns out of
five — once on an ambiguous noise and once on Mandarin — and the reply was fluent,
in-character and useless: *"你說韓文，但我不知道你想說什麼"*. She speaks two languages and
Whisper chooses from ninety-nine, so a third answer is a misdetection by construction
rather than a fact to act on. `STTConfig.allowed` re-runs the turn as the best-ranked
language she actually speaks, using the `all_language_probs` the first pass already
produced. The second pass lands only on turns that were going to be wrong anyway, so a
correct detection costs nothing. Verified: a Korean clip that detected as `ru` now comes
back as `en`, and the English and Chinese cases are untouched.

**The voice is chosen per *chunk*, not per turn**, and it reuses the chunker's own script
test rather than a second one. The thing that decided where to cut a chunk should be the
thing that decides how to pronounce it, or a reply gets cut as Chinese and spoken as
English. Every language entry carries the other script's voice too, which is not an edge
case in either direction: an English reply names a 中文 file, a Chinese reply names
`docker`, and `af_bella` reads Han characters as *nothing at all*.

**The chunker only understood ASCII punctuation, and it broke streaming completely.**
Measured before the fix: the same sentence produced **four chunks in English and one in
Chinese**. `。！？` were not in the terminator set, and — the part that would have defeated
just adding them — the pattern required a trailing space, which Chinese never has. One
chunk means she synthesises the whole reply before saying a word, so the entire streaming
design silently stops working. Nothing errors; she just gets slow.

Cut lengths are divided by three for CJK, because that ratio is a property of the writing
system rather than a preference: a CJK character is roughly a syllable where a Latin one
is roughly a fifth of one, so the 40-character first chunk tuned for English is ~2.5 s of
speech and ~10 s of Mandarin. Measured after: 1–2 chunks per reply, first chunk 1.3–1.9 s.

**The prompt block is written in Chinese, and includes Chinese sample lines.** An
instruction to use language X lands far better in language X — in English it reads as a
fact about her and comes back as an English sentence agreeing to speak Chinese. The
samples matter more than the rules, for the reason this project keeps rediscovering:
without them she inherits the register of the English examples and produces stiff,
translated-sounding Mandarin. Verified across three turns: zero Simplified characters, and
the best-friend register intact (`怎麼了老兄`).

**Both of the caveats this section used to end on are now closed, and the first one was
much worse than it looked.**

*Kokoro's Mandarin voices are trained on misaki phonemes, and `kokoro-onnx` phonemises
with espeak.* Nothing errors. `lang="cmn"` is accepted, audio comes out, and it is
recognisably a female Mandarin voice — saying different words. Measured by having Whisper,
which is very good at Mandarin, read her own speech back:

| phonemiser | intelligibility | `早安。` | `聽起來很累。` |
|---|---|---|---|
| espeak `cmn` | 0.46 | 相安 | 听起来**人类** |
| misaki `zh`  | 0.73 | 早安 | 听起来很累 |

The remaining gap is mostly the scorer — Whisper answers in Simplified, so 這/这 counts as
an error against a Traditional source even when the transcription is perfect. What matters
is the *shape*: espeak produces wrong words, which is the difference between an accent and
a wrong sentence. `tts/chinese.py` phonemises with misaki and passes `is_phonemes=True`.
`lang="zh"` in this codebase means "ours", not espeak's.

Latin runs inside a Chinese chunk are cut out and phonemised by espeak separately, because
misaki passes English through as literal text which then reaches Kokoro's tokeniser *as if
it were IPA* — "Rust" becomes whatever /R/, /u/, /s/, /t/ happen to mean. The persona
explicitly tells her to leave package names alone, so that is the common case.

*Whisper transcribes Mandarin as Simplified whatever you ask it.* `opencc`'s `s2twp` runs
in ~0.1 ms, is idempotent on text that is already Traditional, and fixes vocabulary as well
as glyphs (内存 → 記憶體). It is applied to Chinese transcripts only. Without it his own
words arrive on screen and into her history in the script she is told never to use — and a
model copies the conversation it is shown far more readily than it follows a rule above it.
If `opencc` is missing she still starts, with a warning; that is a quality problem, not a
broken assistant.

End to end after both, through the real Whisper and the real Kokoro — her mouth into her
own ears — 6/6 lines detected correctly, routed to the right voice, and transcribed back
in Traditional characters at 0.88–1.00 similarity.

## Voice

`af_bella` by default; `--voice` or `ARIA_VOICE` switches, and there are 54 to choose
from (11 female US, 4 female UK). Kokoro runs on CPU at RTF ~0.34, which keeps the GPU
free for Whisper and the LLM.

**The `[emotion]` markers reach the voice, not just the face.** M4 gave her expressions
and left her sounding identical whether she was happy, shy or sad, which is uncanny in
a specific way. `EMOTION_VOICE` in `tts/kokoro_backend.py` maps each emotion to a rate,
pitch and gain.

Kokoro exposes speed and nothing else — no pitch. Pitch comes free anyway: resampling
by `p` moves pitch *and* rate together, so asking Kokoro for `rate / pitch` first leaves
the rate correct once the resample has had its way. Measured, the two come out
independent — `surprised` asks for 1.06 pitch and lands at 1.065 while still shortening
the line. It costs nothing: 703 ms to synthesise a 2 s line with emotion, 705 ms without.

Keep the numbers small. Past ~8% the formants move with the pitch and she goes from a
person to a chipmunk; the effect should be felt rather than heard.

## Speaker mode

```bash
uv run --directory . python -m aria --speaker-mode
```

Without it, open speakers break barge-in in the worst way: the mic hears Aria's own
voice, the VAD reads it as you talking, and she interrupts herself into silence the
moment she starts. Speaker mode subtracts what she is playing from what the mic hears.

Off by default because it loads a native DLL (`pyaec` → SpeexDSP), and the headphone
path shouldn't depend on one. If the DLL is missing it says so and carries on with
headphone behaviour rather than refusing to start.

Check your own room before trusting it — a simulated room can't tell you about your
speakers:

```bash
uv run --directory . python tests/check_aec.py
```

It reports the measured round trip and whether echo still trips barge-in, and names the
value to put in `barge_in.aec_delay_ms` if the configured one is leaving echo through.

**Set the round trip or nothing works.** This machine measures ~512 ms, which is past
the entire 400 ms filter — uncompensated, the canceller removed 11 dB and made the
trigger count *worse* than leaving it off. `aec_delay_ms` shifts the reference back by
the round trip so the filter only models the room's tail; with it set, the same
recording goes to 0 triggers and 65 dB. The estimate wanders a few tens of
milliseconds between runs and that is fine — the filter absorbs it.

The trade: speech quieter than the residual echo can't be told apart from it, so
whispering over her won't interrupt; normal speech will.

## Gotchas that cost real time here

- **Silero v5's ONNX graph needs 64 samples of context prepended to every 512-sample
  frame.** Feed it a bare frame and it runs happily, returns ~0.0 for everything, and
  the VAD simply never fires. No error. `tests/test_vad.py` guards this.
- **A linear echo canceller is not enough on its own, and more filter doesn't help.**
  Speex plateaus around 14 dB on real speech at any tail length, and Silero still hears
  speech in the residual. The suppressor in `audio/aec.py` is what actually closes it.
- **The echo reference must be what was *played*, not what was queued.** Feeding the
  queue lines the filter up against audio the room hasn't heard yet; it never converges
  and the failure looks exactly like having no canceller at all.
- **In the persona, examples beat instructions.** A 9B copies the sample lines far more
  literally than it follows the rules above them. The first draft asked for contractions
  in the rules and then wrote every example without them — she said "I am" and "it is"
  all day. A sample line about sleep turned into asking about sleep three replies
  running. Write the examples as the output you actually want, and vary them.
- **`think=False` is mandatory** for reasoning models. Qwen3.5 routes everything to a
  separate `thinking` field; the spoken `content` never arrives inside any usable token
  budget.
- **int8 compute types fail on Blackwell** (`CUBLAS_STATUS_NOT_SUPPORTED`). float16 works.
- **First CUDA call costs ~16 s** of PTX JIT for sm_120, then ~0.5 s once cached.
  Warmup at startup keeps it out of the first turn.
- **Whisper hallucinates on near-silence** — "Thank you.", "you", "Thanks for watching!"
  arrive with high confidence. Filtered by an explicit list, not by probability, because
  neither of Whisper's own confidence signals works here. Measured on **digital
  silence**: it returns "Thank you." at `no_speech_prob` **0.000** with an `avg_logprob`
  of **-0.28** — a *better* score than a genuine "Okay." at -0.68. Both point the wrong
  way, so no threshold on either can separate them.
- **That list was also eating real speech, which is worse.** Every phrase on it is an
  ordinary thing to say to an assistant, so "Okay.", "Thanks.", "Thank you.", "Bye." and
  "So?" were transcribed perfectly and then discarded — she did nothing, which from the
  outside is indistinguishable from not having heard. The discriminator Whisper lacks is
  whether anyone was talking, and Silero already knows: `Utterance.voiced_s` measures
  0.35–0.58 s for real one-word turns and **0.00 s** for silence, hiss, hum, a click or
  a breath. The list now only applies below `stt.min_voiced_ms` (250 ms), which sits in
  the middle of that gap with wide margin either way. Verified end to end through the
  real endpointer and the real Whisper: 7/7 short replies reach her, silence and fan
  hiss still discarded.
