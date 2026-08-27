# Decisions

## Decided

### Name: **Aria**

A solo vocal performance — fits a character whose whole existence is voice. Two syllables,
distinct phonetics, and it survives becoming a wake word later (a lot of names don't;
anything starting with a soft consonant is hard to detect reliably).

The project name and the character name do not have to stay coupled. The repo can be
stable while the character gets renamed on a whim.

### Character format: **Live2D**

The aesthetic people actually picture when they say "vtuber skin". Cheap to render, sharp
at any size, and there is a large body of existing models.

Rendered with `pixi-live2d-display` on PixiJS inside the Electron overlay. Three
consequences to plan around:

- **The Cubism Core runtime is not redistributable.** `pixi-live2d-display` is a wrapper —
  it needs Live2D's own `live2dcubismcore.js`, which has to be downloaded from Live2D and
  cannot be committed to the repo. A setup script fetches it; `.gitignore` keeps it out.
- **Check the fork against your PixiJS version.** The original library targets Pixi 6.
  Community forks track newer Pixi and some add lip-sync helpers. Pin versions early —
  this is a known source of wasted afternoons.
- **Models are art assets you don't get for free.** Live2D publishes official sample models
  (Hiyori, Haru, Natori and others) under a free-material license that covers personal and
  limited use — the right starting point. A custom character means commissioning art and
  rigging, which is real money and real time. Worth starting on a sample and only
  commissioning once the system is worth looking at.

Chosen over VRM knowingly: VRM would have let you generate a character in VRoid this
afternoon for free, but mediocre 3D reads as *cheap* in a way mediocre 2D does not.
[PROTOCOL.md](PROTOCOL.md) stays format-agnostic regardless, so this is one renderer module
if it ever needs revisiting.

### LLM: **Local default, cloud for vision**

Ollama drives conversation — free, private, offline, no rate limits, already installed here
with a 9B model. Chat is the overwhelming majority of turns and a 9B handles it fine.

The screen-reading path routes to the Claude API, where the quality gap over a local model
is largest and the call volume is lowest. Both sit behind one interface:

```python
async def stream(messages, images=None) -> AsyncIterator[str]
```

The persona lives in the system prompt, not the model choice. Swapping backends must not
change who the character is.

VRAM note: a local model shares 16 GB with Whisper and Kokoro. If VRAM gets tight, Whisper
is the one to shrink — `large-v3-turbo` is already fast enough that a smaller model buys
little, but a `distil` variant frees real headroom.

### TTS: **Kokoro-82M**

Small, fast, genuinely good quality for its size, Apache-2.0. The current sweet spot for
streaming low-latency speech.

Build the interface so swapping takes an afternoon. Expect to change this once you have
heard the character talk for an hour — voice is the most identity-defining component in
the system and you will have opinions you cannot predict now. Piper is the CPU-only
fallback if GPU contention becomes a problem; GPT-SoVITS is the upgrade path if you decide
you want a *distinctive* voice rather than a good generic one.

### Architecture

- **Python core + Electron overlay, split over a WebSocket.** Neither ecosystem does the
  other's job well. See [ARCHITECTURE.md](ARCHITECTURE.md).
- **Voice loop before character.** The character is the fun part; the voice loop is where
  the project can fail. See [ROADMAP.md](ROADMAP.md).
- **Screen capture on-demand and change-gated, never a continuous feed.** Continuous costs
  a fortune on cloud models and adds nothing on a screen that is static most of the time.
- **Barge-in ships with headphones first, AEC last.** The half-duplex middle ground feels
  bad and teaches you nothing you need for the real fix.

---

## Still open

### Always-listening or wake word

Always-listening is a better experience and simpler to build. It also means every
conversation in the room, every video you play, and every phone call gets transcribed and
sent to an LLM.

**Leaning: always-listening for development, with a hard mute that is obvious from the
overlay.** Add a wake word once you start leaving it running around other people —
openWakeWord trains custom words locally, and "Aria" has the clear consonants to detect
well.

Deferring this costs nothing; it is additive to the VAD path either way.

### Voice character

Kokoro ships several voices. Which one the character *is* — and whether it should
eventually be a cloned custom voice — is a decision best made by listening, not by
reasoning about it in a document. Revisit after M1.
