# Aria

A voice-first AI companion with an animated character that sits on your desktop.

You talk to her out loud and she talks back. You can cut her off mid-sentence. She
remembers things about you between sessions, she can look at your screen if you ask,
and she reaches you on Discord when you're away from the desk. The character is a
transparent always-on-top window — a desktop pet, not an app you switch to.

Roughly **1.2–1.4 seconds** from the moment you stop talking to the moment she starts.

```
Setup.bat          <- run this once
Start Aria.bat     <- then this
```

---

## Will it run on my PC?

**Windows, and about 3 GB of disk.** Beyond that it depends on your graphics card, and
setup works this out for you rather than making you guess.

| | Speech recognition | Her voice | The chat model |
|---|---|---|---|
| **NVIDIA, 6 GB+** | GPU, best quality | your PC | your PC, or cloud |
| **NVIDIA, under 6 GB** | GPU, smaller model | your PC | cloud recommended |
| **No NVIDIA GPU** | processor, smaller model | your PC | **cloud needed** |

Only the *chat model* really needs the GPU. Everything else — hearing you, speaking,
the character — runs fine without one. A local chat model on a machine with no GPU is
minutes per reply, which is why setup steers you to a free cloud key instead.

Nothing is hardcoded. If setup guesses wrong, every choice is overridable in
`core\.env`.

## Setting it up

You need two things installed first. Setup checks and tells you if they're missing:

```
winget install --id=astral-sh.uv
winget install --id=OpenJS.NodeJS.LTS
```

Then run **`Setup.bat`**. It installs the Python and Node packages, downloads the speech
models and the character, and writes you a settings file. Takes a few minutes, mostly
downloading. Safe to run again if it stops partway — it picks up where it left off.

If something breaks later, `Setup.bat -Check` diagnoses without changing anything.

## Chat model

The one decision setup can't make for you: where her words come from.

**On your PC** — private, free, no limits, and needs an NVIDIA GPU with ~6 GB spare.

`Setup.bat` offers to install Ollama and download the model for you, and `Start Aria.bat`
starts Ollama if it isn't already running. If you'd rather do it by hand:

```
winget install --id=Ollama.Ollama
ollama pull qwen3:8b
```

Qwen because she's bilingual: a Llama-class 8B answers English well and Traditional
Chinese badly, which quietly breaks half of what `ARIA_LANGUAGE=auto` is for.

Any Ollama model works — put its tag in `core\.env` as `ARIA_LLM_MODEL=`. Her persona
was actually written against
[`nexusriot/Qwen3.5-Uncensored-HauhauCS-Aggressive:9b`](https://ollama.com/nexusriot/Qwen3.5-Uncensored-HauhauCS-Aggressive),
which holds her register noticeably better than the stock model does. It isn't the
default because it's a community fine-tune with no content filtering, and that should be
something you choose rather than something you get.

**In the cloud** — works on any machine, and much better at staying in character. Free
tiers are generous enough for real use. Get a key, put it in `core\.env`:

```
GROQ_API_KEY=gsk_...
ARIA_LLM_BACKEND=groq
```

| Provider | Free tier | Worth knowing |
|---|---|---|
| **Groq** | 1000 requests/day | Fastest. The one to start with. Key: https://console.groq.com/keys |
| Google | generous | Best models, but the free tier says it uses what you send to improve their products |
| OpenRouter | 50/day | Widest model choice, smallest allowance |

Then start her with **`Start Aria (cloud).bat`**.

Either way your voice, her voice, and what she remembers stay on your machine. Only the
text of the conversation goes to a cloud provider, and only if you set one up.

## Starting her

Four launchers. Two choices: where the words come from, and whether the character is on
screen.

| | On screen | Voice only |
|---|---|---|
| **Local model** | `Start Aria.bat` | `Start Aria (voice only).bat` |
| **Cloud model** | `Start Aria (cloud).bat` | `Start Aria (voice only, cloud).bat` |

The voice-only ones skip the character window. She still listens, answers, remembers and
works on Discord — there's just nothing sitting over your work.

**The console window is her.** Closing it stops her. The control panel — `Ctrl+Shift+A`,
or the tray icon — is where you mute, swap her voice, edit her personality, and see what
she remembers. Closing the panel does *not* stop her.

## What she can do

| | |
|---|---|
| **Talk** | Always listening, no push-to-talk. Interrupt her and she stops in 5 ms. |
| **Remember** | Notes about you, kept between sessions. Editable and deletable in the panel. |
| **Emote** | Expressions and lip sync driven by her own speech, on the sentence they belong to. |
| **See your screen** | Only when asked, and only after you arm it. Off by default. |
| **Discord** | Text and voice calls, so she's reachable from your phone. Optional. |
| **Speak Chinese** | English, Traditional Chinese, or follow whichever you speak. |

## Optional: Discord

Off unless you set a token. Lets you talk to her from your phone, and she'll join a
voice channel with you. Setup is a few minutes in Discord's developer portal —
`core\.env.example` walks through it, and `core\tests\check_discord.py` tells you what's
still wrong.

## Optional: your own character

Only Haru ships, because she's a Live2D sample that setup downloads from Live2D's own
repository rather than art bundled into this one. Any Live2D Cubism 3+ model works —
**[docs/CHARACTERS.md](docs/CHARACTERS.md)** covers dropping one in, and
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) covers what Haru's licence does and
doesn't let you do.

## How it fits together

Two processes talking over a local WebSocket:

```
core/       Python — microphone, speech recognition, the model, her voice, memory
overlay/    Electron — the character, the control panel, the tray
```

The split is deliberate: the ML stack wants Python, and transparent always-on-top
windows with a GPU-accelerated character renderer want Electron. Neither is good at the
other's job. Either half runs without the other — the character survives core
restarting, and core runs fine headless.

| Piece | Choice |
|---|---|
| Speech recognition | faster-whisper, `large-v3-turbo` or smaller |
| Voice detection | Silero VAD |
| Chat model | Ollama locally, or Groq / Google / OpenRouter |
| Her voice | Kokoro-82M |
| Character | Live2D via `pixi-live2d-display` on PixiJS |

## Docs

- **[core/README.md](core/README.md)** — the voice loop in depth: latency budget, every
  measurement, and the traps found the hard way
- **[overlay/README.md](overlay/README.md)** — the character window
- [docs/CHARACTERS.md](docs/CHARACTERS.md) — using your own Live2D model
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — how the pieces fit
- [docs/PROTOCOL.md](docs/PROTOCOL.md) — the core ↔ overlay event contract
- [docs/DECISIONS.md](docs/DECISIONS.md) — why each piece of the stack
- [docs/ROADMAP.md](docs/ROADMAP.md) — what was built, in order, and what each step cost

## Licence

MIT for the code — see [LICENSE](LICENSE).

The Live2D runtime and the Haru character are Live2D's, not ours. Neither is in this
repository; setup downloads both onto your machine, and using them means accepting
Live2D's terms. They're free for individuals and for businesses under 10,000,000 JPY of
annual revenue, they don't allow you to redistribute Haru, and they place limits on what
the character may be shown saying — which is worth reading if you plan to point Aria at
an uncensored model.

**[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md)** has the details and links to the
agreements themselves.
