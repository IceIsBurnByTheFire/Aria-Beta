# aria-overlay — M3: the character

Aria's Live2D character as a transparent, always-on-top, click-through desktop window.

## Setup

```bash
npm install --prefix overlay
```

`postinstall` fetches Live2D's Cubism Core runtime and the Haru model. To re-run it by
hand:

```bash
npm run fetch-assets --prefix overlay
```

## Run

Core and overlay are separate processes. Start core first — though the order does not
actually matter, since the overlay reconnects on a backoff and core runs happily with
nothing attached.

```bash
uv run --directory core python -m aria
```

```bash
npm start --prefix overlay
```

Add `--debug` to open devtools detached:

```bash
npm start --prefix overlay -- --debug
```

## Controls

The window has no title bar, no taskbar entry and never takes focus, so **the tray icon
is the only way to quit**. `Ctrl+Alt+Q` backs it up in case the icon lands in Windows'
hidden-icon overflow.

| Action | How |
|---|---|
| Quit | Tray icon → Quit, or `Ctrl+Alt+Q` |
| **Switch character** | **Tray → Character** — hot-swaps, no restart |
| Move | Drag the character |
| Resize | Scroll wheel over the character |
| Size presets | Tray → Size (Small / Medium / Large / Huge) |
| Back to normal size | Tray → Reset size |
| Recentre | Tray → Reset position |
| Reload after editing config | Tray → Reload character |
| **Try an expression** | **Tray → Expression** — fires it directly, no LLM involved |
| **Caption size & position** | **Tray → Subtitle** — changes apply live and show a preview |
| Devtools | Tray → Open devtools, or `--debug` |

Two dev flags, both useful for eyeballing a newly added character:

```bash
npm start --prefix overlay -- --cycle-emotions
```

```bash
npm start --prefix overlay -- --cycle-models
```

Position and scale are saved to `window-state.json` in Electron's userData directory and
restored next launch. Everything off the character stays click-through, so dragging and
scrolling only respond where she actually is.

## Characters

Three are installed. **Tray → Character** swaps between them live — the old model is torn
down and the new one built in place, and the choice is written back to `config.json`.

| | Format | Size | Expressions / motions | Notes |
|---|---|---|---|---|
| `haru` | Cubism 3.0 | 2400×4500 | 8 expressions, `Idle`/`Tap` | Live2D's sample |
| `tsubaki` | Cubism 5.0 | 3024×6264 | 5 expressions, `Idle` | Arms are physics-only |
| `aimeng` | Cubism 4.0 | 3600×8000 | 8 motion groups | Has real `hand_raise`/`hand_lower` |

`aimeng` is heavy — a single 8192px texture, ~268 MB decoded. Total footprint across all
three sits around 1.5 GB and is bounded, not leaking: textures are freed on swap, but
PixiJS caches what it has seen. Downscaling that sheet to 4096 would halve it.

`aimeng` also has no `.exp3.json` files at all — its expressions are *motions*
(`eyes_heart`, `blush_on`, `mouth_surprised`). M4 will need to drive it through
`motion` events rather than `expression` ones.

### Captions

**Tray → Subtitle** covers size, position (top / middle / bottom), alignment, nudging by
12 px at a time, and turning captions off. Changes are written back to `config.json`
immediately.

**Tray → Subtitle → Pin sample caption** puts a sample line on screen and leaves it
there until you switch it off, so a run of adjustments can be judged against something
real. The sample deliberately wraps onto two lines — width, line spacing and `max_lines`
are all things you want to see before committing.

Without pinning, any change still previews for 12 seconds, and the timer resets on each
change, so a burst of small tweaks stays visible throughout. The pin is
session-only — it is an adjustment aid, and having it survive a restart would be
baffling.

For pixel-precise placement, edit it directly:

```json
"subtitle": {
  "enabled": true,
  "font_size": 13,
  "position": "bottom",
  "align": "center",
  "offset_x": 0,
  "offset_y": 0,
  "max_lines": 4,
  "max_chars": 40
}
```

`max_chars` caps line length so the caption stays near the character's own width rather
than spanning the whole window. It is applied in CSS `ch` units, which measure the width
of "0" — wider than the average letter in a proportional font, so a line fits somewhat
more than the number suggests. Treat it as a dial, not a precise count.

`align` also decides where the narrowed box sits, not just how text sits inside it —
aligning left within a full-width box would strand the text far from the character.

`offset_y` is positive *downward* from whichever `position` anchors it, so the same
number means the same direction wherever the caption sits. Font size is clamped to
8–48 px and `max_chars` to 12–120. Any key you leave out falls back to the defaults
above, so a partial block is fine.

### Emotions

Core never names an expression file. It sends a *semantic* emotion — `happy`, `shy`,
`angry` — and `EMOTIONS` in `renderer.js` translates it for whichever character is
loaded. That table is the only place model-specific knowledge lives.

| Character | Emotions it can show | Driven by |
|---|---|---|
| `haru` | neutral, happy, sad, angry, surprised, shy, thinking | `.exp3.json` |
| `tsubaki` | neutral, happy, shy, sad, angry, serious | `.exp3.json` |
| `aimeng` | neutral, happy, love, surprised, shy, smug, sad, thinking | motion clips |

The overlay reports this set in its `hello`, and core puts it straight into the system
prompt — so the LLM is only ever offered what the loaded character can actually do, and
a hot-swap changes the vocabulary mid-conversation.

Two things worth knowing if you edit the table:

- **haru's mapping was read off its parameters, not guessed.** `F05` closes the eyes
  with `EyeLSmile`/`EyeRSmile` at 1 (happy); `F07` sets `ParamTere` — 照れ, the blush
  parameter (shy); `F03` lowers the brows and turns the mouth down (angry).
- **One motion per emotion.** A Cubism motion manager holds a single slot, so firing two
  clips at once just means the second replaces the first. That is why `aimeng`'s entries
  are single motions even where a pair would read better.

### Adding another

Drop the model's whole folder into `assets/models/` — `.moc3`, `.model3.json`, textures,
expressions, motions and physics together. It appears in the tray menu automatically;
no config edit needed.

### Keeping things away from the AI

```json
"exclude_from_ai": ["nsfw", "jb"]
```

Any expression or motion group whose name contains one of these is withheld from the
`hello` core receives, so the LLM is never offered it and cannot trigger it on its own.
Case-insensitive substring match. The motions still exist and still work if triggered
deliberately — this only shapes what the model is *told about*.

### Will it work?

Don't go by what the listing calls it. Run:

```bash
npm run check-model --prefix overlay
```

It reads the model's actual format and prints the exact `config.json` line to paste.

The rule it applies is about the **moc format**, not the marketing version:

| File | Verdict |
|---|---|
| `.moc3` + `.model3.json` | Loads |
| `.moc` + `.model.json` | Does not — Cubism 2.x, a separate deprecated runtime this project does not ship |

Every `.moc3` the editor has produced is accepted by a newer runtime, so "is it Cubism 4
or 5" barely matters — Haru is moc3 version 1, which is **Cubism 3.0**, and it renders
fine on a Cubism 5.1 core. The only real failure is a moc3 *newer* than the bundled
runtime, fixed by deleting `src/vendor/` and re-running `fetch-assets`.

The version byte at offset 4 of any `.moc3`, per Cubism Core's own `MocVersion_*`
constants:

| Byte | Editor |
|---|---|
| 1 | Cubism 3.0 |
| 2 | Cubism 3.3 |
| 3 | Cubism 4.0 |
| 4 | Cubism 4.2 |
| 5 | Cubism 5.0 |

Bundled Cubism Core 5.1 accepts up to 5.

### Where to get one

- **Live2D's official samples** — free, several characters, the same license Haru comes
  under. Good for trying things.
- **Booth.pm** — the main marketplace for Live2D models. Plenty of paid ones, some free.
  Check the license covers your use, then run `check-model` on it before believing the
  listing's version claim.
- **Commission an artist** — VGen and Twitter are where most vtuber rigging work happens.
  Expect real money and weeks of turnaround for something custom.
- **Rig your own** in Live2D Cubism, if you have the art and the patience.

After swapping, check the terminal. The renderer logs the model's real expression and
motion names at startup, and warns if the model is missing any parameter the animation
expects — mismatched parameter names are otherwise a silent failure where everything
renders beautifully and the mouth never moves.

## How it works

Core decides, the overlay renders. Every event is described in
[../docs/PROTOCOL.md](../docs/PROTOCOL.md).

**Two clocks run here, deliberately separate.** Idle life — breathing, blinking, gaze
following the cursor — runs locally and never stops, connected or not. Driven animation
— mouth, expression, state — comes from core. A character that freezes when the backend
hiccups looks broken; one that keeps breathing looks like it is waiting for you. When
core goes away the character desaturates and keeps living.

**Parameter layering is decided up front**, because idle and driven animation both write
to overlapping Cubism parameters each frame and whichever runs last silently wins:

| Parameter | Owner |
|---|---|
| `ParamMouthOpenY` | Lip sync, outright |
| `ParamEyeLOpen` / `ParamEyeROpen` | Idle blink (expressions will take these in M4) |
| `ParamAngleX/Y/Z`, `ParamEyeBallX/Y` | Gaze + idle sway |
| `ParamBodyAngleX`, `ParamBreath` | Idle |

**Lip sync is amplitude, not phonemes.** Core samples playback RMS at 50 Hz and sends a
0–1 opening; the renderer eases between frames so 50 Hz reads smoothly at 60+ fps. It
costs nothing, cannot drift out of sync, and looks convincing. Real visemes would need
phoneme timings Kokoro does not expose.

**Click-through is what makes it livable.** `setIgnoreMouseEvents(true, {forward: true})`
passes clicks to whatever is underneath while still delivering mousemove, so the
character can watch your cursor and ask for clicks back when you hover it. Without this
the window blocks a rectangle of desktop and you uninstall it within a day.

## The Haru model's real capability names

Reported by the model itself at startup, and **not** what the filenames suggest:

- **Expressions:** `f00` … `f07` — lowercase, despite the files being `F01.exp3.json` …
  `F08.exp3.json`
- **Motion groups:** `Idle`, `Tap` — groups, not individual clips

M4 must map Aria's emotional vocabulary onto these. The overlay already sends the real
list in its `hello`, so core can constrain the LLM to what the loaded model actually has
rather than to a hardcoded guess.

## Gotchas

- **`npm install` succeeding does not mean Electron works.** Electron ships a stub
  package and fetches a ~200 MB binary from a postinstall script. When that script
  fails, npm still exits 0 and `node_modules/electron` still exists — with no
  `electron.exe` in it and no `path.txt`. The launcher starts the overlay hidden, so the
  process dies in under a second and the only symptom is a character that never appears
  while the voice loop works perfectly. Seen twice: once on Electron 33, whose pinned
  `extract-zip@2.0.1` silently no-ops on Node 24 (the download succeeded, the unpack did
  nothing), and once on a plain reinstall where npm simply skipped the script. Both
  `setup.ps1` and `start-aria.ps1` now run `node -e "require('electron')"` and say so.
  The fix is `npm install --force`, or `node node_modules/electron/install.js`.
- **She can be restored onto a display that no longer exists.** `window-state.json`
  keeps her last position, and nothing about it is checked against the monitors that are
  attached *now*. Unplug a screen, change resolution, or change DPI scaling and the
  saved rect can sit outside every display — invisibly, because the window is
  transparent, frameless, `skipTaskbar` and unfocusable. Nothing to see, nothing to
  alt-tab to. `placeOnScreen` checks her *feet* rather than the window: she is drawn
  anchored to the bottom centre, so a rect can overlap a display generously and still
  show only the empty air above her head. Measured on a real machine: 681x1021 saved at
  (1210, 540) against a 1707x1067 work area — 497x527 of window on screen, none of it
  character.
- **Never drag with `setPosition` on a fractionally-scaled display.** It changes the
  window's *size*. Measured on a 150% monitor: 400 calls took 400x600 to 800x1000 —
  one pixel per call, both axes — because each move round-trips the size DIP → physical
  → DIP and the rounding only goes one way. One drag is hundreds of mousemove events, so
  she grew visibly while being dragged and `layout()` refit the model to match. Dragging
  uses `setBounds` with the size recomputed from `state.scale` every event, which makes
  the size a pure function of what the user chose rather than a value that carries
  between calls; measured drift is then zero. It never survived a restart, because
  `state.scale` was never wrong — only the live window was — which is what makes it easy
  to dismiss as imagination.
- **The wheel only works over the character, so the minimum scale has to leave something
  to aim at.** At 0.4 the window was 160x240 and the model inside it is tall and narrow,
  which left barely a hundred pixels of character — too small to put the cursor on, and
  the wheel is the obvious way back. `MIN_SCALE` is 0.6, and Tray → Reset size exists so
  there is always a way out that needs no aiming.
- **Cubism Core is not redistributable.** `fetch-assets.mjs` pulls it from Live2D's CDN
  into `src/vendor/`, which is gitignored. Without it the canvas renders nothing and
  says nothing — a completely silent failure.
- **PixiJS 6 needs `new Function`** for shader codegen, which the page's CSP forbids.
  `@pixi/unsafe-eval` is Pixi's own supported replacement. Its browser build
  self-installs onto `globalThis.PIXI` at load — there is no `install()` to call, and
  calling one throws.
- **Script order in `index.html` is load-bearing:** pixi → unsafe-eval → cubism core →
  the plugin. The plugin registers against a `PIXI` that must already be patched.
- **Pinned to PixiJS 6.** `pixi-live2d-display@0.4.0` peer-depends on `^6` across all
  its Pixi packages. Do not upgrade one without the other.
- **Renderer errors are invisible without forwarding.** The window is transparent, so a
  crashed renderer and a working one look identical. `main.js` pipes the renderer console
  to the terminal for exactly this reason.
- **Expressions and motions load lazily**, and the fetch-plus-parse lands on the frame
  where an emotion is first used — which is the start of a sentence, so it reads as a
  stutter. It also disappears once each emotion has been used once, which makes it look
  intermittent and easy to blame on something else. `preloadAnimations()` loads
  everything at model load instead; the renderer logs how many and how long.
- **The renderer warns about slow frames.** Anything over 28 ms is logged with the last
  event that preceded it, so a stutter can be attributed rather than guessed at. Tune
  with `frame_warn_ms` in `config.json`.
- **Archives from China and Japan often arrive with mangled filenames.** If a model's
  files look like `┤╗.moc3`, the zip was extracted with GBK or Shift-JIS bytes read as
  CP437. `model3.json` still references the correct names, so nothing resolves and you
  get a blank canvas with no error. `check-model` catches this — it verifies every
  referenced path actually exists. The fix is to rename the files (ASCII is safest) and
  update `model3.json` to match.
- **VTube Studio models hide their expressions.** VTS keeps expression and motion
  bindings in its own `.vtube.json`, so `model3.json` frequently lists neither even
  though the `.exp3.json` files are right there. They have to be added to
  `FileReferences.Expressions` before anything outside VTS can use them.
- **Two parameter naming conventions exist.** `ParamMouthOpenY` (Cubism 3+) and
  `PARAM_MOUTH_OPEN_Y` (carried over from Cubism 2) — `aimeng` uses the latter. Writing
  to a parameter a model lacks is a silent no-op, so a convention mismatch gives you a
  character that renders perfectly and never moves its mouth. `renderer.js` resolves
  each logical parameter against `PARAM_CANDIDATES` at load and warns about anything it
  cannot find; add new spellings there rather than per-model config.
- **VTS model3.json files reference motions that do not exist.** `aimeng`'s listed two
  missing `MaoMao` clips. `check-model` catches this.
- **`config.json` may acquire a BOM** if you edit it in a Windows editor. `JSON.parse`
  rejects those, which would silently reset your chosen character — `main.js` strips it
  before parsing.
- **Leave the `EyeBlink` and `LipSync` groups empty.** `pixi-live2d-display` builds its
  own auto-blink and lip-sync from them, which then fight the renderer — it drives
  `ParamEyeLOpen`, `ParamEyeROpen` and `ParamMouthOpenY` directly every frame.

## Using a different character

Drop a `.moc3`-based model under `assets/models/` and set `model` in `config.json` to its
`.model3.json`. Everything else is parameter names, which are conventional across Cubism
models, and the expression/motion lists are read from the model at runtime.

Haru is one of Live2D's sample models — fine for personal and development use. A public
release wants their terms checked, or a commissioned character.
