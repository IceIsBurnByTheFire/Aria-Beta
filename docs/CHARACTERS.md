# Using your own character

Only **Haru** ships with this project, and she isn't really shipped either — setup
downloads her from [Live2D's own sample repository](https://github.com/Live2D/CubismWebSamples),
pinned to a commit, because the art isn't ours to redistribute. She works, she's
expressive enough to demo everything, and most people want someone else.

Any **Live2D Cubism 3.0 or newer** model works. Here's how to put one in.

## The short version

1. Drop the model folder into `overlay/assets/models/`.
2. Check it will load: `npm run check-model --prefix overlay`
3. Pick it from the tray menu → **Character**. It hot-swaps, no restart.
4. If you want expressions, add a mapping — see below.

Steps 1–3 give you a character that renders, breathes, blinks, follows your cursor and
lip-syncs to her speech. Step 4 is what makes her *emote*.

## What actually decides compatibility

Not the Cubism version in the marketing copy. The **moc format**:

| File in the folder | Result |
|---|---|
| `.moc3` | Loads. Every moc3 ever produced by the editor works, 3.0 through 5.0. |
| `.moc` | Does not load. That's Cubism 2.x — a different runtime this project doesn't ship. |

The only real failure is a moc3 *newer* than the bundled runtime, which needs a Cubism
version that didn't exist when Cubism Core 5.1 was built.

```bash
npm run check-model --prefix overlay
```

That reads the moc header of every model in the folder and tells you the version and
whether it will load. Run it before wondering why the window is blank.

## The folder

Whatever the model came as, as long as one `.model3.json` sits at the top of it:

```
overlay/assets/models/your-character/
├── your-character.model3.json      <- this is what gets found
├── your-character.moc3
├── your-character.physics3.json
├── textures/
├── motions/
└── expressions/
```

The overlay scans `assets/models/` for any folder containing a `.model3.json`, so the
folder name is yours to choose. It's the name that appears in the tray menu.

## Expressions: the part worth doing

Core talks in **semantic emotions** — `happy`, `sad`, `shy`, `angry`, `surprised`,
`thinking` — and never in your character's file names. That layer exists because the LLM
has to *choose* an emotion, and a model whose expressions are called `f00`–`f07` gives it
nothing to choose with.

So the translation lives in the overlay, in `EMOTIONS` near the top of
`overlay/src/renderer.js`. Add an entry keyed by your folder name:

```js
const EMOTIONS = {
  haru: {
    neutral:   { expression: "F01" },
    happy:     { expression: "F05" },
    sad:       { expression: "F04" },
    angry:     { expression: "F03" },
    surprised: { expression: "F06" },
    shy:       { expression: "F07" },
    thinking:  { expression: "F08" },
  },

  "your-character": {
    happy: { expression: "smile" },      // the name of the .exp3.json, without extension
    shy:   { expression: "blush" },
  },
};
```

**Some characters have no expressions at all** and do it with motion clips instead. Those
map by motion group and index:

```js
  "your-character": {
    happy: { motion: ["Eyes", 1] },      // group name from model3.json, then index
    shy:   { motion: ["Face", 0] },
  },
```

The `aimeng` entry already in that table is a worked example of the motion style, and
`tsubaki` of the expression style — neither ships, but both are left there because they
show the two shapes.

You don't need all seven. Map what the character actually has: core is told the list at
connect time and only ever asks for emotions on it. Mapping an emotion the character
can't show teaches the model to emit a marker that gets silently dropped.

### Finding the names

Open the `.model3.json`. Expressions are under `FileReferences.Expressions` with a `Name`
each; motion groups are the keys under `FileReferences.Motions`, and the index is the
position in that group's array.

Guessing which file is "happy" from the name alone is unreliable — `F05` on Haru is the
smile because it sets `EyeLSmile`/`EyeRSmile` to 1, not because of what it's called. If
the names are opaque, cycle through them and look:

```bash
npm start --prefix overlay -- --cycle-emotions
```

## Hiding things from the AI

Some models ship expressions you don't want an LLM reaching for. `exclude_from_ai` in
`overlay/config.json` is a case-insensitive substring filter over expression and motion
names — anything matching is never reported to core, so it's never in the prompt:

```json
"exclude_from_ai": ["nsfw", "jb"]
```

The character can still use them; the model is just never offered them.

## Size and position

Scroll wheel over the character scales her, and the result is remembered. Base size is
`width` / `height` in `overlay/config.json`. The control panel's **Look** tab has the
same controls plus the quick-action strip's corner.

## If the window is blank

In order of likelihood:

1. **The Live2D runtime didn't download.** `Setup.bat -Check` says so. Without
   `overlay/src/vendor/live2dcubismcore.min.js` the canvas renders nothing and there is
   *no error* — the single most confusing failure in this project.
2. **It's a `.moc`, not a `.moc3`.** Cubism 2.x. `check-model` will tell you.
3. **No `.model3.json` at the top of the folder.** Some downloads nest everything one
   level deeper than you expect.
4. **The character is off-screen or fully transparent.** Try `--cycle-models`.

## Licences

Whatever you drop in is between you and whoever made it. Models bought from Booth,
VTuber commissions and the like all come with their own terms; redistribution is usually
the thing they forbid, which is why this repo downloads Haru rather than containing her.

Haru's own terms are in [THIRD-PARTY-NOTICES.md](../THIRD-PARTY-NOTICES.md). One clause
is worth knowing before you swap her out for anything, because it is about behaviour
rather than files: Live2D restrict what their sample characters may be shown **saying**,
and this project puts a language model's words in a character's mouth. If you point Aria
at an uncensored model, that combination is yours to judge — and it is a reason to prefer
a character whose licence you control.
