# Core ↔ Overlay protocol

JSON messages over a localhost WebSocket. Core is the server, the overlay connects.

Pinning this down early is worth the effort: it is the seam between the two halves, and a
stable seam means the character renderer and the voice pipeline can be built and tested
independently.

## Principles

- **Core decides, overlay renders.** The overlay never makes a state decision. If the
  character does something, core told it to — with the sole exception of idle animation.
- **Events are fire-and-forget.** No request/response, no acks. A dropped event should
  degrade the animation, never desync the system.
- **The overlay must survive core disconnecting.** Character keeps breathing and blinking,
  reconnects on a backoff. A crashed backend should not leave a frozen corpse on screen.

## Core → Overlay

### `state`

Broadcast on every state machine transition. The overlay's main animation cue.

```json
{ "type": "state", "value": "idle" | "listening" | "thinking" | "speaking" }
```

### `expression`

Set the character's emotional expression. Emitted when core parses an `[emotion]` marker
out of the LLM stream.

```json
{ "type": "expression", "value": "happy", "intensity": 0.8, "hold_ms": 2000 }
```

`intensity` 0.0-1.0. `hold_ms` is how long before decaying back to neutral; omit for
"hold until replaced".

Start with a small vocabulary and grow it only when the character actually needs more:
`neutral`, `happy`, `sad`, `angry`, `surprised`, `thinking`, `embarrassed`.

### `viseme`

Mouth shape, streamed at audio frame rate during playback (~50 Hz is plenty).

```json
{ "type": "viseme", "open": 0.65 }
```

v1 is a single amplitude-derived `open` value, written straight to `ParamMouthOpenY`. If
the TTS backend later exposes phoneme timings, this grows a `shape` field without breaking
anything that ignores it.

### `motion`

Trigger a one-shot animation clip.

```json
{ "type": "motion", "name": "wave", "loop": false }
```

### `subtitle`

Text for an optional caption bubble. Sent incrementally as the LLM streams.

```json
{ "type": "subtitle", "text": "partial text so far", "final": false }
```

### `vision`

Whether Aria can currently see the screen, and whether she is capturing right now.

```json
{ "type": "vision", "watching": true, "capturing": false }
```

Sent on every change, and it must render as a **persistent** indicator rather than a
transient notice. Screen capture is the most invasive thing in the system; the user has
to be able to tell it is on by looking, not by remembering. `capturing` is true only for
the moment a frame is actually grabbed.

### `transcript`

Both sides of the conversation, kept and in order. Distinct from `subtitle`, which is a
caption for the character: it replaces itself and carries no author.

```json
{ "type": "transcript", "role": "you" | "aria", "text": "...", "at": 1738-, "via": "discord" }
```

`via` is present only when the turn did not come through the microphone. Absent means
spoken. Aria's side is what was *heard* — a reply cut off by barge-in appears in the
transcript the way it sounded, not the way it was generated.

### `notice`

Something the user should see — errors, capability state changes.

```json
{ "type": "notice", "level": "info" | "warn" | "error", "text": "Screen capture ON" }
```

### `settings`

A full snapshot of everything the control panel renders — voice, mute, memory, wake word,
screen watching, and a read-only `discord` block (`configured`, `connected`, `user`,
`owner_set`, `speak_replies`). Broadcast on connect and after every command, because the
panel draws nothing optimistically: arming screen capture can *refuse*, and a toggle that
flips instantly then silently disagrees with the thing it controls is worse than no
toggle.

Three fields carry the editable half of who she is:

- **`memory`** — `[{id, text, source, created_at}]`. The `id` is what makes a note
  editable: `forget` matches text loosely so it works said out loud, which is the wrong
  rule for a button next to one specific line.
- **`persona`** — the editable persona only, not the assembled prompt.
- **`system_prompt`** — the whole assembly she actually receives, read-only. Persona,
  screen rules, emotion vocabulary, her notes, the clock. The most useful thing in the
  panel when she is behaving oddly.

## Overlay → Core

Sparse by design. The overlay is mostly an output device.

### `interact`

The user clicked or dragged the character. Core decides whether that means anything.

```json
{ "type": "interact", "kind": "click" | "drag_start" | "drag_end", "target": "head" }
```

### `command`

User-triggered control from the overlay's tray menu or hotkeys.

```json
{ "type": "command", "name": "mute" | "unmute" | "screen_on" | "screen_off" | "stop" }
```

`stop` is barge-in by button — the same cancellation path the voice interrupt uses. Worth
having early because it lets you test cancellation before AEC works.

Editing commands carry a `value`:

| `name` | `value` | |
|---|---|---|
| `add_note` | text | Saved as `source: "you"`, so eviction never drops it |
| `edit_note` | `{id, text}` | Keeps `created_at` and `source` — a correction is not a new fact |
| `delete_note` | id | Exactly one note. `forget` (text) may take more, by design |
| `set_persona` | text | Applies to the next turn; refused if empty or enormous |
| `reset_persona` | — | Deletes the override. The built-in was never overwritten |

Anything refused answers with a `notice` at `warn` before the snapshot. A save that
silently does nothing is the failure this whole panel is built to avoid.

**Every handler is guarded.** An exception out of one used to propagate into the
connection handler and close the client with a 1011 — a `print` hitting an unencodable
character on a cp1252 stdout took the entire control panel offline. One bad command
should be the only casualty.

### `hello`

Sent on connect so core knows what the renderer can do.

```json
{
  "type": "hello",
  "role": "renderer" | "panel",
  "model_format": "live2d",
  "model_name": "hiyori",
  "emotions": ["neutral", "happy", "sad"],
  "motions": ["idle", "wave", "nod"]
}
```

Core uses this to constrain what it asks for — no point emitting `[embarrassed]` if the
loaded model has no such expression file. Feeding the available list into the LLM's system
prompt is the clean way to keep it honest, and it means dropping in a different Live2D
model reconfigures the character's emotional range automatically. It is also what lets
core repair a marker that arrived from the model missing a bracket: a closed vocabulary is
the only thing that makes `shy]` safe to strip and `I'm happy for you` safe to leave.

The field is **`emotions`**, and they are semantic names the overlay maps onto whatever
the loaded character actually has — this document said `expressions` for some time while
the code read `emotions`, which is a good way to lose a character's entire emotional
range to a silent empty list.

`role` distinguishes the control panel from a character renderer. The panel is a second
client on this same socket and it also says hello, but it draws no character: core reads
capabilities from the first non-panel client, because an empty list from the panel
otherwise strips every expression out of the system prompt and the only symptom is a face
that stops moving.
