/**
 * The panel's bridge to the overlay process.
 *
 * There are genuinely two sources of truth and the panel needs both. Voice, memory,
 * wake word and screen watching live in *core* and arrive over the WebSocket the
 * character already uses. Subtitle styling, window scale, which Live2D model is
 * loaded and where the quick strip sits are *overlay* state, held in main.js and
 * written to config.json — core has never heard of any of them.
 *
 * Rather than route overlay settings through core and back, which would make core the
 * owner of things it does not control, the panel just gets a second channel. Two
 * channels because there are two owners; each one still has a single source of truth.
 */

const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("overlay", {
  /** Everything main.js knows: subtitle, model, models list, strip, scale. */
  get: () => ipcRenderer.invoke("panel:get"),

  /** Merge a patch into the subtitle style. Applies live and persists. */
  setSubtitle: (patch) => ipcRenderer.send("panel:subtitle", patch),

  /** Swap the Live2D character in place. No restart. */
  setModel: (rel) => ipcRenderer.send("panel:model", rel),

  /** Where the quick strip sits on the character window. */
  setStrip: (position) => ipcRenderer.send("panel:strip", position),

  /** Window scale, same range as the tray's Small/Medium/Large/Huge. */
  setScale: (scale) => ipcRenderer.send("panel:scale", scale),

  /** Fires when main.js changes something, so two open panels cannot drift. */
  onChanged: (fn) => ipcRenderer.on("panel:changed", (_e, data) => fn(data)),
});
