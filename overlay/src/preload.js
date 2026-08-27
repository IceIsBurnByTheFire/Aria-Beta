/**
 * The only bridge between the renderer and Electron.
 *
 * Deliberately narrow. The renderer needs Node for nothing — Live2D is canvas work and
 * the core connection is a plain WebSocket — so `contextIsolation` stays on and this
 * exposes a handful of named calls rather than an open IPC channel.
 */

const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("aria", {
  /** Read once at load; main owns the file. */
  config: ipcRenderer.sendSync("get-config"),

  /** Ask for clicks: true while the cursor is over the character, false otherwise. */
  setInteractive: (on) => ipcRenderer.send("set-interactive", Boolean(on)),

  /** Strip → gear: open the control panel window. */
  openPanel: () => ipcRenderer.send("open-panel"),

  /** Panel → strip: move the quick actions to another corner, or hide them. */
  onStripPosition: (cb) =>
    ipcRenderer.on("strip-position", (_e, position) => cb(position)),

  /** Move the window by a screen-space delta. */
  dragMove: (dx, dy) => ipcRenderer.send("drag-move", dx, dy),
  dragEnd: () => ipcRenderer.send("drag-end"),

  /** Multiply the current scale (wheel up grows, wheel down shrinks). */
  scaleBy: (factor) => ipcRenderer.send("scale-by", factor),

  showMenu: () => ipcRenderer.send("show-menu"),
  quit: () => ipcRenderer.send("quit"),

  /** Hot-swap: main asks the renderer to replace the model, no restart. */
  onSwapModel: (cb) =>
    ipcRenderer.on("swap-model", (_event, rel) => cb(String(rel))),

  /** Tell the tray which emotions the loaded character supports. */
  reportEmotions: (list) => ipcRenderer.send("emotions", list),

  /** Tray → Expression: fire one directly, bypassing core. */
  onPlayEmotion: (cb) =>
    ipcRenderer.on("play-emotion", (_event, name) => cb(String(name))),

  /** Tray → Subtitle: restyle the caption live and show a preview. */
  onSubtitleStyle: (cb) =>
    ipcRenderer.on("subtitle-style", (_event, style) => cb(style)),

  /** Tray → Subtitle → Show sample: pin a sample caption while adjusting. */
  onSubtitleSample: (cb) =>
    ipcRenderer.on("subtitle-sample", (_event, on) => cb(Boolean(on))),
});
