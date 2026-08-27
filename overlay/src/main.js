/**
 * Electron main process — the desktop-pet window.
 *
 * The window itself is four settings plus one method call:
 *   transparent + frame:false  → no chrome, no background
 *   alwaysOnTop                → stays above other windows
 *   focusable:false            → clicking her never steals focus from your real work
 *   setIgnoreMouseEvents       → clicks pass through to whatever is underneath
 *
 * That last one is what makes this livable. Without it the window blocks a rectangle
 * of desktop. With `forward: true` the renderer still receives mousemove, so it can
 * tell when the cursor is over the character and ask for the clicks back.
 *
 * The consequence of all that: there is no title bar, no taskbar entry, and no focus,
 * so there is nothing to close. The tray icon is not a nicety — it is the only way out,
 * and a global shortcut backs it up in case the tray is hidden by the system.
 */

const {
  app, BrowserWindow, ipcMain, screen, Menu, Tray, nativeImage, globalShortcut,
} = require("electron");
const fs = require("node:fs");
const path = require("node:path");

const CONFIG_PATH = path.join(__dirname, "..", "config.json");
const QUIT_SHORTCUT = "CommandOrControl+Alt+Q";

/**
 * Run the panel without the character on screen.
 *
 * The two windows were always independent — the panel is a second client on core's
 * socket and has never needed the character for anything — so this is a startup choice
 * rather than a mode. It exists because the voice loop is the useful half: you may want
 * her listening, and her controls to hand, without a Live2D model sitting over your work.
 *
 * Everything character-shaped is hidden rather than left to no-op. A Size menu that
 * silently does nothing is worse than one that isn't there.
 */
const PANEL_ONLY =
  process.argv.includes("--panel-only") || process.env.ARIA_PANEL_ONLY === "1";

let win = null;
let tray = null;
let interactive = false;
let currentEmotions = [];
let config = { model: "haru/Haru.model3.json", width: 400, height: 600 };
let state = { x: null, y: null, scale: 1 };

const statePath = () => path.join(app.getPath("userData"), "window-state.json");

/** Parse JSON tolerating a leading BOM — Windows editors add them freely, and
 *  JSON.parse rejects them, which would silently reset the user's chosen character. */
function readJson(file) {
  return JSON.parse(fs.readFileSync(file, "utf-8").replace(/^﻿/, ""));
}

function loadConfig() {
  try {
    config = { ...config, ...readJson(CONFIG_PATH) };
  } catch (err) {
    console.error(`config.json unreadable, using defaults: ${err.message}`);
  }
  // A config pointing at a deleted model would otherwise open to an empty window.
  const models = discoverModels();
  if (models.length && !models.some((m) => m.rel === config.model)) {
    console.error(`config model "${config.model}" not found — using ${models[0].rel}`);
    config.model = models[0].rel;
  }
}

function saveConfig() {
  try {
    // Round-trips cleanly: the "_comment" keys in config.json are ordinary string
    // values, so they survive parse/stringify untouched.
    const onDisk = readJson(CONFIG_PATH);
    // Only the keys the UI owns — everything else the user wrote is left alone.
    onDisk.model = config.model;
    if (config.subtitle) onDisk.subtitle = config.subtitle;
    fs.writeFileSync(CONFIG_PATH, JSON.stringify(onDisk, null, 2) + "\n", "utf-8");
  } catch (err) {
    console.error(`could not save config.json: ${err.message}`);
  }
}

/** Every folder under assets/models that contains a .model3.json. */
function discoverModels() {
  const dir = path.join(__dirname, "..", "assets", "models");
  const found = [];
  try {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      if (!entry.isDirectory()) continue;
      const settings = fs
        .readdirSync(path.join(dir, entry.name))
        .find((n) => n.endsWith(".model3.json"));
      if (settings) found.push({ name: entry.name, rel: `${entry.name}/${settings}` });
    }
  } catch (err) {
    console.error(`could not scan models directory: ${err.message}`);
  }
  return found;
}

const DEFAULT_SUBTITLE = {
  enabled: true,
  font_size: 13,
  position: "bottom",
  align: "center",
  offset_x: 0,
  offset_y: 0,
  max_lines: 4,
  max_chars: 40, // line length cap, keeps the caption near the character's own width
};

/** Whether a sample caption is pinned on screen. Session-only, not persisted —
 *  it is an adjustment aid, and leaving it on across restarts would confuse. */
let sampleShown = false;

function setSample(on) {
  sampleShown = on;
  win?.webContents.send("subtitle-sample", on);
  tray?.setContextMenu(buildMenu());
}

/** Change caption styling, push it to the renderer, and remember it. */
function updateSubtitle(patch) {
  config.subtitle = { ...DEFAULT_SUBTITLE, ...(config.subtitle ?? {}), ...patch };
  // Clamp so a stuck keypress cannot make the caption unreadable or invisible.
  config.subtitle.font_size = Math.min(48, Math.max(8, config.subtitle.font_size));
  config.subtitle.max_chars = Math.min(120, Math.max(12, config.subtitle.max_chars));
  saveConfig();
  win?.webContents.send("subtitle-style", config.subtitle);
  tray?.setContextMenu(buildMenu());
}

/** Hot-swap: the renderer replaces the model in place, no restart. */
function swapModel(rel) {
  if (rel === config.model) return;
  config.model = rel;
  saveConfig();
  win?.webContents.send("swap-model", rel);
  tray?.setContextMenu(buildMenu()); // refresh the radio ticks
}

function loadState() {
  try {
    // Through `readJson` for the reason written on it: a leading BOM makes JSON.parse
    // throw, and everything here is in a `catch` that treats a throw as "first run".
    // `loadConfig` was given the tolerant reader so a BOM could not silently reset the
    // chosen character; this file is written by the same kinds of tools and the same
    // silence loses where she is and how big she is instead.
    state = { ...state, ...readJson(statePath()) };
  } catch {
    // First run. Defaults are fine.
  }
}

function saveState() {
  if (!win) return;
  try {
    const [x, y] = win.getPosition();
    fs.writeFileSync(statePath(), JSON.stringify({ ...state, x, y }, null, 2));
  } catch (err) {
    console.error(`could not save window state: ${err.message}`);
  }
}

function sizeForScale(scale) {
  return {
    width: Math.round(config.width * scale),
    height: Math.round(config.height * scale),
  };
}

/** Bottom-right of the primary display, which is where she starts life. */
function defaultPosition(width, height) {
  const area = screen.getPrimaryDisplay().workArea;
  return {
    x: area.x + area.width - width - 20,
    y: area.y + area.height - height - 10,
  };
}

/**
 * Where to actually open her, given what a previous run saved.
 *
 * Restoring x/y unchecked is correct only while the displays are the ones that saved
 * them. Unplug a monitor, change resolution, or change DPI scaling and the saved rect
 * can land outside every display — and that is not a *visible* failure, because the
 * window is transparent, frameless, `skipTaskbar` and unfocusable. There is nothing to
 * see, nothing in the taskbar, and nothing to alt-tab to. It reads as "the character
 * stopped working" when the character is in fact loading perfectly, off the edge.
 *
 * The test is her feet, not the window. She is drawn anchored to the bottom centre of
 * her window (`renderer.js`, `model.anchor.set(0.5, 1.0)`), so a rect can overlap a
 * display generously and still show nothing but the empty air above her head. Measured
 * on a machine that had done exactly this: 681x1021 saved at (1210, 540) against a
 * 1707x1067 work area — 497x527 of window on screen, none of it character.
 */
function placeOnScreen(x, y, width, height) {
  if (x == null || y == null) return defaultPosition(width, height);
  const feet = { x: x + width / 2, y: y + height };
  const onADisplay = screen
    .getAllDisplays()
    .some(
      ({ workArea: a }) =>
        feet.x >= a.x && feet.x <= a.x + a.width &&
        feet.y >= a.y && feet.y <= a.y + a.height,
    );
  if (onADisplay) return { x, y };
  console.log(
    `saved position ${x},${y} puts her off every display — starting bottom-right instead`,
  );
  return defaultPosition(width, height);
}

/**
 * Shrink her to fit a screen smaller than the one the scale was chosen on.
 *
 * Same class of problem as the position: 1.7x of a 400x600 base is a 681x1021 window,
 * which is taller than the work area of a 1707x1067 display. Clamping the position
 * alone would then plant her feet on screen and push her head through the ceiling.
 */
function fitScale(scale) {
  const area = screen.getPrimaryDisplay().workArea;
  const fits = Math.min(area.width / config.width, area.height / config.height);
  return Math.max(MIN_SCALE, Math.min(MAX_SCALE, scale, fits));
}

function createWindow() {
  state.scale = fitScale(state.scale);
  const { width, height } = sizeForScale(state.scale);
  const { x, y } = placeOnScreen(state.x, state.y, width, height);

  win = new BrowserWindow({
    width, height, x, y,
    transparent: true,
    frame: false,
    alwaysOnTop: true,
    resizable: false, // resizing is done from the wheel, not a drag handle we cannot draw
    skipTaskbar: true,
    hasShadow: false,
    focusable: false,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  win.setAlwaysOnTop(true, "screen-saver");
  win.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });
  setClickThrough(true);

  // Renderer console goes to the terminal. Without this a failure inside renderer.js
  // is completely invisible — the window is transparent, so a crashed renderer and a
  // working one look identical.
  win.webContents.on("console-message", (_e, level, message, line, sourceId) => {
    const tag = ["debug", "info", "warn", "error"][level] ?? "log";
    const where = sourceId ? `${path.basename(sourceId)}:${line}` : "";
    console.log(`[renderer/${tag}] ${message}${where ? `  (${where})` : ""}`);
  });

  win.webContents.on("did-fail-load", (_e, code, desc, url) => {
    console.error(`[renderer] failed to load ${url}: ${desc} (${code})`);
  });

  win.webContents.on("render-process-gone", (_e, details) => {
    console.error("[renderer] process gone:", details.reason);
  });

  if (process.argv.includes("--debug")) {
    win.webContents.openDevTools({ mode: "detach" });
  }

  win.on("moved", saveState);
  win.loadFile(path.join(__dirname, "index.html"));
  // Write the corrected geometry back now rather than waiting for her to be moved.
  // Re-correcting an unusable saved position on every launch would work, but it also
  // means the file on disk keeps saying something false about where she is.
  saveState();
}

function setClickThrough(on) {
  if (!win) return;
  interactive = !on;
  // forward:true keeps mousemove flowing to the renderer while clicks pass through —
  // without it the renderer goes blind and can never ask to become interactive again.
  win.setIgnoreMouseEvents(on, { forward: true });
}

/**
 * How small the wheel may make her.
 *
 * Was 0.4, which on a 400x600 base is a 160x240 window — and the model inside is tall
 * and narrow, so the character within that is barely a hundred pixels wide. She was
 * reported as "way too small to the point when I almost can't see her", and the real
 * problem is worse than looking silly: the scroll wheel only works *over the character*,
 * so once she is that small there is nothing left to aim at and no way to scroll back.
 * The floor has to leave a target big enough to recover with.
 *
 * `Character > Size` in the tray and the panel's Look tab both set a scale directly, so
 * there is a way back that needs no aiming at all — but it should not be the only one.
 */
const MIN_SCALE = 0.6;
const MAX_SCALE = 2.5;

function applyScale(scale) {
  if (!win) return;
  state.scale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, scale));
  const { width, height } = sizeForScale(state.scale);
  const [x, y] = win.getPosition();
  const [oldW, oldH] = win.getSize();
  // Grow from the bottom centre so she stays planted where she is instead of
  // drifting up and to the left as she gets bigger.
  //
  // Deliberately *not* run through `placeOnScreen`. It looks like it belongs here —
  // scaling moves her, and near an edge it could in principle move her off — but the
  // arithmetic already preserves the bottom edge exactly (y + oldH - height + height
  // is y + oldH), so her feet do not move and there is nothing to correct. Adding the
  // check anyway made it fire during ordinary scrolling and snap her back to the
  // corner mid-gesture, which is a worse failure than the one it was guarding against.
  // The restore path is where displays actually change underneath her.
  win.setBounds({
    x: Math.round(x + (oldW - width) / 2),
    y: Math.round(y + (oldH - height)),
    width,
    height,
  });
  saveState();
}

/** 16x16 tray icon drawn in memory — avoids shipping a binary asset for one dot. */
function trayIcon() {
  const size = 16;
  const buf = Buffer.alloc(size * size * 4);
  const c = (size - 1) / 2;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const d = Math.hypot(x - c, y - c);
      // Soft edge so it does not look jagged at 16px.
      const alpha = Math.round(255 * Math.max(0, Math.min(1, (c - d + 0.5) / 1.5)));
      const i = (y * size + x) * 4;
      buf[i] = 220;     // B
      buf[i + 1] = 120; // G
      buf[i + 2] = 150; // R
      buf[i + 3] = alpha;
    }
  }
  // createFromBitmap, not createFromBuffer — the latter expects an encoded PNG/JPEG
  // and silently yields an empty image for raw pixels, which gives an invisible tray
  // icon and therefore no way to quit.
  return nativeImage.createFromBitmap(buf, { width: size, height: size });
}

function buildMenu() {
  if (PANEL_ONLY) {
    return Menu.buildFromTemplate([
      { label: "Aria — voice only", enabled: false },
      { type: "separator" },
      { label: "Control panel…", accelerator: "CommandOrControl+Shift+A", click: openPanel },
      { type: "separator" },
      { label: `Quit  (${QUIT_SHORTCUT.replace("CommandOrControl", "Ctrl")})`, click: () => app.quit() },
    ]);
  }
  return Menu.buildFromTemplate([
    { label: "Aria", enabled: false },
    { type: "separator" },
    {
      label: "Character",
      submenu: discoverModels().map((m) => ({
        label: m.name,
        type: "radio",
        checked: m.rel === config.model,
        click: () => swapModel(m.rel),
      })),
    },
    {
      // Fires an emotion straight at the renderer, bypassing core — the way to see
      // what each one looks like without having to coax the LLM into feeling it.
      label: "Expression",
      submenu: currentEmotions.length
        ? currentEmotions.map((name) => ({
            label: name,
            click: () => win?.webContents.send("play-emotion", name),
          }))
        : [{ label: "(none — character still loading)", enabled: false }],
    },
    {
      label: "Size",
      submenu: [
        { label: "Small", click: () => applyScale(0.7) },
        { label: "Medium", click: () => applyScale(1.0) },
        { label: "Large", click: () => applyScale(1.5) },
        { label: "Huge", click: () => applyScale(2.0) },
      ],
    },
    // Both resets exist for the same reason: whatever has gone wrong with where she is
    // or how big she is, the way back must not require aiming at her.
    { label: "Reset size", click: () => applyScale(1.0) },
    { label: "Reset position", click: resetPosition },
    {
      label: "Subtitle",
      submenu: (() => {
        const s = { ...DEFAULT_SUBTITLE, ...(config.subtitle ?? {}) };
        const nudge = 12;
        return [
          {
            label: "Show captions",
            type: "checkbox",
            checked: s.enabled,
            click: () => updateSubtitle({ enabled: !s.enabled }),
          },
          {
            // Stays up until switched off, so a run of adjustments can be judged
            // against something real instead of a preview that keeps vanishing.
            label: "Pin sample caption",
            type: "checkbox",
            checked: sampleShown,
            enabled: s.enabled,
            click: () => setSample(!sampleShown),
          },
          { type: "separator" },
          { label: `Text size: ${s.font_size}px`, enabled: false },
          { label: "Bigger", click: () => updateSubtitle({ font_size: s.font_size + 2 }) },
          { label: "Smaller", click: () => updateSubtitle({ font_size: s.font_size - 2 }) },
          { type: "separator" },
          { label: `Line width: ~${s.max_chars} characters`, enabled: false },
          { label: "Wider", click: () => updateSubtitle({ max_chars: s.max_chars + 5 }) },
          { label: "Narrower", click: () => updateSubtitle({ max_chars: s.max_chars - 5 }) },
          { type: "separator" },
          ...["top", "middle", "bottom"].map((position) => ({
            label: position[0].toUpperCase() + position.slice(1),
            type: "radio",
            checked: s.position === position,
            click: () => updateSubtitle({ position, offset_y: 0 }),
          })),
          { type: "separator" },
          ...["left", "center", "right"].map((align) => ({
            label: `Align ${align}`,
            type: "radio",
            checked: s.align === align,
            click: () => updateSubtitle({ align }),
          })),
          { type: "separator" },
          { label: "Nudge up", click: () => updateSubtitle({ offset_y: s.offset_y - nudge }) },
          { label: "Nudge down", click: () => updateSubtitle({ offset_y: s.offset_y + nudge }) },
          { label: "Nudge left", click: () => updateSubtitle({ offset_x: s.offset_x - nudge }) },
          { label: "Nudge right", click: () => updateSubtitle({ offset_x: s.offset_x + nudge }) },
          { type: "separator" },
          {
            label: "Reset captions",
            click: () => updateSubtitle({ ...DEFAULT_SUBTITLE, enabled: s.enabled }),
          },
        ];
      })(),
    },
    { type: "separator" },
    { label: "Control panel…", accelerator: "CommandOrControl+Shift+A", click: openPanel },
    { type: "separator" },
    { label: "Reload character", click: () => win?.webContents.reload() },
    { label: "Open devtools", click: () => win?.webContents.openDevTools({ mode: "detach" }) },
    { type: "separator" },
    { label: `Quit  (${QUIT_SHORTCUT.replace("CommandOrControl", "Ctrl")})`, click: () => app.quit() },
  ]);
}

/**
 * The control panel: an ordinary window, deliberately.
 *
 * The character overlay is transparent, frameless and click-through, which is right
 * for a character and wrong for anything with a scrollbar. Rather than fight those
 * properties, the panel is a second window with none of them, talking to core over
 * the same socket. It owns no state — core pushes a settings snapshot and the panel
 * draws it.
 */
let panel = null;

function openPanel() {
  if (panel && !panel.isDestroyed()) {
    panel.show();
    panel.focus();
    return;
  }
  panel = new BrowserWindow({
    width: 420,
    height: 560,
    minWidth: 340,
    minHeight: 400,
    title: "Aria",
    backgroundColor: "#16161a",
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, "panel-preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  panel.loadFile(path.join(__dirname, "panel.html"));
  panel.on("closed", () => {
    panel = null;
  });
}

function resetPosition() {
  if (!win) return;
  const { width, height } = sizeForScale(state.scale);
  const { x, y } = defaultPosition(width, height);
  win.setPosition(x, y);
  saveState();
}

function createTray() {
  tray = new Tray(trayIcon());
  tray.setToolTip("Aria — right-click for options");
  tray.setContextMenu(buildMenu());
  // Left-clicking a tray icon should do something obvious; showing the menu is the
  // least surprising choice when there is no window to raise.
  tray.on("click", () => tray.popUpContextMenu());
}

// --- IPC --------------------------------------------------------------------
ipcMain.on("set-interactive", (_e, wantInteractive) => {
  if (wantInteractive !== interactive) setClickThrough(!wantInteractive);
});

/**
 * Where the drag has reached, in fractional DIPs, held across the whole gesture.
 *
 * Re-reading `getPosition()` every event does not work: it returns integers while the
 * deltas are fractional, so a slow drag rounds the remainder away each time and she
 * lags behind the cursor.
 */
let dragAt = null;

ipcMain.on("drag-move", (_e, dx, dy) => {
  if (!win) return;
  if (!dragAt) {
    const [x, y] = win.getPosition();
    dragAt = { x, y };
  }
  dragAt.x += dx;
  dragAt.y += dy;

  // `setBounds` with the size pinned, rather than `setPosition`. This display runs at
  // 150%, and moving a window converts its size DIP -> physical -> DIP on every call.
  // That rounding is one-directional, and one drag is *hundreds* of mousemove events —
  // so she grew steadily while being dragged, and `layout()` refit the model to the
  // bigger window, which is what made the character grow rather than just the frame.
  // Recomputing from `state.scale` each time makes the size a pure function of what was
  // actually chosen, so there is nothing left to accumulate.
  const { width, height } = sizeForScale(state.scale);
  win.setBounds({ x: Math.round(dragAt.x), y: Math.round(dragAt.y), width, height });
});

ipcMain.on("drag-end", () => {
  dragAt = null;
  saveState();
});

ipcMain.on("scale-by", (_e, factor) => applyScale(state.scale * factor));

ipcMain.on("show-menu", () => buildMenu().popup({ window: win }));

ipcMain.on("quit", () => app.quit());

ipcMain.on("open-panel", openPanel);

// --- panel ↔ overlay settings ----------------------------------------------
// Everything here is overlay-local: core owns none of it. Each handler applies the
// change, persists it, and broadcasts the result so the tray menu, the character
// window and any open panel all end up agreeing.
const STRIP_POSITIONS = ["bottom-left", "bottom-right", "top-left", "top-right", "hidden"];

function overlaySettings() {
  return {
    subtitle: { ...DEFAULT_SUBTITLE, ...(config.subtitle ?? {}) },
    model: config.model,
    models: discoverModels(),
    strip: config.strip ?? "bottom-left",
    scale: state.scale,
    // The panel's Look tab is entirely about a character that isn't on screen. Told
    // rather than inferred, so the panel doesn't have to guess from an empty model list.
    panel_only: PANEL_ONLY,
  };
}

function broadcastOverlaySettings() {
  const data = overlaySettings();
  if (panel && !panel.isDestroyed()) panel.webContents.send("panel:changed", data);
}

ipcMain.handle("panel:get", () => overlaySettings());

ipcMain.on("panel:subtitle", (_e, patch) => {
  updateSubtitle(patch ?? {});
  broadcastOverlaySettings();
});

ipcMain.on("panel:model", (_e, rel) => {
  swapModel(rel);
  broadcastOverlaySettings();
});

ipcMain.on("panel:strip", (_e, position) => {
  if (!STRIP_POSITIONS.includes(position)) return;
  config.strip = position;
  saveConfig();
  win?.webContents.send("strip-position", position);
  broadcastOverlaySettings();
});

ipcMain.on("panel:scale", (_e, scale) => {
  applyScale(Number(scale) || 1);
  broadcastOverlaySettings();
});

ipcMain.on("get-config", (event) => {
  event.returnValue = config;
});

ipcMain.on("emotions", (_e, list) => {
  currentEmotions = Array.isArray(list) ? list : [];
  tray?.setContextMenu(buildMenu()); // the set changes with the character
});

// --- lifecycle --------------------------------------------------------------
app.whenReady().then(() => {
  loadConfig();
  loadState();
  if (PANEL_ONLY) {
    openPanel();
  } else {
    createWindow();
  }
  createTray();

  // Backstop for the tray: on some Windows setups the icon lands in the hidden
  // overflow area, and without this there would again be no way out.
  if (!globalShortcut.register(QUIT_SHORTCUT, () => app.quit())) {
    console.error(`could not register ${QUIT_SHORTCUT} — use the tray icon to quit`);
  }

  console.log(`Aria overlay running. Quit from the tray icon or ${QUIT_SHORTCUT}.`);

  // Dev aid: rotate through every installed character on a timer. Useful for checking
  // a newly added model swaps cleanly — the swap path frees GPU textures and rebuilds
  // parameter bindings, and neither failure is visible from a single load.
  // Dev aid / eyeball tool: step through every emotion the character has. Same thing
  // the tray's Expression submenu does, on a timer.
  if (process.argv.includes("--cycle-emotions")) {
    let n = 0;
    setInterval(() => {
      if (!currentEmotions.length) return;
      const name = currentEmotions[n++ % currentEmotions.length];
      console.log(`cycle emotion: ${name}`);
      win?.webContents.send("play-emotion", name);
    }, 2000);
  }

  if (process.argv.includes("--cycle-models")) {
    const models = discoverModels();
    let i = models.findIndex((m) => m.rel === config.model);
    console.log(`cycling ${models.length} models every 8s`);
    setInterval(() => {
      i = (i + 1) % models.length;
      swapModel(models[i].rel);
    }, 8000);
  }
});

app.on("will-quit", () => globalShortcut.unregisterAll());
app.on("window-all-closed", () => {
  // With a character on screen this never fires — that window is never closed. Without
  // one, closing the panel would otherwise take the whole app down and leave the voice
  // loop running with no way back to its controls. The tray is still there; reopen from
  // it, or Ctrl+Shift+A.
  if (!PANEL_ONLY) app.quit();
});
