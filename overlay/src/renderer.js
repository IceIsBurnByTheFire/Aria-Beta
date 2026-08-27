/**
 * The character: render, animate, react to core, and hot-swap between models.
 *
 * Two independent clocks run here, and keeping them separate is the point:
 *
 *   - Idle life (breathing, blinking, gaze) runs locally, always, whether or not core
 *     is connected. A character that freezes when the backend hiccups looks broken;
 *     one that keeps breathing looks like it is waiting for you.
 *   - Driven animation (mouth, expression, state) comes from core over WebSocket.
 *
 * Parameter layering is decided up front, because idle and driven animation both write
 * to overlapping Cubism parameters and whichever runs last silently wins:
 *
 *   mouth       → lip sync owns it outright
 *   eyes/brows  → expressions own them, idle blink yields while an expression holds
 *   everything  → idle owns the rest
 */

const { Live2DModel } = PIXI.live2d;

const CONFIG = window.aria?.config ?? {};
const CORE_URL = CONFIG.core ?? "ws://127.0.0.1:8765";
const modelUrl = (rel) => `../assets/models/${rel}`;

const stage = document.getElementById("stage");
const statusEl = document.getElementById("status");
const subtitleEl = document.getElementById("subtitle");
const watchingEl = document.getElementById("watching");

/** Filled in for anything config.json leaves out, so a partial block still works. */
const DEFAULT_SUBTITLE = {
  enabled: true,
  font_size: 13,
  position: "bottom",
  align: "center",
  offset_x: 0,
  offset_y: 0,
  max_lines: 4,
  //: Roughly this many characters per line. Expressed in `ch` units, which measure the
  //: width of "0" — in a proportional font that glyph is wider than average, so the
  //: real line fits a little more than the number says. Keeps the caption near the
  //: character's own width instead of spanning the whole window.
  max_chars: 40,
};

/**
 * Logical parameter -> the ids models actually use for it, best first.
 *
 * Two naming conventions are in the wild: `ParamMouthOpenY` (Cubism 3+) and
 * `PARAM_MOUTH_OPEN_Y` (carried over from Cubism 2). Writing to a parameter a model
 * does not have is a silent no-op, so a mismatched convention produces a character
 * that renders perfectly and never moves its mouth, with nothing logged anywhere.
 * Resolving against the model at load time makes that impossible.
 */
const PARAM_CANDIDATES = {
  mouthOpen: ["ParamMouthOpenY", "PARAM_MOUTH_OPEN_Y"],
  mouthForm: ["ParamMouthForm", "PARAM_MOUTH_FORM"],
  angleX: ["ParamAngleX", "PARAM_ANGLE_X"],
  angleY: ["ParamAngleY", "PARAM_ANGLE_Y"],
  angleZ: ["ParamAngleZ", "PARAM_ANGLE_Z"],
  eyeBallX: ["ParamEyeBallX", "PARAM_EYE_BALL_X"],
  eyeBallY: ["ParamEyeBallY", "PARAM_EYE_BALL_Y"],
  eyeLOpen: ["ParamEyeLOpen", "PARAM_EYE_L_OPEN"],
  eyeROpen: ["ParamEyeROpen", "PARAM_EYE_R_OPEN"],
  bodyAngleX: ["ParamBodyAngleX", "PARAM_BODY_ANGLE_X"],
  breath: ["ParamBreath", "PARAM_BREATH"],
};

/** Resolved ids for the currently loaded model. Rebuilt on every swap. */
let P = {};

/**
 * Semantic emotion -> what each character actually does about it.
 *
 * Core speaks emotions; only the overlay knows the assets, so the translation lives
 * here. Without this layer the LLM would have to choose between `f00`..`f07` — names
 * that carry no meaning — and would have nothing at all to say to a model like
 * `aimeng`, whose expressions are motion clips rather than expressions.
 *
 * haru's mapping was read off its expression parameters, not guessed: F05 closes the
 * eyes with EyeLSmile/EyeRSmile at 1, F07 sets ParamTere (照れ, the blush parameter),
 * F03 lowers the brows and turns the mouth down.
 */
const EMOTIONS = {
  haru: {
    neutral: { expression: "F01" },
    happy: { expression: "F05" },
    sad: { expression: "F04" },
    angry: { expression: "F03" },
    surprised: { expression: "F06" },
    shy: { expression: "F07" },
    thinking: { expression: "F08" },
  },
  tsubaki: {
    neutral: { expression: "neutral" },
    happy: { expression: "heart_eyes" },
    shy: { expression: "blush" },
    sad: { expression: "cry" },
    angry: { expression: "dark_face" },
    serious: { expression: "no_eye_highlight" },
  },
  // Motion-driven rather than expression-driven — this model ships no .exp3.json at
  // all. One motion per emotion, because a Cubism motion manager holds a single slot:
  // firing two at once just means the second replaces the first.
  aimeng: {
    neutral: { motion: ["Eyes", 6] },      // eyes_default
    happy: { motion: ["Eyes", 1] },        // eyes_star
    love: { motion: ["Eyes", 0] },         // eyes_heart
    surprised: { motion: ["Mouth", 1] },   // mouth_surprised
    shy: { motion: ["Face", 0] },          // blush_on
    smug: { motion: ["Face", 2] },         // sly_on
    sad: { motion: ["Eyes", 4] },          // eyes_bead
    thinking: { motion: ["Eyes", 3] },     // eyes_tense
  },
};

/** The emotion table for the loaded model, or {} if we have no mapping for it. */
let emotions = {};

const state = {
  aria: "idle",
  mouthTarget: 0,
  mouth: 0,
  gazeX: 0,
  gazeY: 0,
  pointer: { x: 0, y: 0 },
};

let app = null;
let model = null;
let currentModelPath = null;

async function main() {
  console.log("renderer starting");
  Live2DModel.registerTicker(PIXI.Ticker);

  app = new PIXI.Application({
    view: createCanvas(),
    resizeTo: window,
    backgroundAlpha: 0, // the desktop shows through
    antialias: true,
    autoDensity: true,
    resolution: window.devicePixelRatio || 1,
  });

  window.addEventListener("resize", layout);
  app.ticker.add(() => animate(app.ticker.deltaMS));

  await loadModel(CONFIG.model ?? "haru/Haru.model3.json");

  connect();
  trackPointer();
  wireStrip();
  applySubtitleStyle();
  window.aria?.onSwapModel?.((rel) => loadModel(rel));
  window.aria?.onPlayEmotion?.((name) => applyEmotion(name));
  window.aria?.onSubtitleStyle?.((s) => {
    applySubtitleStyle(s);
    previewSubtitle();
  });
  window.aria?.onSubtitleSample?.((on) => pinSample(on));
}

function createCanvas() {
  const canvas = document.createElement("canvas");
  stage.appendChild(canvas);
  return canvas;
}

// --- model loading ----------------------------------------------------------

/**
 * Load a model, replacing whatever is on stage. Safe to call at any time.
 *
 * The old model is only torn down once the new one has loaded, so a bad path leaves
 * the current character on screen instead of an empty window.
 */
async function loadModel(rel) {
  note(`load:${rel.split("/")[0]}`);
  setStatus(`loading ${rel.split("/")[0]}…`);
  let next;
  try {
    next = await Live2DModel.from(modelUrl(rel), { autoInteract: false });
  } catch (err) {
    console.error(`model failed to load: ${rel} — ${err.message}`);
    setStatus(`could not load ${rel.split("/")[0]}`, 4000);
    return false;
  }

  if (model) {
    app.stage.removeChild(model);
    // Textures here run to tens of megabytes — one of these models ships an 8192px
    // sheet. Without freeing them, swapping a few times exhausts GPU memory.
    model.destroy({ children: true, texture: true, baseTexture: true });
  }

  model = next;
  currentModelPath = rel;
  app.stage.addChild(model);
  layout();
  resolveParameters();

  const key = rel.split("/")[0];
  emotions = EMOTIONS[key] ?? {};
  if (!EMOTIONS[key]) {
    console.warn(
      `no emotion mapping for "${key}" — it will render and lip-sync, but core will ` +
        `be told it has no expressions. Add an entry to EMOTIONS in renderer.js.`,
    );
  }

  console.log(
    `model loaded: ${rel} (${model.internalModel.originalWidth}x` +
      `${model.internalModel.originalHeight})`,
  );
  console.log(`emotions: ${emotionNames().join(",") || "none"}`);
  await preloadAnimations();
  window.aria?.reportEmotions?.(emotionNames());

  // Capabilities changed, so core's idea of what it can ask for is now stale.
  sendHello();
  setStatus(rel.split("/")[0], 1500);
  return true;
}

function layout() {
  if (!model || !app) return;
  const scale = Math.min(
    app.renderer.width / model.internalModel.originalWidth,
    app.renderer.height / model.internalModel.originalHeight,
  );
  model.scale.set(scale);
  model.x = app.renderer.width / 2;
  model.y = app.renderer.height;
  model.anchor.set(0.5, 1.0); // stand her on the bottom edge
}

/**
 * Load every expression and motion file up front.
 *
 * pixi-live2d-display fetches and parses these lazily — `loadExpression` only does the
 * work `if (this.expressions[index] === null)`. That puts a fetch plus a parse on the
 * frame where an emotion is used for the *first* time, which lands precisely at the
 * start of a sentence and reads as a stutter. It fixes itself once every emotion has
 * been used once, which is exactly why it looks intermittent and is easy to misdiagnose.
 *
 * Doing it here costs a few hundred milliseconds at load, when nobody is watching.
 */
async function preloadAnimations() {
  const mm = model.internalModel.motionManager;
  const em = mm?.expressionManager;
  const jobs = [];

  for (let i = 0; i < (em?.definitions?.length ?? 0); i++) {
    jobs.push(em.loadExpression(i));
  }
  for (const [group, list] of Object.entries(mm?.definitions ?? {})) {
    for (let i = 0; i < (list?.length ?? 0); i++) jobs.push(mm.loadMotion(group, i));
  }
  if (!jobs.length) return;

  const t0 = performance.now();
  const settled = await Promise.allSettled(jobs);
  const failed = settled.filter((r) => r.status === "rejected").length;
  console.log(
    `preloaded ${settled.length - failed}/${settled.length} animations in ` +
      `${(performance.now() - t0).toFixed(0)}ms`,
  );
}

/** Pick the id each logical parameter actually maps to on this model. */
function resolveParameters() {
  const core = model.internalModel.coreModel;
  const has = (id) =>
    typeof core.getParameterIndex === "function" && core.getParameterIndex(id) >= 0;

  P = {};
  const missing = [];
  for (const [key, candidates] of Object.entries(PARAM_CANDIDATES)) {
    const found = candidates.find(has);
    if (found) P[key] = found;
    else missing.push(key);
  }
  if (missing.length) {
    console.warn(
      `model has no parameter for: ${missing.join(", ")} — ` +
        `that animation will do nothing. Add the id to PARAM_CANDIDATES.`,
    );
  }
}

/** Write a parameter only if this model has it. */
function set(core, key, value) {
  if (P[key]) core.setParameterValueById(P[key], value);
}

// --- animation --------------------------------------------------------------
let elapsed = 0;
let nextBlink = 2000;
let blinkPhase = -1;

/** Frame-hitch watchdog. Names the last thing that happened, so a stutter can be
 *  attributed instead of guessed at. Rare enough to leave on permanently. */
let lastEvent = "startup";
let lastEventAt = 0;
//: A dropped frame at 60fps is ~33ms. Lower while chasing a stutter.
const FRAME_WARN_MS = Number(CONFIG.frame_warn_ms ?? 28);

function note(what) {
  lastEvent = what;
  lastEventAt = performance.now();
}

function animate(deltaMS) {
  if (!model) return;
  if (deltaMS > FRAME_WARN_MS) {
    console.warn(
      `slow frame: ${deltaMS.toFixed(0)}ms — last event "${lastEvent}" ` +
        `${(performance.now() - lastEventAt).toFixed(0)}ms earlier`,
    );
  }
  elapsed += deltaMS;
  const core = model.internalModel.coreModel;
  const t = elapsed / 1000;

  // Mouth: ease toward the target so 50 Hz viseme frames render smoothly at 60+ fps.
  state.mouth += (state.mouthTarget - state.mouth) * Math.min(1, deltaMS / 60);
  set(core, "mouthOpen", state.mouth);

  // Breathing — a slow sine, faster and deeper while speaking.
  const breathRate = state.aria === "speaking" ? 0.9 : 0.55;
  set(core, "breath", (Math.sin(t * Math.PI * 2 * breathRate) + 1) / 2);

  const sway = Math.sin(t * 0.7) * 3;
  const lean = state.aria === "listening" ? 4 : 0;
  const tilt = state.aria === "thinking" ? Math.sin(t * 1.4) * 6 + 6 : 0;
  set(core, "bodyAngleX", sway * 0.6);
  set(core, "angleZ", tilt);

  // Gaze follows the cursor, eased. Head turns a little, eyes turn more.
  state.gazeX += (state.pointer.x - state.gazeX) * 0.06;
  state.gazeY += (state.pointer.y - state.gazeY) * 0.06;
  set(core, "angleX", state.gazeX * 22 + sway);
  set(core, "angleY", state.gazeY * 14 + lean);
  set(core, "eyeBallX", state.gazeX);
  set(core, "eyeBallY", state.gazeY);

  blink(core, deltaMS);
}

function blink(core, deltaMS) {
  nextBlink -= deltaMS;
  if (blinkPhase < 0 && nextBlink <= 0) {
    blinkPhase = 0;
    // Irregular on purpose — a metronome blink reads as uncanny.
    nextBlink = 2200 + Math.random() * 4000;
  }
  if (blinkPhase >= 0) {
    blinkPhase += deltaMS;
    const openness =
      blinkPhase < 60 ? 1 - blinkPhase / 60 : Math.min(1, (blinkPhase - 60) / 90);
    set(core, "eyeLOpen", openness);
    set(core, "eyeROpen", openness);
    if (blinkPhase > 150) blinkPhase = -1;
  }
}

// --- interaction ------------------------------------------------------------
let drag = null;

function overCharacter(e) {
  if (!model) return false;
  const b = model.getBounds();
  return (
    e.clientX >= b.x && e.clientX <= b.x + b.width &&
    e.clientY >= b.y && e.clientY <= b.y + b.height
  );
}

function trackPointer() {
  // forward:true on setIgnoreMouseEvents keeps these arriving even while clicks pass
  // through, which is what lets the character watch the cursor from a click-through
  // window.
  window.addEventListener("mousemove", (e) => {
    state.pointer.x = (e.clientX / window.innerWidth) * 2 - 1;
    state.pointer.y = -((e.clientY / window.innerHeight) * 2 - 1);

    if (drag) {
      // Screen-space deltas, not client-space: the window is moving out from under
      // the cursor, so client coordinates would fight the drag.
      window.aria.dragMove(e.screenX - drag.x, e.screenY - drag.y);
      drag.x = e.screenX;
      drag.y = e.screenY;
      return; // stay interactive for the whole drag, even if we outrun the bounds
    }
    // The strip has to count as "interactive" too, or the window stays click-through
    // over it and the buttons are decorative.
    window.aria?.setInteractive(overCharacter(e) || overStrip(e));
  });

  window.addEventListener("mousedown", (e) => {
    if (e.button === 0 && overCharacter(e)) drag = { x: e.screenX, y: e.screenY };
  });

  window.addEventListener("mouseup", () => {
    if (drag) {
      window.aria.dragEnd();
      drag = null;
    }
  });

  window.addEventListener(
    "wheel",
    (e) => {
      if (!overCharacter(e)) return;
      e.preventDefault();
      window.aria.scaleBy(e.deltaY < 0 ? 1.08 : 1 / 1.08);
    },
    { passive: false },
  );

  window.addEventListener("contextmenu", (e) => {
    e.preventDefault();
    window.aria.showMenu();
  });
}

// --- core connection --------------------------------------------------------
let ws = null;
let retry = 500;
let visemes = 0;

function connect() {
  ws = new WebSocket(CORE_URL);

  ws.onopen = () => {
    retry = 500;
    stage.classList.remove("offline");
    setStatus("connected", 1500);
    console.log(`connected to core at ${CORE_URL}`);
    sendHello();
  };

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    // State changes are ~3 per turn — cheap to log and the fastest way to tell
    // whether core is actually driving the character.
    if (msg.type === "state") console.log(`state: ${msg.value} (${visemes} visemes)`);
    if (msg.type === "viseme") visemes++;
    handle(msg);
  };

  ws.onclose = () => {
    state.mouthTarget = 0;
    stage.classList.add("offline");
    setStatus("core offline — reconnecting");
    // Backoff, capped. She keeps breathing throughout; only the tint changes.
    setTimeout(connect, retry);
    retry = Math.min(retry * 1.6, 8000);
  };

  ws.onerror = () => ws.close();
}

/** Is the cursor over the control strip? */
function overStrip(e) {
  const strip = document.getElementById("strip");
  if (!strip) return false;
  const r = strip.getBoundingClientRect();
  return (
    e.clientX >= r.left && e.clientX <= r.right &&
    e.clientY >= r.top && e.clientY <= r.bottom
  );
}

/**
 * The strip's three buttons.
 *
 * Mute is the only one that renders state, and it renders what *core* last said —
 * `settings` messages arrive here as well as at the panel, so the two can't drift
 * apart. Muting from the panel lights the strip's button and vice versa.
 */
function wireStrip() {
  const send = (name) => {
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "command", name }));
    }
  };
  const mic = document.getElementById("s-mic");
  mic.onclick = () => send(muted ? "unmute" : "mute");
  document.getElementById("s-stop").onclick = () => send("stop");
  document.getElementById("s-panel").onclick = () => window.aria?.openPanel();

  setStripPosition(window.aria?.config?.strip ?? "bottom-left");
  window.aria?.onStripPosition?.(setStripPosition);
}

/** Corner placement, driven from the panel. `hidden` removes it entirely. */
function setStripPosition(position) {
  const strip = document.getElementById("strip");
  if (!strip) return;
  strip.className = position || "bottom-left";
}

let muted = false;

function applySettings(msg) {
  muted = Boolean(msg.muted);
  const mic = document.getElementById("s-mic");
  if (mic) {
    mic.classList.toggle("on", muted);
    mic.title = muted ? "Unmute microphone" : "Mute microphone";
  }
}

function sendHello() {
  if (ws?.readyState !== WebSocket.OPEN || !model) return;
  ws.send(
    JSON.stringify({
      type: "hello",
      model_format: "live2d",
      model_name: (currentModelPath ?? "").split("/")[0],
      // The semantic set is what core puts in the system prompt. Raw names go too,
      // for diagnostics — they are meaningless to an LLM (`f00`..`f07`) and differ
      // per character, which is exactly why the mapping exists.
      emotions: emotionNames(),
      expressions: expressionNames(),
      motions: motionNames(),
    }),
  );
}

function handle(msg) {
  if (msg.type !== "viseme") note(msg.type + (msg.value ? `:${msg.value}` : ""));
  switch (msg.type) {
    case "state":
      state.aria = msg.value;
      setStatus(msg.value, msg.value === "idle" ? 1200 : 0);
      break;
    case "viseme":
      state.mouthTarget = msg.open;
      break;
    case "expression":
      applyEmotion(msg.value);
      break;
    case "motion":
      if (model) model.motion(msg.name);
      break;
    case "subtitle":
      showSubtitle(msg.text, msg.final);
      break;
    case "vision":
      // Persistent, never auto-hidden: the user must be able to tell whether Aria
      // can see their screen without remembering what they last said to her.
      watchingEl.classList.toggle("on", Boolean(msg.watching));
      watchingEl.classList.toggle("capturing", Boolean(msg.capturing));
      console.log(`vision: watching=${msg.watching} capturing=${msg.capturing}`);
      break;
    case "notice":
      setStatus(msg.text, 2500);
      break;
    case "settings":
      // The same snapshot the panel renders. Both windows listen, so the strip's
      // mute button and the panel's can never disagree about whether the mic is on.
      applySettings(msg);
      break;
  }
}

/**
 * Show a semantic emotion using whatever this character has for it.
 *
 * Expressions cross-fade on their own — pixi-live2d-display defaults to a 1 s fade
 * when an .exp3.json declares none — so nothing here has to blend by hand.
 */
function applyEmotion(name) {
  if (!model) return;
  const spec = emotions[name];
  if (!spec) {
    console.warn(`no mapping for emotion "${name}" on this character`);
    return;
  }
  console.log(`emotion: ${name}`);
  if (spec.expression) {
    model.expression(spec.expression);
  }
  if (spec.motion) {
    const [group, index] = spec.motion;
    // FORCE, or a currently-playing clip would swallow the request.
    model.motion(group, index, PIXI.live2d.MotionPriority?.FORCE ?? 3);
  }
}

/** Groups core should never be offered — see `exclude_from_ai` in config.json. */
function excluded(name) {
  return (CONFIG.exclude_from_ai ?? []).some((x) =>
    name.toLowerCase().includes(x.toLowerCase()),
  );
}

function emotionNames() {
  return Object.keys(emotions).filter((n) => !excluded(n));
}

function expressionNames() {
  const defs = model?.internalModel?.settings?.expressions ?? [];
  return defs.map((e) => e.Name ?? e.name).filter((n) => n && !excluded(n));
}

function motionNames() {
  const groups = model?.internalModel?.settings?.motions ?? {};
  return Object.keys(groups).filter((n) => !excluded(n));
}

// --- ui ---------------------------------------------------------------------
let statusTimer = null;

function setStatus(text, hideAfter = 0) {
  statusEl.textContent = text;
  statusEl.style.opacity = "1";
  clearTimeout(statusTimer);
  if (hideAfter) {
    statusTimer = setTimeout(() => (statusEl.style.opacity = "0"), hideAfter);
  }
}

let subtitleTimer = null;
let subtitleStyle = { ...DEFAULT_SUBTITLE, ...(CONFIG.subtitle ?? {}) };

/**
 * Position and size the caption from config.
 *
 * Driven from JS rather than a stylesheet because the tray adjusts it live and the
 * result is written back to config.json — one source of truth beats a CSS rule and a
 * config key that can disagree.
 */
function applySubtitleStyle(s = subtitleStyle) {
  subtitleStyle = { ...DEFAULT_SUBTITLE, ...s };
  const { font_size, position, align, offset_x, offset_y, max_lines, max_chars } =
    subtitleStyle;
  const el = subtitleEl;
  const margin = 10;

  el.style.fontSize = `${font_size}px`;
  el.style.textAlign = align;
  el.style.left = `${margin + offset_x}px`;
  el.style.right = `${margin - offset_x}px`;
  el.style.maxHeight = `${max_lines * 1.45}em`;

  // Cap the line length, then let `align` decide where the narrowed box sits within
  // the window — aligning the text left inside a full-width box would leave it
  // stranded far from the character.
  el.style.maxWidth = `${max_chars}ch`;
  el.style.marginLeft = align === "left" ? "0" : "auto";
  el.style.marginRight = align === "right" ? "0" : "auto";

  // Only one of top/bottom may be set, or the box stretches between them.
  el.style.top = "auto";
  el.style.bottom = "auto";
  el.style.transform = "none";
  if (position === "top") {
    el.style.top = `${margin + offset_y}px`;
  } else if (position === "middle") {
    el.style.top = "50%";
    el.style.transform = `translateY(calc(-50% + ${offset_y}px))`;
  } else {
    el.style.bottom = `${24 - offset_y}px`; // y is positive downward everywhere
  }

  console.log(
    `subtitle: ${subtitleStyle.enabled ? "on" : "off"} ${font_size}px ${position}/${align}` +
      ` offset ${offset_x},${offset_y} max ${max_chars}ch x ${max_lines} lines`,
  );
}

//: Long enough to wrap, so width, line spacing and max_lines can all be judged.
const SAMPLE_TEXT =
  "This is where captions appear while Aria is speaking. Longer replies wrap " +
  "onto several lines like this, so you can see the width and spacing too.";

//: How long an unpinned sample lingers after a styling change. Generous, and reset on
//: every change, so a run of small adjustments keeps it on screen throughout.
const PREVIEW_MS = 12000;

/** True while the sample is pinned from the tray and should not auto-hide. */
let samplePinned = false;

function pinSample(on) {
  samplePinned = on;
  console.log(`sample caption ${on ? "pinned" : "unpinned"}`);
  if (on) showSample();
  else hideSubtitle();
}

function showSample() {
  clearTimeout(subtitleTimer);
  // The sample obeys the on/off switch too — a sample caption showing while captions
  // are disabled would just be confusing.
  if (!subtitleStyle.enabled) {
    subtitleEl.classList.remove("visible");
    return;
  }
  subtitleEl.textContent = SAMPLE_TEXT;
  subtitleEl.classList.add("visible");
}

function hideSubtitle() {
  clearTimeout(subtitleTimer);
  // Pinned means it stays visible between utterances, not that speech never replaces it.
  if (samplePinned) showSample();
  else subtitleEl.classList.remove("visible");
}

/** Show a sample caption, so a styling change is visible without making her talk. */
function previewSubtitle() {
  if (samplePinned) {
    showSample();
    return;
  }
  showSample();
  subtitleTimer = setTimeout(hideSubtitle, PREVIEW_MS);
}

function showSubtitle(text, final) {
  if (!subtitleStyle.enabled) {
    subtitleEl.classList.remove("visible");
    return;
  }
  subtitleEl.textContent = text || "";
  subtitleEl.classList.toggle("visible", Boolean(text));
  clearTimeout(subtitleTimer);
  if (final && text) {
    subtitleTimer = setTimeout(hideSubtitle, 4000);
  }
}

main();
