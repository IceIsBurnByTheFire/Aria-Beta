/**
 * The control panel.
 *
 * It talks to core over the same WebSocket the character uses, because the seam is
 * already there and adding a second one would mean two things to keep in sync. It
 * connects as a second client rather than routing through the character window —
 * core broadcasts to every client, so both stay current on their own.
 *
 * The panel renders state, it never decides it. Every toggle sends a command and then
 * waits to be told what happened; nothing is drawn optimistically. A switch that flips
 * instantly and then silently disagrees with the thing it controls is worse than one
 * that takes a moment — screen watching in particular can *refuse*, because arming it
 * runs a preflight that fails when the vision backend is out of credit.
 */

const CORE = "ws://127.0.0.1:8765";
const state = { ws: null, settings: null };

const $ = (id) => document.getElementById(id);
const send = (name, value) => {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ type: "command", name, value }));
  }
};

/* ---- tabs ---------------------------------------------------------------- */
document.querySelectorAll("nav button").forEach((tab) => {
  tab.onclick = () => {
    document.querySelectorAll("nav button").forEach((b) => b.classList.remove("on"));
    document.querySelectorAll("section").forEach((s) => s.classList.remove("on"));
    tab.classList.add("on");
    $(tab.dataset.tab).classList.add("on");
  };
});

/* ---- controls ------------------------------------------------------------ */
const toggle = (el, command, read) => {
  el.onclick = () => {
    // Send the opposite of what core last told us, never the opposite of what is
    // drawn — those diverge the moment a command is refused.
    const now = state.settings ? read(state.settings) : false;
    send(command, !now);
  };
};

$("mic").onclick = () => send(state.settings?.muted ? "unmute" : "mute");
toggle($("watch"), "set_watching", (s) => s.watching);
toggle($("wake"), "set_wake", (s) => s.wake_enabled);
toggle($("barge"), "set_barge_in", (s) => s.barge_in);
toggle($("emo"), "set_emotion_voice", (s) => s.emotion_voice);
$("stop").onclick = () => send("stop");
$("voice").onchange = (e) => send("set_voice", e.target.value);

/* ---- rendering ----------------------------------------------------------- */
function setToggle(el, on) {
  el.setAttribute("aria-pressed", on ? "true" : "false");
}

function renderSettings(s) {
  state.settings = s;
  setToggle($("mic"), !s.muted); // shown as "microphone on", not "muted"
  setToggle($("watch"), s.watching);
  setToggle($("wake"), s.wake_enabled);
  setToggle($("barge"), s.barge_in);
  setToggle($("emo"), s.emotion_voice);
  $("wakeword").textContent = s.wake_enabled ? `"${s.wake_word}"` : "";

  const sel = $("voice");
  if (sel.options.length !== (s.voices || []).length) {
    sel.innerHTML = "";
    (s.voices || []).forEach((v) => sel.add(new Option(v, v)));
  }
  sel.value = s.voice;

  renderNotes(s.memory || []);
  renderPersona(s);

  $("status").textContent =
    `${s.sessions} session${s.sessions === 1 ? "" : "s"} · ` +
    `${(s.memory || []).length} notes · vision: ${s.vision_backend}` +
    (s.speaker_mode ? " · speaker mode" : "");
}

/* ---- memory ---------------------------------------------------------------
 * Notes are edited in place, and addressed by id rather than by text. `forget` matches
 * loosely on purpose so it works said out loud; a button next to one specific line has
 * to take that line and nothing else.
 */
function renderNotes(list) {
  const notes = $("notes");

  // Core pushes a fresh snapshot after every command from anywhere — her writing a note,
  // a voice command, a second panel — so a naive rebuild would swallow a half-typed
  // correction and move the caret.
  //
  // The first version skipped the whole rebuild whenever anything inside the list had
  // focus, and driving the real panel showed why that is wrong: click one note and the
  // list stops updating *for good*, because focus survives clicking away. It sat there
  // showing two notes from a core that had been restarted, while the footer counted one.
  // So: carry the one note being edited across the rebuild, and let everything else
  // stay live.
  const active = document.activeElement;
  const editing =
    notes.contains(active) && active.tagName === "INPUT"
      ? { id: active.dataset.id, value: active.value, caret: active.selectionStart }
      : null;

  if (!list.length) {
    notes.innerHTML = '<div class="empty">She hasn\'t kept any notes yet.</div>';
    return;
  }
  notes.innerHTML = "";
  list.forEach((n) => {
    const row = document.createElement("div");
    row.className = "note";

    const text = document.createElement("input");
    text.value = n.text;
    text.dataset.id = n.id;
    text.title = "Edit, then press Enter";
    const commit = () => {
      const next = text.value.trim();
      if (!next || next === n.text) {
        text.value = n.text; // nothing to do, and don't leave it looking edited
        return;
      }
      send("edit_note", { id: n.id, text: next });
    };
    text.onkeydown = (e) => {
      if (e.key === "Enter") text.blur();
      if (e.key === "Escape") { text.value = n.text; text.blur(); }
    };
    text.onblur = commit;

    const src = document.createElement("span");
    src.className = "src";
    // Which notes she was told and which she inferred is the thing you need when
    // deciding whether to trust one.
    src.textContent = n.source === "you" ? "you told her" : "she noticed";

    const del = document.createElement("button");
    del.textContent = "×";
    del.title = "Forget this";
    del.onclick = () => send("delete_note", n.id);

    row.append(text, src, del);
    notes.append(row);
  });

  // Put the caret back where it was, in the same note, if that note still exists.
  if (editing && editing.id) {
    const again = notes.querySelector(`input[data-id="${editing.id}"]`);
    if (again) {
      again.value = editing.value;
      again.focus();
      again.setSelectionRange(editing.caret, editing.caret);
    }
  }
}

const addNote = () => {
  const box = $("note-new");
  const text = box.value.trim();
  if (!text) return;
  send("add_note", text);
  box.value = "";
};
$("note-add").onclick = addNote;
$("note-new").onkeydown = (e) => { if (e.key === "Enter") addNote(); };

/* ---- persona --------------------------------------------------------------
 * The only editable part of the prompt. Everything else — screen rules, emotion
 * vocabulary, her notes, the clock — is assembled per turn and shown read-only below.
 */
let personaSaved = null;   // what core last told us is in force
let resetArmed = false;

function personaDirty() {
  return personaSaved !== null && $("persona-text").value !== personaSaved;
}

function renderPersona(s) {
  const box = $("persona-text");
  // Same rule as the notes, and it matters more here: this is a long piece of writing,
  // and every snapshot would otherwise throw away whatever had been typed since the
  // last save. Only adopt core's copy when there is nothing to lose.
  //
  // A refused save lands here too, and lands correctly: core did not change its copy,
  // so the box still counts as dirty and his text survives for him to fix.
  if (!personaDirty()) box.value = s.persona || "";
  personaSaved = s.persona || "";
  $("prompt-full").value = s.system_prompt || "";
  paintPersonaState(s.persona_is_custom);
}

function paintPersonaState(isCustom) {
  const state = $("persona-state");
  const dirty = personaDirty();
  $("persona-save").disabled = !dirty;
  state.classList.toggle("dirty", dirty);
  state.textContent = dirty
    ? "unsaved changes"
    : isCustom
      ? "edited — saved on disk"
      : "the built-in persona";
}

$("persona-text").oninput = () => paintPersonaState(state.settings?.persona_is_custom);
$("persona-save").onclick = () => send("set_persona", $("persona-text").value);

$("persona-reset").onclick = () => {
  // Two-step rather than a dialog. Reset deletes the edited persona for good, so a
  // single mis-click should not be able to do it — but a modal for a two-second action
  // is worse than a button that asks once.
  const btn = $("persona-reset");
  if (!resetArmed) {
    resetArmed = true;
    btn.textContent = "Really? This deletes your version";
    setTimeout(() => {
      resetArmed = false;
      btn.textContent = "Reset to built-in";
    }, 4000);
    return;
  }
  resetArmed = false;
  btn.textContent = "Reset to built-in";
  personaSaved = null;              // let the next snapshot overwrite the box
  send("reset_persona");
};

function addLine(role, text, at) {
  const log = $("log");
  const empty = log.querySelector(".empty");
  if (empty) empty.remove();
  const row = document.createElement("div");
  row.className = role === "aria" ? "aria" : "you";
  const when = document.createElement("time");
  when.textContent = new Date((at || Date.now() / 1000) * 1000)
    .toTimeString()
    .slice(0, 5);
  const who = document.createElement("span");
  who.className = "who";
  who.textContent = role === "aria" ? "aria " : "you ";
  row.append(when, who, document.createTextNode(text));
  log.append(row);
  // Only follow the tail if you were already at it — otherwise reading back through
  // the conversation yanks you to the bottom every time she speaks.
  const main = document.querySelector("main");
  if (main.scrollHeight - main.scrollTop - main.clientHeight < 60) {
    main.scrollTop = main.scrollHeight;
  }
}

/* ---- overlay-local settings ----------------------------------------------
 * A second channel, because these have a different owner. Core knows nothing about
 * subtitles or where the strip sits — that state lives in the overlay process and is
 * written to config.json. Routing it through core would make core the authority on
 * things it cannot see.
 */
function renderOverlay(o) {
  if (!o) return;
  if (o.panel_only) {
    // Voice-only run: every control on this tab adjusts a character that isn't on
    // screen. Hiding it beats leaving four dropdowns that appear to do nothing.
    $("look").innerHTML =
      '<div class="empty">Running without the character.<br />' +
      "Appearance settings need her window.</div>";
    document.querySelector('nav button[data-tab="look"]')?.remove();
    return;
  }
  const models = $("model");
  if (models.options.length !== (o.models || []).length) {
    models.innerHTML = "";
    (o.models || []).forEach((m) => models.add(new Option(m.name, m.rel)));
  }
  models.value = o.model;
  $("strip").value = o.strip || "bottom-left";
  // Nearest preset rather than exact — the scroll wheel scales continuously, so an
  // exact match would leave the dropdown blank most of the time.
  const presets = [0.7, 1, 1.5, 2];
  const near = presets.reduce((a, b) =>
    Math.abs(b - o.scale) < Math.abs(a - o.scale) ? b : a
  );
  $("scale").value = String(near);

  const s = o.subtitle || {};
  setToggle($("sub-on"), s.enabled);
  $("sub-pos").value = s.position || "middle";
  $("sub-size").value = s.font_size ?? 17;
  $("sub-size-val").textContent = `${s.font_size ?? 17}px`;
  $("sub-chars").value = s.max_chars ?? 40;
  $("sub-chars-val").textContent = `${s.max_chars ?? 40} chars`;
  overlayState = o;
}

let overlayState = null;

if (window.overlay) {
  window.overlay.get().then(renderOverlay);
  window.overlay.onChanged(renderOverlay);

  $("model").onchange = (e) => window.overlay.setModel(e.target.value);
  $("strip").onchange = (e) => window.overlay.setStrip(e.target.value);
  $("scale").onchange = (e) => window.overlay.setScale(Number(e.target.value));
  $("sub-on").onclick = () =>
    window.overlay.setSubtitle({ enabled: !overlayState?.subtitle?.enabled });
  $("sub-pos").onchange = (e) =>
    window.overlay.setSubtitle({ position: e.target.value });
  // `input` not `change`, so dragging the slider shows the caption resizing live.
  $("sub-size").oninput = (e) => {
    $("sub-size-val").textContent = `${e.target.value}px`;
    window.overlay.setSubtitle({ font_size: Number(e.target.value) });
  };
  $("sub-chars").oninput = (e) => {
    $("sub-chars-val").textContent = `${e.target.value} chars`;
    window.overlay.setSubtitle({ max_chars: Number(e.target.value) });
  };
} else {
  // Opened outside Electron (a browser, a test). Core settings still work.
  $("look").innerHTML =
    '<div class="empty">Appearance settings need the overlay app.</div>';
}

/* ---- feedback -------------------------------------------------------------
 * A save that silently does nothing is the failure this panel exists to avoid, and it
 * is the likeliest one here: the persona and note limits both refuse quietly at the
 * core end. The footer is already a status line, so it says what happened and then goes
 * back to being one.
 */
let flashTimer = null;
function flash(text, level) {
  if (!text) return;
  const bar = $("status");
  bar.textContent = text;
  bar.style.color = level === "warn" || level === "error" ? "#ffd479" : "";
  clearTimeout(flashTimer);
  flashTimer = setTimeout(() => {
    bar.style.color = "";
    if (state.settings) renderSettings(state.settings); // restores the usual summary
  }, 4000);
}

/* ---- connection ---------------------------------------------------------- */
function connect() {
  const ws = new WebSocket(CORE);
  state.ws = ws;

  ws.onopen = () => {
    $("status").textContent = "connected";
    // Core pushes a snapshot on hello, so announcing ourselves is also how we ask.
    // `role: panel` matters. Core reads the first hello it sees to learn which
    // expressions the character supports; without this flag an empty list from the
    // panel reads as "this model can't emote" and her whole range disappears from
    // the system prompt.
    ws.send(JSON.stringify({ type: "hello", role: "panel", model_name: "panel" }));
  };
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === "settings") renderSettings(msg);
    else if (msg.type === "transcript") addLine(msg.role, msg.text, msg.at);
    else if (msg.type === "notice") flash(msg.text, msg.level);
  };
  ws.onclose = () => {
    $("status").textContent = "core not running — retrying";
    setTimeout(connect, 1500);
  };
  ws.onerror = () => ws.close();
}

connect();
