/**
 * Tell me whether a Live2D model will load in Aria.
 *
 * The question people ask is "is this Cubism 4 or 5". That is the wrong question. What
 * decides compatibility is the *moc format*:
 *
 *   .moc3 → loads. Every moc3 the editor has ever produced is accepted by a newer
 *           runtime, so a Cubism 3.0 model and a Cubism 5.0 model both work.
 *   .moc  → does not load. That is Cubism 2.x, a different runtime this project
 *           deliberately does not ship.
 *
 * The only real failure is a moc3 *newer* than the runtime, which needs a Cubism
 * version that did not exist when the bundled core was built.
 *
 * Usage:
 *   node scripts/check-model.mjs              # every model under assets/models
 *   node scripts/check-model.mjs haru         # just one
 */

import { readdir, readFile, stat } from "node:fs/promises";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const MODELS = resolve(HERE, "..", "assets", "models");

/** Straight from Live2DCubismCore's own MocVersion_* constants. */
const MOC_VERSIONS = {
  1: "Cubism 3.0",
  2: "Cubism 3.3",
  3: "Cubism 4.0",
  4: "Cubism 4.2",
  5: "Cubism 5.0",
};

/** Highest moc3 version the bundled Cubism Core 5.1 will accept. */
const RUNTIME_MAX = 5;

async function walk(dir, found = []) {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) await walk(full, found);
    else found.push(full);
  }
  return found;
}

async function mocVersion(path) {
  const head = Buffer.alloc(5);
  const fh = await (await import("node:fs/promises")).open(path, "r");
  try {
    await fh.read(head, 0, 5, 0);
  } finally {
    await fh.close();
  }
  if (head.subarray(0, 4).toString("ascii") !== "MOC3") return null;
  return head[4];
}

async function inspect(name) {
  const dir = join(MODELS, name);
  const files = await walk(dir);
  const rel = (p) => relative(dir, p).replace(/\\/g, "/");

  const moc3 = files.find((f) => f.endsWith(".moc3"));
  const moc2 = files.find((f) => f.endsWith(".moc"));
  const settings = files.find((f) => f.endsWith(".model3.json"));

  console.log(`\n${name}`);

  if (!moc3 && moc2) {
    console.log(`  ✗ Cubism 2.x  (${rel(moc2)})`);
    console.log("    Will not load. Needs the legacy runtime this project does not ship.");
    console.log("    Look for a Cubism 3+ version of the model, or a different model.");
    return false;
  }

  if (!moc3) {
    console.log("  ✗ no .moc3 found — this does not look like a Live2D model folder");
    return false;
  }

  const version = await mocVersion(moc3);
  if (version === null) {
    console.log(`  ✗ ${rel(moc3)} is not a valid moc3 (bad magic bytes)`);
    return false;
  }

  const label = MOC_VERSIONS[version] ?? `unknown (version byte ${version})`;
  const ok = version <= RUNTIME_MAX;

  console.log(`  ${ok ? "✓" : "✗"} moc3 version ${version} — ${label}`);
  if (!ok) {
    console.log(`    Newer than the bundled runtime accepts (max ${RUNTIME_MAX}).`);
    console.log("    Update Cubism Core: npm run fetch-assets, after deleting src/vendor/.");
    return false;
  }

  if (!settings) {
    console.log("  ✗ no .model3.json — the runtime has nothing to load");
    return false;
  }

  const json = JSON.parse(await readFile(settings, "utf-8"));
  const refs = json.FileReferences ?? {};
  const expressions = (refs.Expressions ?? []).map((e) => e.Name);
  const motions = Object.keys(refs.Motions ?? {});

  // Every path model3.json points at must actually exist. Archives extracted with the
  // wrong codepage — common for models packaged in China or Japan — leave the files
  // renamed to mojibake while the json still names them correctly. Nothing resolves,
  // and the only symptom is a blank canvas.
  const have = new Set(files.map(rel));
  const absent = (list) => list.filter(Boolean).filter((r) => !have.has(r));

  // Without these the model cannot render at all.
  const required = absent([
    refs.Moc,
    ...(refs.Textures ?? []),
    ...(refs.Expressions ?? []).map((e) => e.File),
    ...Object.values(refs.Motions ?? {}).flat().map((m) => m.File),
  ]);
  // These degrade gracefully — Cubism skips them. Haru ships without its cdi3 and
  // renders perfectly.
  const optional = absent([refs.Physics, refs.Pose, refs.DisplayInfo]);

  if (required.length) {
    console.log(`  ✗ ${required.length} required file(s) missing from disk:`);
    for (const b of required.slice(0, 6)) console.log(`      ${b}`);
    if (required.length > 6) console.log(`      … and ${required.length - 6} more`);
    console.log("    If the names look garbled, the archive was unzipped with the wrong");
    console.log("    codepage. Rename the files to match, or repack with UTF-8 names.");
    return false;
  }
  if (optional.length) {
    console.log(`  ~ optional file(s) referenced but absent: ${optional.join(", ")}`);
  }

  console.log(`    settings:    ${rel(settings)}`);
  console.log(`    textures:    ${(refs.Textures ?? []).length}`);
  console.log(`    physics:     ${refs.Physics ? "yes" : "no"}`);
  console.log(`    expressions: ${expressions.length ? expressions.join(", ") : "none"}`);
  console.log(`    motions:     ${motions.length ? motions.join(", ") : "none"}`);
  console.log(`\n    To use it, set this in config.json:`);
  console.log(`      "model": "${name}/${rel(settings)}"`);
  return true;
}

async function main() {
  const [only] = process.argv.slice(2);
  let names;
  try {
    names = only
      ? [only]
      : (await readdir(MODELS, { withFileTypes: true }))
          .filter((e) => e.isDirectory())
          .map((e) => e.name);
  } catch {
    console.error(`No models directory at ${MODELS}`);
    process.exit(1);
  }

  if (!names.length) {
    console.log(`No models in ${MODELS}`);
    return;
  }

  let allOk = true;
  for (const name of names) {
    try {
      await stat(join(MODELS, name));
      allOk = (await inspect(name)) && allOk;
    } catch {
      console.error(`\n${name}\n  ✗ not found in ${MODELS}`);
      allOk = false;
    }
  }
  console.log();
  process.exit(allOk ? 0 : 1);
}

main();
