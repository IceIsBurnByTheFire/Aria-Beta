/**
 * Fetch the pieces that cannot live in the repo.
 *
 * Two different reasons they are absent, and both are licensing rather than size:
 *   - live2dcubismcore.min.js is Live2D's runtime and is not redistributable. It has
 *     to come from their CDN. Without it the canvas renders nothing and the failure
 *     is silent, so this script is not optional.
 *   - Haru is one of Live2D's sample models, covered by their Free Material License.
 *     Clause 4.1.1 of that agreement says the customer "may not Redistribute all or
 *     part of the Material to third party", and clause 1.10 defines Redistribute
 *     broadly enough to include shipping a copy inside a repository. The only
 *     redistribution exception is 2.1.2, for sample *code* in executable form.
 *
 * So neither is committed, and both are downloaded onto the machine that will use
 * them — which is a licensed use rather than a redistribution.
 *
 * **Both URLs point at the rights holder on purpose.** Haru used to come from a mirror
 * inside a third party's test assets, which is convenient and pushes the same clause
 * 4.1.1 question onto someone else's repository. Live2D publish Haru themselves in
 * Live2D/CubismWebSamples, so there is no reason to ask it of anyone.
 *
 * Using Haru means accepting Live2D's terms:
 *   https://www.live2d.com/eula/live2d-free-material-license-agreement_en.html
 *   https://www.live2d.com/eula/live2d-sample-model-terms_en.html
 * They are free for individuals and businesses under 10,000,000 JPY of annual revenue,
 * and they place limits on what the character may be shown saying — see
 * THIRD-PARTY-NOTICES.md, which is the short version of both.
 */

import { mkdir, writeFile, access } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const VENDOR = resolve(HERE, "..", "src", "vendor");
const MODELS = resolve(HERE, "..", "assets", "models");

const CUBISM_CORE = "https://cubism.live2d.com/sdk-web/cubismcore/live2dcubismcore.min.js";

//: Pinned to a commit rather than a branch. `develop` is what Live2D actually maintain
//: this on — the only tags are `beta1`..`beta6` and are older than the model — so the
//: choice is a moving branch or an immutable hash, and a character that silently
//: changes under people is not a trade worth making for freshness.
const HARU_COMMIT = "b1de66b0b1f1cb881d95fb6158622aeb6a2827bd";
const HARU =
  `https://cdn.jsdelivr.net/gh/Live2D/CubismWebSamples@${HARU_COMMIT}/Samples/Resources/Haru`;

//: Every file `Haru.model3.json` references. Taken from that file rather than from the
//: old list: the official package is not merely a rename of the mirrored one. The
//: motions live under `motions/` rather than `motion/`, four of the six clips are
//: different, and it ships `Haru.cdi3.json` — which the mirrored copy referenced and
//: did not include, so `check-model.mjs` reported it missing on every run.
const HARU_FILES = [
  "Haru.model3.json",
  "Haru.moc3",
  "Haru.physics3.json",
  "Haru.pose3.json",
  "Haru.cdi3.json",
  "Haru.2048/texture_00.png",
  "Haru.2048/texture_01.png",
  ...["haru_g_idle", "haru_g_m15", "haru_g_m26", "haru_g_m06", "haru_g_m20", "haru_g_m09"]
    .map((m) => `motions/${m}.motion3.json`),
  // Names, not just filenames: the emotion table in renderer.js maps moods onto F01..F08
  // and resolves them by name, so these have to stay exactly as Live2D ship them.
  ...["F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08"].map(
    (f) => `expressions/${f}.exp3.json`,
  ),
];

async function exists(path) {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

async function download(url, dest) {
  if (await exists(dest)) {
    return "skip";
  }
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText} — ${url}`);
  }
  await mkdir(dirname(dest), { recursive: true });
  await writeFile(dest, Buffer.from(await res.arrayBuffer()));
  return "ok";
}

async function main() {
  const jobs = [
    [CUBISM_CORE, join(VENDOR, "live2dcubismcore.min.js"), "cubism core"],
    ...HARU_FILES.map((f) => [`${HARU}/${f}`, join(MODELS, "haru", f), `haru/${f}`]),
  ];

  let fetched = 0;
  let skipped = 0;
  for (const [url, dest, label] of jobs) {
    try {
      const result = await download(url, dest);
      if (result === "skip") {
        skipped++;
      } else {
        fetched++;
        console.log(`  ${label}`);
      }
    } catch (err) {
      console.error(`\nFailed: ${label}\n  ${err.message}`);
      process.exit(1);
    }
  }

  console.log(
    `\n${fetched} fetched, ${skipped} already present.` +
      `\n  runtime: ${VENDOR}\n  model:   ${join(MODELS, "haru")}`,
  );

  // Said only when something was actually downloaded. Printing it on every `npm
  // install` would train people to scroll past it, which is the opposite of the point.
  if (fetched > 0) {
    console.log(
      "\n  The Live2D runtime and the Haru character are Live2D's, not this project's," +
        "\n  and using them means accepting their terms. Free for individuals and for" +
        "\n  businesses under 10,000,000 JPY of annual revenue, with limits on what the" +
        "\n  character may be shown saying." +
        "\n    https://www.live2d.com/eula/live2d-free-material-license-agreement_en.html" +
        "\n    https://www.live2d.com/eula/live2d-sample-model-terms_en.html" +
        "\n  Short version: THIRD-PARTY-NOTICES.md\n",
    );
  }
}

main();
