# Third-party notices

Aria's own code is MIT — see [LICENSE](LICENSE). Two things it uses are **not** ours and
are **not** MIT. Neither is in this repository; both are downloaded by `Setup.bat` onto
the machine that will run them, which is a licensed use rather than a redistribution.

This page is a summary written for someone deciding whether they can use this project. It
is not legal advice and it is not a substitute for the agreements themselves, which are
linked below and which are the things that actually bind you.

---

## Live2D Cubism Core

`overlay/src/vendor/live2dcubismcore.min.js` — the runtime that renders the character.

Downloaded from Live2D at
<https://cubism.live2d.com/sdk-web/cubismcore/live2dcubismcore.min.js>, and gitignored.
It is not redistributable, which is why this repo contains a fetch script instead of a
copy.

Covered by the **Live2D Proprietary Software License Agreement**:
<https://www.live2d.com/eula/live2d-proprietary-software-license-agreement_en.html>

Businesses with more than **10,000,000 JPY** of annual gross revenue in their most recent
financial year additionally need a **Cubism SDK Release License**:
<https://www.live2d.com/en/download/cubism-sdk/release-license/>

## Haru

`overlay/assets/models/haru/` — the character that ships as the default.

© Live2D Inc. Haru is one of Live2D's sample models. Downloaded from Live2D's own
repository, [Live2D/CubismWebSamples](https://github.com/Live2D/CubismWebSamples), pinned
to a commit, and gitignored.

Covered by **both** of these, and using her means accepting both:

- Free Material License Agreement —
  <https://www.live2d.com/eula/live2d-free-material-license-agreement_en.html>
- Terms of Use for Live2D Cubism Sample Data —
  <https://www.live2d.com/eula/live2d-sample-model-terms_en.html>

### What that means in practice

**Free for most people.** The Free Material License covers General Users and Small-Scale
Enterprises — under **10,000,000 JPY** of annual revenue (clauses 1.22 and 1.23) — for
both commercial and non-commercial creative work.

**You may not redistribute her.** Clause 4.1.1: the customer "may not Redistribute all or
part of the Material to third party". Clause 1.10 defines Redistribute broadly — "to
perform, show, publicly transmit, display or distribute part or whole duplicate of the
Material". The only redistribution exception, clause 2.1.2, is for sample *code* in
executable form, not for art. So: fork this repo freely, but do not commit Haru into your
fork, and do not bundle her into a release or an installer. Leave the download where it
is.

**There are limits on what she may be shown saying**, and this is the clause most worth
your attention here, because it is the one this project can walk into by accident. Clause
4.1.7 prohibits using the Material for works Live2D deems to carry "violent, aggressive,
insulting, or discriminatory expressions", "political or religious assertions", or
"grotesque expressions". Live2D's sample data terms separately treat erotic content as
unsuitable where the character may be perceived as expressing it.

Aria puts a language model's output into Haru's face, lip-synced, with expressions. The
words are not written in advance and they are not reviewed before they are spoken. The
default model is a stock one, and the persona is written for a friend rather than a
romantic partner — but you can point Aria at any model you like, including an uncensored
one, and the README tells you how. If you do that, what the character is shown saying
becomes your call under the clause above.

If you want a character with none of these constraints, see
[docs/CHARACTERS.md](docs/CHARACTERS.md) — any Cubism 3.0+ model works, and a commissioned
one comes with terms you negotiate yourself.

---

## Everything else

Python and Node dependencies carry their own licences, recorded in `core/uv.lock` and
`overlay/package-lock.json`. The speech models downloaded by `aria.setup_models` — Silero
VAD and Kokoro — are fetched at setup and carry their own terms; neither is in this
repository either.
