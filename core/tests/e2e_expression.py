"""M4: do expressions land on the sentence they belong to?

The LLM is replaced with a scripted reply so the markers, the chunk boundaries and the
expected audio offsets are all known exactly. Everything downstream — chunker, marker
extraction, TTS, playback, the expression scheduler — is the real thing.

Run:  uv run --directory . python tests/e2e_expression.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import websockets  # noqa: E402
from _harness import rebase, speak_into_pipeline  # noqa: E402

from aria.config import Config  # noqa: E402
from aria.loop import VoiceLoop  # noqa: E402

RED, GREEN, RESET = "\033[31m", "\033[32m", "\033[0m"
results: list[tuple[bool, str]] = []

# What the overlay claims its character can show. `thinking` is deliberately absent.
AVAILABLE = ["neutral", "happy", "sad", "angry", "shy"]

# Two emotions that exist, one that does not, and a bracket that is not a marker.
SCRIPT = (
    "[happy] Good news, that actually worked. "
    "[thinking] I am not entirely sure why though. "
    "[sad] The old version is gone now. "
    "See item [3] for details."
)


def check(ok: bool, label: str, detail: str = "") -> None:
    results.append((ok, label))
    mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"  [{mark}] {label}" + (f"  — {detail}" if detail else ""))


async def mock_overlay(url, received, ready):
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({
            "type": "hello", "model_format": "live2d", "model_name": "test",
            "emotions": AVAILABLE, "expressions": [], "motions": [],
        }))
        ready.set()
        try:
            async for raw in ws:
                msg = json.loads(raw)
                if msg["type"] in ("expression", "state"):
                    received.append((time.perf_counter(), msg))
        except websockets.ConnectionClosed:
            pass


async def main() -> int:
    cfg = Config()
    cfg.validate()
    # Never touch her real memory. These conversations are fiction, and a test run
    # that leaves "he has a demo on Friday" in the file she loads tomorrow is a bug
    # in the tests, not a feature.
    cfg.memory.path = Path(tempfile.mkdtemp(prefix="aria-test-")) / "memory.json"
    cfg.memory.auto_extract = False
    loop = VoiceLoop(cfg)
    await loop.setup(listen=False)

    url = f"ws://{cfg.server.host}:{cfg.server.port}"
    received: list = []
    ready = asyncio.Event()
    client = asyncio.create_task(mock_overlay(url, received, ready))
    await asyncio.wait_for(ready.wait(), timeout=5)
    await asyncio.sleep(0.1)

    check(loop._available_emotions() == AVAILABLE,
          "core picked up the character's emotion set",
          str(loop._available_emotions()))
    prompt = loop._system_prompt()
    offered = {e for e in AVAILABLE if f"[{e}]" in prompt}
    check(offered == set(AVAILABLE), "system prompt offers every available emotion",
          str(sorted(offered)))
    # Including the worked example — demonstrating a marker the character lacks would
    # teach the model to emit something that gets silently dropped.
    check("[thinking]" not in prompt,
          "prompt never mentions an unavailable emotion, example included")

    # --- scripted reply, real everything else --------------------------------
    async def scripted(_messages, images=None):
        for word in SCRIPT.split(" "):
            yield word + " "
            await asyncio.sleep(0.01)

    loop.llm.stream = scripted

    spoken_to_tts: list[str] = []
    real_synth = loop.tts.synth

    def spy(text: str, emotion: str | None = None):
        spoken_to_tts.append(text)
        return real_synth(text, emotion)

    loop.tts.synth = spy

    utt, _ = speak_into_pipeline(cfg, real_synth, "Tell me what happened.")
    if utt is None:
        print("FAIL: VAD produced no utterance")
        return 1
    rebase(cfg, utt)

    t_speaking = None
    await loop._handle_turn(utt)
    await asyncio.sleep(0.3)

    # --- what reached TTS ----------------------------------------------------
    tts_text = " ".join(spoken_to_tts)
    print(f'\n  to TTS: "{tts_text}"')
    leaked = [e for e in ("happy", "thinking", "sad", "neutral") if f"[{e}]" in tts_text]
    check(not leaked, "no emotion marker was ever sent to TTS", str(leaked))
    # This assertion used to be the other way round: brackets that aren't emotion
    # markers were left alone so `extract` couldn't silently eat "[1]" or "[see
    # below]". Correct for the *parser*, wrong for the *speaker* — nothing in brackets
    # has a sensible pronunciation, and once the date was in her prompt she began
    # inventing them ("I'll remember. [two days] It feels like…"), which Kokoro read
    # out loud. `extract` is still narrow; `clean_for_speech` now takes the rest.
    check("[3]" not in tts_text,
          "ordinary brackets never reach TTS — nothing bracketed is speakable")

    # --- which expressions fired --------------------------------------------
    exprs = [(t, m["value"]) for t, m in received if m["type"] == "expression"]
    states = {m["value"]: t for t, m in received if m["type"] == "state"}
    print(f"  expressions: {[e for _, e in exprs]}")

    check([e for _, e in exprs] == ["happy", "sad"],
          "only available emotions fired, in order", str([e for _, e in exprs]))
    check(all(e != "thinking" for _, e in exprs),
          "unavailable emotion was dropped, not forwarded")

    # --- timing: the second expression must wait for the first sentence ------
    if len(exprs) == 2 and "speaking" in states:
        t_speak = states["speaking"]
        first_gap = exprs[0][0] - t_speak
        second_gap = exprs[1][0] - t_speak
        print(f"  [happy] at +{first_gap * 1000:.0f}ms, [sad] at +{second_gap * 1000:.0f}ms "
              f"after speech started")
        check(first_gap < 0.35,
              "first expression lands as speech starts", f"+{first_gap * 1000:.0f}ms")
        check(second_gap > 1.0,
              "second expression waits for its own sentence to play",
              f"+{second_gap * 1000:.0f}ms — not fired early with the rest of the text")
    else:
        check(False, "timing measurable", f"{len(exprs)} expressions, states={list(states)}")

    client.cancel()
    loop.shutdown()

    failed = [label for ok, label in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    for label in failed:
        print(f"  {RED}FAILED{RESET}: {label}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
