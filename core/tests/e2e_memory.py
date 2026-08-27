"""M-next: does she still know him tomorrow?

Runs two sessions against a throwaway memory file. The first teaches her things, the
second is a fresh Memory loaded from disk — the same thing that happens when the
process restarts. What is being checked is that a real note survives, reaches the
system prompt, and comes back out of the model in her own words.

Run:  uv run --directory . python tests/e2e_memory.py
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aria.config import EMOTION_INSTRUCTIONS, Config, emotion_example  # noqa: E402
from aria.emotion import extract  # noqa: E402
from aria.llm.ollama_backend import OllamaLLM  # noqa: E402
from aria.memory import EXTRACT_PROMPT, Memory, parse_extraction  # noqa: E402
from aria.memory_intent import as_note, memory_command  # noqa: E402

RED, GREEN, DIM, RESET = "\033[31m", "\033[32m", "\033[2m", "\033[0m"
results: list[tuple[bool, str]] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    results.append((ok, label))
    mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"  [{mark}] {label}" + (f"  — {detail}" if detail else ""))


def system_for(cfg: Config, memory: Memory) -> str:
    emotions = ["happy", "shy", "sad", "surprised", "angry"]
    return (
        cfg.persona
        + memory.continuity_block()
        + memory.notes_block()
        + "\n"
        + EMOTION_INSTRUCTIONS.format(
            emotions="  ".join(f"[{e}]" for e in emotions),
            example=emotion_example(emotions),
        )
    )


async def say(llm: OllamaLLM, system: str, history: list[dict], text: str) -> str:
    history.append({"role": "user", "content": text})
    reply = ""
    async for token in llm.stream([{"role": "system", "content": system}, *history]):
        reply += token
    spoken, _ = extract(reply.strip())
    history.append({"role": "assistant", "content": spoken})
    return spoken


async def main() -> int:
    cfg = Config()
    cfg.validate()
    llm = OllamaLLM(cfg.llm)
    tmp = Path(tempfile.mkdtemp(prefix="aria-memory-"))
    path = tmp / "memory.json"

    try:
        # ---- session one: she learns -------------------------------------
        print("\n--- session one ---")
        memory = Memory(path).load()
        memory.begin_session(time.time() - 86400 * 2)  # two days ago
        check("first time" in memory.continuity_block(),
              "a brand new memory says they have never met")

        # Dictated: handled without the LLM, exactly as the loop does it.
        parsed = memory_command("remember that my sister is called Mei")
        check(parsed is not None and parsed[0] == "remember",
              "the spoken command is recognised")
        memory.add(as_note(parsed[1]), source="you")

        # Extracted: the real background pass over a real exchange.
        history: list[dict] = []
        system = system_for(cfg, memory)
        said = await say(llm, system, history,
                         "I've got a demo on Friday and I'm nowhere near ready.")
        print(f"{DIM}  aria › {said}{RESET}")

        raw = ""
        async for token in llm.stream([
            {"role": "system", "content": EXTRACT_PROMPT},
            {"role": "user", "content":
             f"Him: I've got a demo on Friday and I'm nowhere near ready.\nYou: {said}"},
        ]):
            raw += token
        note = parse_extraction(raw)
        print(f"{DIM}  extracted: {note!r}{RESET}")
        check(note is not None, "the background pass found something worth keeping",
              repr(note))
        if note:
            memory.add(note)
        check(any("demo" in n.text.lower() or "friday" in n.text.lower()
                  for n in memory.notes),
              "the demo made it into her notes")

        # ---- session two: a fresh process --------------------------------
        print("\n--- session two, loaded from disk ---")
        reloaded = Memory(path).load()
        reloaded.begin_session()
        check(len(reloaded.notes) == len(memory.notes),
              "every note survived the restart", f"{len(reloaded.notes)} notes")
        check("first time" not in reloaded.continuity_block(),
              "she no longer thinks they have just met")
        check("2 days" in reloaded.continuity_block(),
              "she knows how long it has been", reloaded.continuity_block().strip())

        system = system_for(cfg, reloaded)
        check("Mei" in system, "the dictated note reached the system prompt")

        answer = await say(llm, system, [], "Do you remember anything about me?")
        print(f"{DIM}  aria › {answer}{RESET}")
        check("mei" in answer.lower() or "sister" in answer.lower(),
              "she can say a remembered fact back out loud", answer[:80])

        # ---- forgetting ---------------------------------------------------
        print("\n--- forgetting ---")
        gone = reloaded.forget("Mei")
        check(len(gone) == 1, "forget removes exactly the matching note")
        check("Mei" not in system_for(cfg, Memory(path).load()),
              "and it is gone from the prompt after a reload too")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    failed = [label for ok, label in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    for label in failed:
        print(f"  {RED}FAILED{RESET}: {label}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
