"""The control panel's contract with core.

The panel renders whatever core last told it and never predicts. So what has to hold
is: a snapshot arrives unasked on connect, every command produces a fresh one, and the
values in it are the real state rather than an echo of the request.

Also guards the trap this feature walked into: the panel is a *second* client on the
socket the character already uses, and core reads the first `hello` it sees to learn
which expressions the character supports. A panel announcing an empty list must not be
mistaken for a model that cannot emote.

Run:  uv run --directory . python tests/e2e_panel.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile

# Same guard `__main__.main()` applies to the real app. These suites construct VoiceLoop
# directly and so skip it, which leaves stdout as cp1252 on Windows the moment output is
# piped — and core prints `⦿` on several paths. Without this the failure lands somewhere
# unrelated to whatever is actually being tested.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import websockets  # noqa: E402

from aria.config import PERSONA, Config  # noqa: E402
from aria.loop import VoiceLoop  # noqa: E402

RED, GREEN, DIM, RESET = "\033[31m", "\033[32m", "\033[2m", "\033[0m"
results: list[tuple[bool, str]] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    results.append((ok, label))
    mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"  [{mark}] {label}" + (f"  — {detail}" if detail else ""))


async def next_of(ws, kind: str, timeout=5.0) -> dict:
    async with asyncio.timeout(timeout):
        while True:
            msg = json.loads(await ws.recv())
            if msg.get("type") == kind:
                return msg


async def next_settings(ws, timeout=5.0) -> dict:
    return await next_of(ws, "settings", timeout)


async def main() -> int:
    cfg = Config()
    cfg.validate()
    scratch = Path(tempfile.mkdtemp(prefix="aria-test-"))
    cfg.memory.path = scratch / "memory.json"
    # And the persona, for exactly the reason the memory file is redirected: this suite
    # saves and resets one, and pointed at the default it edits *his* Aria. It got away
    # with it only because the last check happens to reset — a failure anywhere before
    # that would have left a test persona on disk as who she is.
    cfg.memory.persona_path = scratch / "persona.txt"
    cfg.memory.auto_extract = False
    cfg.vision.backend = "ollama"

    loop = VoiceLoop(cfg)
    await loop.setup(listen=False)
    url = f"ws://{cfg.server.host}:{cfg.server.port}"

    # A character connects first, as it would in real use.
    async with websockets.connect(url) as character:
        # `emotions`, not `expressions` — PROTOCOL.md documents the latter but
        # `_available_emotions` reads the former, and the overlay sends `emotions`.
        # The doc is the thing that's wrong; noted rather than silently reconciled.
        await character.send(json.dumps({
            "type": "hello", "model_name": "test",
            "emotions": ["happy", "sad", "shy"],
        }))
        await asyncio.sleep(0.2)

        async with websockets.connect(url) as panel:
            await panel.send(json.dumps({
                "type": "hello", "role": "panel", "model_name": "panel",
            }))

            s = await next_settings(panel)
            check(True, "a snapshot arrives on connect, unasked")
            check(s["voice"] == cfg.tts.voice, "it reports the real voice", s["voice"])
            check(len(s["voices"]) > 10, "and the full list to choose from",
                  f"{len(s['voices'])} voices")

            # The trap: the panel's hello must not overwrite the character's.
            check(loop._available_emotions() == ["happy", "sad", "shy"],
                  "the panel did not erase the character's expressions",
                  str(loop._available_emotions()))

            notices: list[dict] = []

            async def command(name, value=None) -> dict:
                """Send a command and read everything it produces, ending at the snapshot.

                Core answers a command with a `settings` broadcast, and sometimes a
                `notice` first. Reading only one leaves the other queued, and the *next*
                command then reads the previous reply — which looks exactly like the
                command having been ignored. Three checks failed that way before this
                drained properly, none of them for the reason they appeared to.
                """
                await panel.send(json.dumps(
                    {"type": "command", "name": name, "value": value}))
                async with asyncio.timeout(5.0):
                    while True:
                        msg = json.loads(await panel.recv())
                        if msg.get("type") == "notice":
                            notices.append(msg)
                        elif msg.get("type") == "settings":
                            return msg

            print("\n--- toggles change real state ---")
            s = await command("set_voice", "af_nicole")
            check(s["voice"] == "af_nicole" and cfg.tts.voice == "af_nicole",
                  "voice hotswap reaches the config", s["voice"])

            s = await command("set_emotion_voice", False)
            check(s["emotion_voice"] is False and cfg.tts.emotion_voice is False,
                  "emotion-in-voice toggles")

            s = await command("set_wake", True)
            check(s["wake_enabled"] and loop.wake is not None, "wake word toggles on")
            s = await command("set_wake", False)
            check(not s["wake_enabled"] and loop.wake is None, "and off again")

            print("\n--- memory ---")
            loop.memory.add("has a demo on Friday", source="you")
            s = await command("settings")
            check(len(s["memory"]) == 1, "notes reach the panel")
            check(s["memory"][0]["source"] == "you",
                  "with who said them, so a wrong note is identifiable")

            s = await command("forget", "demo")
            check(s["memory"] == [] and loop.memory.notes == [],
                  "and can be deleted from the panel")

            print("\n--- editing memory from the panel ---")
            s = await command("add_note", "he takes his coffee black")
            check(len(s["memory"]) == 1, "a note can be added by hand")
            check(s["memory"][0]["source"] == "you",
                  "and counts as told, so eviction never drops it")
            check(bool(s["memory"][0].get("id")),
                  "notes carry an id, so the panel can address one exactly")

            note_id = s["memory"][0]["id"]
            s = await command("edit_note", {"id": note_id, "text": "he takes it white"})
            check(s["memory"][0]["text"] == "he takes it white", "and corrected in place")
            check(s["memory"][0]["id"] == note_id, "keeping its identity")

            # Refused edits must say so. Silently doing nothing is the failure the whole
            # panel is built to avoid.
            notices.clear()
            await command("edit_note", {"id": note_id, "text": "no"})
            check(any(n["level"] == "warn" for n in notices),
                  "a rejected edit warns rather than passing",
                  str([n.get("text") for n in notices])[:70])
            check(loop.memory.notes[0].text == "he takes it white",
                  "and leaves the note alone")

            s = await command("delete_note", note_id)
            check(s["memory"] == [], "and one exact note can be deleted")

            print("\n--- editing the persona ---")
            s = await command("settings")
            check(s["persona"] == cfg.persona, "the panel is given the live persona")
            check(s["persona_is_custom"] is False, "and told it is the built-in")
            check("Right now:" in s["system_prompt"],
                  "plus the whole assembled prompt, read-only")
            check(s["persona"] in s["system_prompt"],
                  "which really does contain the persona it is showing")

            written = "You are Aria. You are terse and you like him anyway."
            s = await command("set_persona", written)
            check(cfg.persona == written, "a saved persona reaches the live config")
            check(s["persona_is_custom"] is True, "and is reported as edited")
            check(written in s["system_prompt"],
                  "so the next turn is built from it", s["system_prompt"][:60])
            check(loop.persona.path.exists(), "and it survives a restart")

            # The edit that silently produces a different assistant.
            notices.clear()
            await command("set_persona", "   ")
            check(any(n["level"] == "warn" for n in notices),
                  "an emptied persona is refused, loudly",
                  str([n.get("text") for n in notices])[:70])
            check(cfg.persona == written, "and she is left exactly as she was")

            s = await command("reset_persona")
            check(s["persona_is_custom"] is False and cfg.persona == PERSONA,
                  "reset brings back the built-in")
            check(not loop.persona.path.exists(),
                  "by deleting the override, not copying over it")

    loop.shutdown()
    failed = [label for ok, label in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    for label in failed:
        print(f"  {RED}FAILED{RESET}: {label}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
