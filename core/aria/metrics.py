"""Latency measurement.

This is the actual deliverable of M1. If these numbers are wrong the whole design has
to change, so they are measured honestly — in particular `perceived_ms` includes the
VAD endpoint hold, because the user is sitting in that silence too.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field


@dataclass
class TurnMetrics:
    turn: int
    speech_s: float = 0.0
    #: VAD hold — the silence we must observe before believing the turn ended.
    endpoint_ms: float = 0.0
    stt_ms: float = 0.0
    #: True when the transcript was already finished before the endpoint fired,
    #: because it was started speculatively during the pause.
    stt_speculative: bool = False
    llm_ttft_ms: float = 0.0
    #: First token -> first complete speakable chunk. Pure LLM generation time.
    chunk_ms: float = 0.0
    tts_ms: float = 0.0
    #: The user talked over this reply and it was cut short.
    interrupted: bool = False
    #: A backend failed and no audio was produced. Excluded from latency stats — a
    #: turn that never spoke has no response time, and averaging its zeros in makes
    #: a broken system look fast.
    failed: bool = False

    @property
    def downstream_ms(self) -> float:
        """Work we control: endpoint decision -> first audio."""
        return self.stt_ms + self.llm_ttft_ms + self.chunk_ms + self.tts_ms

    @property
    def perceived_ms(self) -> float:
        """What the user feels: stopped talking -> heard something."""
        return self.endpoint_ms + self.downstream_ms

    def line(self) -> str:
        return (
            f"  turn {self.turn:<3} "
            f"vad {self.endpoint_ms:5.0f} │ "
            f"stt {self.stt_ms:5.0f}{'*' if self.stt_speculative else ' '}│ "
            f"llm {self.llm_ttft_ms:5.0f} │ "
            f"gen {self.chunk_ms:5.0f} │ "
            f"tts {self.tts_ms:5.0f} │ "
            f"= {self.perceived_ms:6.0f} ms to first audio"
        )


@dataclass
class Session:
    turns: list[TurnMetrics] = field(default_factory=list)

    def add(self, m: TurnMetrics) -> None:
        self.turns.append(m)

    def summary(self) -> str:
        if not self.turns:
            return "No turns recorded."

        n_failed = sum(t.failed for t in self.turns)
        turns = [t for t in self.turns if not t.failed]
        if not turns:
            return (
                f"\nAll {n_failed} turn(s) failed before producing audio — "
                "no latency to report."
            )

        def stat(name: str, values: list[float]) -> str:
            med = statistics.median(values)
            return f"  {name:<22} {med:6.0f} ms   (min {min(values):.0f}, max {max(values):.0f})"

        perceived = [t.perceived_ms for t in turns]
        med = statistics.median(perceived)
        verdict = (
            "conversational" if med < 1200
            else "usable but sluggish" if med < 2000
            else "too slow — needs work"
        )
        hits = sum(t.stt_speculative for t in turns)
        cut = sum(t.interrupted for t in turns)
        lines = [
            "",
            f"Latency over {len(turns)} turns"
            + (f", {cut} interrupted" if cut else "")
            + " (median):",
            stat("VAD endpoint hold", [t.endpoint_ms for t in turns]),
            stat(f"STT ({hits}/{len(turns)} speculative)", [t.stt_ms for t in turns]),
            stat("LLM first token", [t.llm_ttft_ms for t in turns]),
            stat("LLM to first chunk", [t.chunk_ms for t in turns]),
            stat("TTS first chunk", [t.tts_ms for t in turns]),
            "  " + "─" * 52,
            stat("PERCEIVED", perceived),
            "",
            f"  Verdict: {verdict} (target < 1200 ms)",
        ]
        if n_failed:
            lines.append(
                f"  {n_failed} further turn(s) failed and are excluded from the above."
            )
        if med >= 1200:
            worst = max(
                ("VAD hold", statistics.median([t.endpoint_ms for t in turns])),
                ("STT", statistics.median([t.stt_ms for t in turns])),
                ("LLM", statistics.median([t.llm_ttft_ms for t in turns])),
                ("generation", statistics.median([t.chunk_ms for t in turns])),
                ("TTS", statistics.median([t.tts_ms for t in turns])),
                key=lambda kv: kv[1],
            )
            lines.append(f"  Dominant term: {worst[0]} at {worst[1]:.0f} ms")
        return "\n".join(lines)
