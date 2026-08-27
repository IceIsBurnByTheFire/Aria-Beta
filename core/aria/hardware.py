"""What this machine can actually run, decided once at startup.

This project was built on a machine with a 16 GB RTX 5080, and every default reflected
that: `device="cuda"`, `compute_type="float16"`, Whisper `large-v3-turbo`. On anything
else those are not slow, they are a crash — `RuntimeError: CUDA failed with error CUDA
driver version is insufficient` out of CTranslate2 before a single word is spoken, which
reads as a broken program rather than as "you have no NVIDIA GPU".

So the machine is measured rather than assumed. Three things follow from what is found:

- **Where Whisper runs.** CUDA when there is a working NVIDIA GPU, CPU otherwise.
- **Which Whisper.** `large-v3-turbo` needs a GPU to stay inside the latency budget; on
  CPU it is several seconds a turn and the conversation stops feeling like one. `base`
  is the honest choice there — worse at names and jargon, fast enough to talk to.
- **What to say about the chat model.** A local 9B on CPU is minutes per reply, not
  seconds. That is not a slower Aria, it is an unusable one, so on a machine without a
  GPU the setup points at a free cloud key instead of pretending local will do.

Nothing here is a preference. Every value can still be overridden in `.env` — someone
with an 8 GB card may want `small` on GPU, someone with a threadripper may genuinely
prefer `medium` on CPU. Detection only decides what happens when nobody has said.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass

log = logging.getLogger(__name__)

#: VRAM below which `large-v3-turbo` is a bad idea even on a real GPU. The turbo model
#: is ~1.6 GB in float16, and it shares the card with whatever is running the chat
#: model — a 9B at Q4 is another ~6 GB. Under this, take the smaller Whisper and leave
#: the room for the LLM.
COMFORTABLE_VRAM_MB = 6000


@dataclass(frozen=True)
class Machine:
    """What was found, and what it means for the defaults."""

    has_cuda: bool
    gpu_name: str = ""
    vram_mb: int = 0

    @property
    def device(self) -> str:
        return "cuda" if self.has_cuda else "cpu"

    @property
    def compute_type(self) -> str:
        # int8 variants of Whisper fail on Blackwell with CUBLAS_STATUS_NOT_SUPPORTED,
        # so float16 is not a preference on NVIDIA, it is the only thing that works.
        # On CPU it is the reverse: there is no float16 path worth having and int8 is
        # roughly twice as fast for no accuracy anyone will notice at this size.
        return "float16" if self.has_cuda else "int8"

    @property
    def whisper_model(self) -> str:
        if not self.has_cuda:
            return "base"
        return "large-v3-turbo" if self.vram_mb >= COMFORTABLE_VRAM_MB else "small"

    @property
    def can_run_a_local_chat_model(self) -> bool:
        """Local chat needs a GPU. On CPU a 9B is minutes per reply, not seconds."""
        return self.has_cuda and self.vram_mb >= COMFORTABLE_VRAM_MB

    def describe(self) -> str:
        if not self.has_cuda:
            return "no NVIDIA GPU found — Whisper will run on the processor"
        vram = f"{self.vram_mb / 1024:.0f} GB" if self.vram_mb else "unknown VRAM"
        return f"{self.gpu_name} ({vram})"


def _nvidia_smi() -> tuple[str, int] | None:
    """Ask the driver, not the framework.

    Deliberately not `torch.cuda.is_available()`: torch is not a dependency of this
    project and importing a framework to ask a hardware question is how you end up
    with a 2 GB install to decide one boolean. `nvidia-smi` ships with the driver, so
    its absence is itself the answer.
    """
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001 — no GPU is a normal answer, not an error
        log.debug("nvidia-smi did not answer: %s", e)
        return None
    if not out:
        return None
    # First line, because a laptop with switchable graphics lists more than one and the
    # first is the discrete card.
    name, _, mem = out.splitlines()[0].partition(",")
    try:
        return name.strip(), int(mem.strip())
    except ValueError:
        return name.strip(), 0


def detect() -> Machine:
    """What this machine is. Cheap enough to call at startup, never in a loop."""
    if os.getenv("ARIA_FORCE_CPU"):
        # An escape hatch that costs one line and saves an afternoon: a broken CUDA
        # install is far easier to rule out than to diagnose.
        return Machine(has_cuda=False)
    found = _nvidia_smi()
    if found is None:
        return Machine(has_cuda=False)
    name, vram = found
    return Machine(has_cuda=True, gpu_name=name, vram_mb=vram)
