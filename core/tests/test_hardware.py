"""Deciding what this machine can run, and why guessing wrong is not a slow start.

Every default here was fixed to a 16 GB RTX 5080: `cuda`, `float16`, `large-v3-turbo`.
On a machine without an NVIDIA GPU that is not slower, it is a CTranslate2 `RuntimeError`
before the first word — which reads as a broken program rather than as "you have no
NVIDIA GPU". The whole point of this module is that someone else's laptop gets an
answer instead of a traceback.
"""

from __future__ import annotations

import pytest

from aria.hardware import COMFORTABLE_VRAM_MB, Machine, detect

GPU = Machine(has_cuda=True, gpu_name="NVIDIA GeForce RTX 5080", vram_mb=16384)
SMALL_GPU = Machine(has_cuda=True, gpu_name="NVIDIA GeForce GTX 1650", vram_mb=4096)
NO_GPU = Machine(has_cuda=False)


def test_a_real_card_gets_what_this_project_was_built_on():
    assert (GPU.device, GPU.compute_type, GPU.whisper_model) == (
        "cuda", "float16", "large-v3-turbo"
    )


def test_no_gpu_runs_rather_than_crashes():
    assert NO_GPU.device == "cpu"
    assert NO_GPU.whisper_model == "base", "large-v3-turbo on CPU is seconds per turn"


def test_cpu_takes_int8_and_gpu_does_not():
    # Opposite reasons. int8 Whisper fails on Blackwell with
    # CUBLAS_STATUS_NOT_SUPPORTED, so float16 is the only option on NVIDIA. On CPU
    # there is no float16 path worth having and int8 is roughly twice as fast.
    assert NO_GPU.compute_type == "int8"
    assert GPU.compute_type == "float16"


def test_a_small_card_leaves_room_for_the_chat_model():
    # The GPU is shared: turbo is ~1.6 GB in float16 and a 9B at Q4 is another ~6 GB.
    # Taking the big Whisper on a 4 GB card wins the transcription and loses the reply.
    assert SMALL_GPU.device == "cuda"
    assert SMALL_GPU.whisper_model == "small"


def test_local_chat_is_gated_on_a_gpu_that_can_hold_it():
    assert GPU.can_run_a_local_chat_model
    assert not NO_GPU.can_run_a_local_chat_model
    assert not SMALL_GPU.can_run_a_local_chat_model


def test_the_threshold_is_between_the_two_cases():
    assert SMALL_GPU.vram_mb < COMFORTABLE_VRAM_MB < GPU.vram_mb


@pytest.mark.parametrize("machine", [GPU, SMALL_GPU, NO_GPU])
def test_every_machine_can_say_what_it_is(machine):
    # This string goes in front of someone who is about to wait, so it has to be a
    # sentence rather than a repr.
    assert machine.describe() and "Machine(" not in machine.describe()


def test_forcing_cpu_needs_no_broken_driver_to_test(monkeypatch):
    # A broken CUDA install is far easier to rule out than to diagnose, and this is the
    # escape hatch that lets someone do it in one line.
    monkeypatch.setenv("ARIA_FORCE_CPU", "1")
    assert detect().device == "cpu"


def test_detection_answers_on_this_machine_whatever_it_is():
    found = detect()
    assert found.device in ("cuda", "cpu")
    if found.has_cuda:
        assert found.gpu_name, "cuda was claimed with no card behind it"


# --- the first-run settings file ----------------------------------------------
def test_a_blank_backend_means_the_default_not_a_crash():
    """`ARIA_LLM_BACKEND=` sat in the example file waiting to be filled in.

    Copying the example and starting her — the exact first-run path — produced
    `ValueError: unknown LLM backend ''` out of a traceback fifteen frames deep, before
    she had said anything. Found by booting the public copy cold, which is the only way
    it could have been found: every existing test builds a Config with a backend set.

    Blank is not a wrong value, it is an absent one, and the only sensible reading is
    the default.
    """
    from aria.config import LLMConfig
    from aria.llm import build

    built = build(LLMConfig(backend=""))
    assert type(built).__name__ == "OllamaLLM"


def test_a_wrong_backend_still_says_what_to_do():
    from aria.config import LLMConfig
    from aria.llm import build

    with pytest.raises(ValueError) as e:
        build(LLMConfig(backend="chatgpt"))
    message = str(e.value)
    assert "chatgpt" in message, "it has to name what was actually set"
    assert "core/.env" in message, "and where to fix it"
    assert "groq" in message, "and what the valid options are"
