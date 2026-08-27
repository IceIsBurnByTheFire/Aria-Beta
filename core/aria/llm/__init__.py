"""Picking a conversation backend.

Same shape as `vision.describe.build`, and for the same reason: no option is universally
available. Local needs a model pulled and a GPU worth having; every cloud one needs a key
and spends a metered allowance.

`background_for` is the part that is not just symmetry. Aria makes two calls per turn —
the reply, and a background pass that reads the exchange and decides whether anything is
worth remembering. Sending that second one to the cloud halves whatever the provider
allows in order to pay a large model for a small classification, and ships his private
notes to a third party on the way. So it stays local whenever local is there at all.
"""

from __future__ import annotations

import logging

from ..config import CLOUD_PROVIDERS, LLMConfig
from .base import LLMBackend, Message  # noqa: F401 - re-exported

log = logging.getLogger(__name__)


def build(cfg: LLMConfig):
    if cfg.backend in CLOUD_PROVIDERS:
        from .cloud_backend import CloudLLM

        return CloudLLM(cfg)
    if cfg.backend == "ollama":
        from .ollama_backend import OllamaLLM

        return OllamaLLM(cfg)

    # An empty value is a different mistake from a wrong one, and by far the more common
    # on a first run: `ARIA_LLM_BACKEND=` sits in the example file waiting to be filled
    # in, and leaving it blank is what someone does when they have decided to use the
    # local model and assumed blank meant default. Treat it as the default rather than
    # as an error, because it is the only reading that makes sense.
    if not cfg.backend.strip():
        log.info("no ARIA_LLM_BACKEND set, using the local model")
        from .ollama_backend import OllamaLLM

        return OllamaLLM(cfg)

    known = "', '".join(["ollama", *CLOUD_PROVIDERS])
    raise ValueError(
        f"ARIA_LLM_BACKEND is set to {cfg.backend!r}, which is not one of: '{known}'.\n"
        f"Fix it in core/.env, or delete the line to use the local model."
    )


def background_for(cfg: LLMConfig, conversation):
    """The backend for work nobody is waiting on: memory extraction.

    Returns the local model when the conversation is in the cloud, and otherwise the
    conversation backend itself — there is no point holding two clients against the same
    server. Falls back to `conversation` if the local side cannot be built at all,
    because a missing note is a much smaller loss than a turn that fails.
    """
    if not cfg.is_cloud:
        return conversation
    try:
        from .ollama_backend import OllamaLLM

        return OllamaLLM(cfg)
    except Exception as e:  # noqa: BLE001
        log.warning("no local model for background work, using the cloud one: %s", e)
        return conversation
