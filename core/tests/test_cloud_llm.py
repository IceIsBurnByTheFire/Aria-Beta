"""The cloud conversation backend: parsing, failing, and staying out of the way.

Driven through `httpx.MockTransport`, so this is the real `stream()` against real bytes
in the real SSE shape — no key, no network, no allowance spent. What cannot be covered
here is whether a given model follows the persona, which is a question about the model
rather than the code.

The failure paths get more attention than the happy one on purpose. A free tier runs out
by design, and the difference between "she says the limit is gone until tomorrow" and a
traceback naming httpx is most of whether this option is usable at all.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from aria.config import CLOUD_PROVIDERS, LLMConfig
from aria.llm import background_for, build
from aria.llm.cloud_backend import CloudLLM
from aria.llm.ollama_backend import OllamaLLM

ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(autouse=True)
def no_real_keys(monkeypatch):
    """Start every test from a machine with no keys configured.

    `config._load_env` has pulled the developer's `core/.env` into the process by import
    time, so without this the suite reports whichever providers happen to be set up here.
    The Discord tests learned this the hard way when public mode was switched on for
    real and two of them started failing.
    """
    for p in CLOUD_PROVIDERS.values():
        monkeypatch.delenv(p.key_env, raising=False)
    # Model overrides leak the same way, and did: a `.env` holding an OpenRouter id sent
    # that id to Groq and Google too, which is the exact cross-provider mix-up the
    # suffixed variables exist to prevent.
    monkeypatch.delenv("ARIA_CLOUD_MODEL", raising=False)
    for name in CLOUD_PROVIDERS:
        monkeypatch.delenv(f"ARIA_CLOUD_MODEL_{name.upper()}", raising=False)


def sse(*chunks: str) -> str:
    """The wire format these endpoints send, keep-alive comments included."""
    lines = [": OPENROUTER PROCESSING"]  # sent while a free request queues
    for c in chunks:
        lines.append('data: {"choices":[{"delta":{"content":"%s"}}]}' % c)
    lines.append("data: [DONE]")
    return "\n\n".join(lines) + "\n\n"


def backend(handler, monkeypatch, provider="openrouter", **over) -> CloudLLM:
    # Through the real constructor, with only the transport swapped. Replacing the whole
    # client was the first version and it quietly stopped covering the headers set in
    # __init__ - the suite went green on a request carrying no Authorization at all.
    p = CLOUD_PROVIDERS[provider]
    monkeypatch.setenv(p.key_env, f"{p.key_prefix}test")
    return CloudLLM(LLMConfig(backend=provider, **over), transport=httpx.MockTransport(handler))


async def collect(llm: CloudLLM, **kw) -> str:
    return "".join([t async for t in llm.stream([{"role": "user", "content": "hi"}], **kw)])


def run(coro):
    import asyncio

    return asyncio.run(coro)


# --- the happy path ----------------------------------------------------------
def test_streams_text_back_in_order(monkeypatch):
    llm = backend(lambda r: httpx.Response(200, text=sse("Oh, ", "you're ", "up early.")),
                  monkeypatch)
    assert run(collect(llm)) == "Oh, you're up early."


def test_keepalive_comments_are_not_parsed_as_data(monkeypatch):
    # ": OPENROUTER PROCESSING" arrives while a free-tier request queues, which is
    # exactly when things are already going slowly. Treating it as JSON would break
    # precisely the requests this has to survive.
    assert run(collect(backend(lambda r: httpx.Response(200, text=sse("fine")), monkeypatch))) == "fine"


def test_reasoning_deltas_never_reach_the_speaker(monkeypatch):
    body = (
        'data: {"choices":[{"delta":{"reasoning":"the user greeted me"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"Hey."}}]}\n\n'
        "data: [DONE]\n\n"
    )
    llm = backend(lambda r: httpx.Response(200, text=body), monkeypatch)
    assert run(collect(llm)) == "Hey.", "reasoning is not speech"


def test_malformed_lines_are_skipped_rather_than_fatal(monkeypatch):
    body = (
        "data: {not json at all\n\n"
        'data: {"choices":[{"delta":{"content":"still here"}}]}\n\n'
        "data: [DONE]\n\n"
    )
    assert run(collect(backend(lambda r: httpx.Response(200, text=body), monkeypatch))) == "still here"


def test_the_request_carries_what_it_should(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.update(json.loads(request.content))
        seen["auth"] = request.headers.get("authorization")
        seen["url"] = str(request.url)
        return httpx.Response(200, text=sse("ok"))

    run(collect(backend(handler, monkeypatch), num_predict=321))
    assert seen["stream"] is True
    assert seen["max_tokens"] == 321, "a per-call budget must override the default"
    assert seen["reasoning"] == {"exclude": True}
    assert seen["auth"] == "Bearer sk-or-test"
    assert seen["url"] == "https://openrouter.ai/api/v1/chat/completions"


# --- reasoning that leaks into the visible content ---------------------------
def sse_raw(*bodies: str) -> str:
    return "\n\n".join(f"data: {b}" for b in bodies) + "\n\ndata: [DONE]\n\n"


def deltas(*texts: str) -> str:
    import json as _json

    return sse_raw(*(_json.dumps({"choices": [{"delta": {"content": t}}]}) for t in texts))


def test_a_think_block_never_reaches_the_speaker(monkeypatch):
    # Observed on Groq's qwen/qwen3.6-27b, which emits a literal <think> block into
    # content: "<think>\nHere's a thinking process:..." Unfiltered that is read aloud.
    body = deltas("<think>", "Here's a thinking process", "</think>", "Hey, you're back.")
    assert run(collect(backend(lambda r: httpx.Response(200, text=body), monkeypatch))) \
        == "Hey, you're back."


def test_a_think_tag_split_across_chunks_is_still_caught(monkeypatch):
    # Tags arrive token by token, so the naive "does this chunk contain <think>" check
    # passes everything through.
    body = deltas("<th", "ink>secret", " reasoning</th", "ink>", "Real answer.")
    assert run(collect(backend(lambda r: httpx.Response(200, text=body), monkeypatch))) \
        == "Real answer."


def test_text_before_and_after_a_block_both_survive(monkeypatch):
    body = deltas("Before. ", "<think>hidden</think>", " After.")
    assert run(collect(backend(lambda r: httpx.Response(200, text=body), monkeypatch))) \
        == "Before.  After."


def test_an_unclosed_block_is_discarded_rather_than_spoken(monkeypatch):
    body = deltas("Fine. ", "<think>", "cut off mid-thought")
    assert run(collect(backend(lambda r: httpx.Response(200, text=body), monkeypatch))) \
        == "Fine. "


def test_ordinary_angle_brackets_are_not_eaten(monkeypatch):
    body = deltas("use ", "a < b ", "and x <- y", " done")
    assert run(collect(backend(lambda r: httpx.Response(200, text=body), monkeypatch))) \
        == "use a < b and x <- y done"


# --- three providers, one protocol -------------------------------------------
@pytest.mark.parametrize("provider,host", [
    ("openrouter", "openrouter.ai"),
    ("groq", "api.groq.com"),
    ("google", "generativelanguage.googleapis.com"),
])
def test_every_provider_reaches_its_own_endpoint(monkeypatch, provider, host):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["model"] = __import__("json").loads(request.content)["model"]
        return httpx.Response(200, text=sse("ok"))

    run(collect(backend(handler, monkeypatch, provider=provider)))
    assert host in seen["url"]
    assert seen["url"].endswith("/chat/completions"), "base_url joining must not drop a path"
    assert seen["model"] == CLOUD_PROVIDERS[provider].default_model


def test_a_model_id_set_for_one_provider_does_not_follow_you_to_another(monkeypatch):
    # Found by this suite: `.env` held an OpenRouter id, and it was being sent to Groq
    # and Google as well. That fails as a 404 on the first thing you say and reads as
    # the new provider being broken rather than a setting left behind.
    monkeypatch.setenv("ARIA_CLOUD_MODEL_OPENROUTER", "some/thing:free")
    assert LLMConfig(backend="openrouter").active_cloud_model == "some/thing:free"
    assert LLMConfig(backend="groq").active_cloud_model == CLOUD_PROVIDERS["groq"].default_model


def test_the_unsuffixed_override_still_works_for_a_single_provider(monkeypatch):
    monkeypatch.setenv("ARIA_CLOUD_MODEL", "llama-3.1-8b-instant")
    assert LLMConfig(backend="groq").active_cloud_model == "llama-3.1-8b-instant"


def test_each_provider_reads_its_own_key_variable(monkeypatch):
    # Three keys with three names. Reading the wrong one is an auth failure that looks
    # like a bad key rather than a missing one.
    monkeypatch.setenv("GROQ_API_KEY", "gsk_groq")
    monkeypatch.setenv("GEMINI_API_KEY", "aizagoogle")
    assert LLMConfig(backend="groq").api_key == "gsk_groq"
    assert LLMConfig(backend="google").api_key == "aizagoogle"
    assert LLMConfig(backend="openrouter").api_key == "", "unset must not fall back"


def test_a_rejected_provider_extra_is_retried_without_it(monkeypatch):
    # The provider-specific fields are the likeliest thing to rot: three services that
    # change independently share one table. A 400 on them should cost a retry, not the
    # assistant.
    attempts = []

    def handler(request):
        body = __import__("json").loads(request.content)
        attempts.append(body)
        if "reasoning" in body:
            return httpx.Response(400, text="unknown parameter: reasoning")
        return httpx.Response(200, text=sse("recovered"))

    assert run(collect(backend(handler, monkeypatch))) == "recovered"
    assert len(attempts) == 2 and "reasoning" not in attempts[1]


def test_a_real_bad_request_is_not_retried_forever(monkeypatch):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400, text="genuinely malformed")

    said = run(collect(backend(handler, monkeypatch)))
    assert said.strip(), "she still says something"
    assert len(calls) == 2, "one retry without extras, then it gives up"


# --- busy, which is different from out of quota ------------------------------
def test_a_busy_model_is_retried_rather_than_given_up_on(monkeypatch):
    # Google answers 503 "experiencing high demand" often enough on a free key to matter
    # and it clears in a second or two. Observed live on gemini-3.5-flash.
    monkeypatch.setattr("aria.llm.cloud_backend._BACKOFF_S", 0.0)
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(503, text="high demand")
        return httpx.Response(200, text=sse("there you go"))

    assert run(collect(backend(handler, monkeypatch))) == "there you go"
    assert len(calls) == 3


def test_a_model_that_stays_busy_says_so_kindly(monkeypatch):
    monkeypatch.setattr("aria.llm.cloud_backend._BACKOFF_S", 0.0)
    said = run(collect(backend(lambda r: httpx.Response(503, text="busy"), monkeypatch)))
    # Oversubscription is not a setup error, so it must not read as one he has to fix.
    assert "busy" in said.lower() and "again" in said.lower()
    assert "terminal" not in said.lower()


def test_retrying_does_not_multiply_with_the_extras_fallback(monkeypatch):
    # Two independent retry paths in one loop. Unbounded interaction would be 9 requests
    # against an allowance measured in hundreds.
    monkeypatch.setattr("aria.llm.cloud_backend._BACKOFF_S", 0.0)
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503, text="busy")

    run(collect(backend(handler, monkeypatch)))
    assert len(calls) <= 6, f"{len(calls)} requests for one turn"


# --- running out, which is the normal end of a free day ----------------------
def test_the_daily_limit_is_explained_not_raised(monkeypatch, capsys):
    said = run(collect(backend(lambda r: httpx.Response(429, text="slow down"), monkeypatch)))
    assert said, "she must say something rather than fall over"
    assert "limit" in said.lower()
    assert "local" in said.lower(), "and name the way out"
    assert "50 a day" in capsys.readouterr().out, "the terminal gets the real numbers"


def test_the_limit_message_is_the_providers_own(monkeypatch, capsys):
    run(collect(backend(lambda r: httpx.Response(429, text="slow down"), monkeypatch, provider="groq")))
    out = capsys.readouterr().out
    assert "1000 a day" in out and "50 a day" not in out


@pytest.mark.parametrize("status", [401, 403])
def test_a_bad_key_is_explained_not_raised(monkeypatch, capsys, status):
    said = run(collect(backend(lambda r: httpx.Response(status, text="no"), monkeypatch)))
    assert "key" in said.lower()
    assert "OPENROUTER_API_KEY" in capsys.readouterr().out


def test_a_retired_model_id_is_explained(monkeypatch, capsys):
    said = run(collect(backend(lambda r: httpx.Response(404, text="gone"), monkeypatch)))
    assert said.strip()
    assert "ARIA_CLOUD_MODEL" in capsys.readouterr().out


def test_a_timeout_still_produces_a_sentence(monkeypatch):
    def handler(request):
        raise httpx.ReadTimeout("too slow", request=request)

    assert "time" in run(collect(backend(handler, monkeypatch))).lower()


def test_a_network_failure_still_produces_a_sentence(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("no route", request=request)

    assert run(collect(backend(handler, monkeypatch))).strip()


# --- preflight ---------------------------------------------------------------
@pytest.mark.parametrize("provider", list(CLOUD_PROVIDERS))
def test_a_missing_key_is_caught_before_anything_is_spent(provider):
    p = CLOUD_PROVIDERS[provider]
    problem = run(CloudLLM(LLMConfig(backend=provider)).preflight())
    assert problem and p.keys_url in problem
    assert p.key_env in problem, "it must name the variable to set"
    assert "Start Aria.bat" in problem, "and the way back to local"


def test_a_key_from_the_wrong_service_is_caught(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-ant-nope")
    problem = run(CloudLLM(LLMConfig(backend="openrouter")).preflight())
    assert problem and "sk-or-" in problem


def test_a_provider_without_a_stable_prefix_is_not_second_guessed(monkeypatch):
    # Google does not document a stable key prefix, so guessing one would reject valid
    # keys - a much worse failure than accepting an invalid one, which the first request
    # catches anyway.
    monkeypatch.setenv("GEMINI_API_KEY", "whatever-shape-this-is")
    assert run(CloudLLM(LLMConfig(backend="google")).preflight()) is None


# --- choosing a backend ------------------------------------------------------
def test_ollama_is_not_a_cloud_provider():
    assert LLMConfig(backend="ollama").is_cloud is False
    assert LLMConfig(backend="ollama").provider is None


def test_the_local_launcher_pins_the_backend():
    # `Start Aria (cloud).bat` deliberately passes no backend so that ARIA_LLM_BACKEND
    # in .env chooses the provider. That makes it possible for setting up cloud mode to
    # silently change what the *login shortcut* starts - so the local launcher has to
    # say ollama out loud rather than relying on the default.
    bat = (ROOT / "Start Aria.bat").read_text(encoding="utf-8", errors="replace")
    assert "--llm-backend ollama" in bat


def test_build_returns_what_was_asked_for(monkeypatch):
    assert isinstance(build(LLMConfig(backend="ollama")), OllamaLLM)
    for provider in CLOUD_PROVIDERS:
        assert isinstance(build(LLMConfig(backend=provider)), CloudLLM), provider


def test_an_unknown_backend_names_the_ones_that_exist():
    with pytest.raises(ValueError, match="groq"):
        build(LLMConfig(backend="anthropic"))


def test_memory_extraction_stays_local_when_the_conversation_does_not():
    # Two calls per turn against OpenRouter's 50 a day is 25 conversations. This is the
    # difference between that and 50 - and it keeps the notes he never sees off the
    # network, on Google's free tier especially.
    for provider in CLOUD_PROVIDERS:
        cfg = LLMConfig(backend=provider)
        assert isinstance(background_for(cfg, build(cfg)), OllamaLLM), provider


def test_local_mode_does_not_hold_two_clients_against_one_server():
    cfg = LLMConfig(backend="ollama")
    conversation = build(cfg)
    assert background_for(cfg, conversation) is conversation
