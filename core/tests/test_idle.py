"""Speaking first: when she may, and — mostly — when she must not.

An unprompted voice is the feature in this project with the worst failure mode. Too
eager and it is a machine nagging an empty room; wrong moment and it talks over you.
So these pin the conditions rather than the timer, because the timer is the easy part.
"""

from __future__ import annotations

import asyncio

import pytest

from aria.config import Config
from aria.loop import VoiceLoop
from aria.state import State


class FakeListener:
    def __init__(self, muted=False):
        self.is_muted = muted


@pytest.fixture
def loop() -> VoiceLoop:
    # No setup() — that would load Whisper and open devices. Only the gate is under
    # test, and it reads plain attributes.
    cfg = Config()
    ln = VoiceLoop.__new__(VoiceLoop)
    ln.cfg = cfg
    ln.state = type("S", (), {"current": State.IDLE})()
    ln._speaking_task = None
    ln.listener = FakeListener()
    ln._nudges = 0
    ln._turn_lock = asyncio.Lock()
    return ln


def test_may_speak_into_an_idle_silence(loop):
    assert loop._may_nudge()


def test_never_mid_reply(loop):
    loop._speaking_task = object()
    assert not loop._may_nudge(), "she would talk over herself"


@pytest.mark.parametrize("state", [State.LISTENING, State.THINKING, State.SPEAKING])
def test_never_while_a_turn_is_in_flight(loop, state):
    loop.state.current = state
    assert not loop._may_nudge(), f"she would interrupt during {state}"


def test_never_when_muted(loop):
    # He took her hearing away on purpose. Calling out to someone who has silenced
    # you, and who cannot answer, is a tantrum rather than company.
    loop.listener = FakeListener(muted=True)
    assert not loop._may_nudge()


def test_never_while_a_turn_is_running_elsewhere(loop):
    # She is answering him on Discord right now. The state machine says IDLE because
    # nothing is being spoken, so without the lock check this is the one moment she
    # would call out to an empty room mid-conversation.
    async def scenario() -> None:
        async with loop._turn_lock:
            assert not loop._may_nudge()
        assert loop._may_nudge(), "and she is free again once it finishes"

    asyncio.run(scenario())  # no pytest-asyncio; the loop is one `async with` deep


def test_gives_up_rather_than_nagging(loop):
    loop._nudges = loop.cfg.idle.max_nudges
    assert not loop._may_nudge(), "an empty room should get quieter, not more insistent"


def test_backoff_lengthens_each_time(loop):
    c = loop.cfg.idle
    waits = [c.after_s * (c.backoff**n) for n in range(c.max_nudges)]
    assert waits == sorted(waits) and waits[0] < waits[-1]
    assert waits[0] >= 60, "a first call inside a minute would be twitchy"


def test_headless_runs_never_start_the_watcher():
    # setup(listen=False) is every e2e test in this repo. Speaking first there would
    # be talking to nobody, and would make the suites nondeterministic.
    import inspect

    src = inspect.getsource(VoiceLoop.setup)
    guard = src[src.index("if listen:") :]
    assert "_idle_watch" in guard, "the watcher must start only under `if listen:`"
