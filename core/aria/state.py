"""The four-state machine every other component reacts to."""

from __future__ import annotations

from enum import Enum
from typing import Callable


class State(str, Enum):
    IDLE = "idle"          # nothing happening
    LISTENING = "listening"  # user is mid-utterance
    THINKING = "thinking"    # transcribing or waiting on the first token
    SPEAKING = "speaking"    # audio is going out

    def __str__(self) -> str:
        return self.value


class StateMachine:
    """Holds current state and notifies listeners on change.

    In M3 the overlay subscribes to this and the character animates off it. For now
    it drives the console display, but the contract is already the right one.
    """

    def __init__(self, on_change: Callable[[State, State], None] | None = None):
        self._state = State.IDLE
        self._listeners: list[Callable[[State, State], None]] = []
        if on_change:
            self._listeners.append(on_change)

    @property
    def current(self) -> State:
        return self._state

    def subscribe(self, fn: Callable[[State, State], None]) -> None:
        self._listeners.append(fn)

    def to(self, new: State) -> None:
        if new is self._state:
            return
        old, self._state = self._state, new
        for fn in self._listeners:
            fn(old, new)
