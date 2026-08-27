"""WebSocket server the overlay connects to.

Contract lives in docs/PROTOCOL.md. Three rules shape this file:

  - Core decides, overlay renders. Nothing here asks the overlay anything.
  - Events are fire-and-forget. A dropped message degrades an animation; it must never
    desync the system or block the voice loop.
  - The overlay survives core going away, so core must equally survive having no
    overlay attached. Running headless is the normal case during development.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable

import websockets
from websockets.asyncio.server import ServerConnection, serve

log = logging.getLogger(__name__)

#: Per-client outbound queue depth. Visemes stream at 50 Hz, so a stalled client would
#: otherwise accumulate unbounded backlog. Dropping the oldest frame is exactly right
#: for animation data: a stale mouth position is worse than a missing one.
_SEND_QUEUE = 64


class _Client:
    def __init__(self, ws: ServerConnection):
        self.ws = ws
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=_SEND_QUEUE)
        self.hello: dict[str, Any] = {}

    def offer(self, payload: str) -> None:
        try:
            self.queue.put_nowait(payload)
        except asyncio.QueueFull:
            try:
                self.queue.get_nowait()  # drop oldest, keep the freshest frame
                self.queue.put_nowait(payload)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass


class OverlayServer:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        on_command: Callable[[dict], None] | None = None,
        on_interact: Callable[[dict], None] | None = None,
        on_hello: Callable[[], None] | None = None,
    ):
        self.host = host
        self.port = port
        self._on_command = on_command
        self._on_interact = on_interact
        #: Fired once a renderer has announced itself. The control panel needs to be
        #: told what everything is currently set to, and connect is the only moment
        #: core knows there is someone to tell.
        self._on_hello = on_hello
        self._clients: set[_Client] = set()
        self._server = None

    @property
    def connected(self) -> bool:
        return bool(self._clients)

    @property
    def capabilities(self) -> dict[str, Any]:
        """What the attached renderer said it can do, from its `hello`.

        M4 feeds the expression list into the system prompt so the LLM cannot ask for
        an expression the loaded model does not have.

        Only renderers count. The control panel is a second client on the same socket
        and it also says hello — but it draws no character, so its empty expression
        list must never be mistaken for "this model has no expressions". Left
        unguarded that silently strips her whole emotional range from the prompt, and
        the symptom is just a face that stops moving.
        """
        for client in self._clients:
            if client.hello and client.hello.get("role") != "panel":
                return client.hello
        return {}

    async def start(self) -> None:
        self._server = await serve(self._handle, self.host, self.port)
        log.info("overlay server on ws://%s:%d", self.host, self.port)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    def close_nowait(self) -> None:
        """Tear down from a synchronous shutdown path, without awaiting."""
        if self._server is not None:
            self._server.close()
            self._server = None
        self._clients.clear()

    def send(self, **msg: Any) -> None:
        """Broadcast one event. Safe to call from any sync context on the loop."""
        if not self._clients:
            return  # headless is normal, not an error
        payload = json.dumps(msg, separators=(",", ":"))
        for client in self._clients:
            client.offer(payload)

    # --- connection handling --------------------------------------------------
    async def _handle(self, ws: ServerConnection) -> None:
        client = _Client(ws)
        self._clients.add(client)
        log.info("overlay connected (%d total)", len(self._clients))
        writer = asyncio.create_task(self._writer(client))
        try:
            async for raw in ws:
                self._on_message(client, raw)
        except websockets.ConnectionClosed:
            pass
        finally:
            writer.cancel()
            self._clients.discard(client)
            log.info("overlay disconnected (%d left)", len(self._clients))

    async def _writer(self, client: _Client) -> None:
        try:
            while True:
                payload = await client.queue.get()
                await client.ws.send(payload)
        except (websockets.ConnectionClosed, asyncio.CancelledError):
            pass
        except Exception as e:
            log.warning("overlay send failed: %s", e)

    def _on_message(self, client: _Client, raw: str | bytes) -> None:
        try:
            msg = json.loads(raw)
            kind = msg.get("type")
        except (ValueError, AttributeError):
            log.warning("unparseable overlay message: %r", raw[:100])
            return

        # Every callback below is guarded. This file's contract is that a dropped or bad
        # message degrades an animation and never desyncs the system — but an exception
        # out of a handler propagates into `_handle`, which only catches
        # `ConnectionClosed`, so the client is dropped with a 1011 and the panel goes
        # dark. Found for real: a `print` in a command handler hit a `⦿` on a cp1252
        # stdout, and one unprintable log line disconnected the whole control panel.
        # The command that raised should be the only casualty.
        try:
            if kind == "hello":
                client.hello = msg
                log.info(
                    "overlay: %s model=%s expressions=%d",
                    msg.get("model_format"), msg.get("model_name"),
                    len(msg.get("expressions", [])),
                )
                if self._on_hello:
                    self._on_hello()
            elif kind == "command" and self._on_command:
                # The whole message, not just the name — a control panel sends arguments
                # ("set voice to af_nicole", "forget this note") and a bare string cannot
                # carry them.
                self._on_command(msg)
            elif kind == "interact" and self._on_interact:
                self._on_interact(msg)
            else:
                log.debug("ignored overlay message: %s", kind)
        except Exception:  # noqa: BLE001 — one bad command, not the whole connection
            log.exception("overlay %s handler failed: %s", kind, str(msg)[:120])
