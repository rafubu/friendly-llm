from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Callable

import websockets

logger = logging.getLogger(__name__)


class SignalingClient:
    def __init__(
        self,
        url: str,
        api_key: str,
        node_name: str,
        models: list[dict[str, Any]],
        visibility: str = "public",
        max_sessions: int = 5,
        allowed_keys: list[str] | None = None,
        on_message: Callable | None = None,
    ):
        self.url = url
        self.api_key = api_key
        self.node_name = node_name
        self.models = models
        self.visibility = visibility
        self.max_sessions = max_sessions
        self.allowed_keys = allowed_keys or []
        self.on_message = on_message

        self.ws = None
        self.connected = False
        self.node_id: str | None = None
        self._reconnect_delay = 1
        self._max_reconnect_delay = 60
        self._stop = False
        self._heartbeat_task: asyncio.Task | None = None
        self._listen_task: asyncio.Task | None = None

    async def connect(self):
        while not self._stop:
            try:
                self.ws = await websockets.connect(self.url, ping_interval=25, ping_timeout=10)
                await self.ws.send(json.dumps({"type": "auth", "api_key": self.api_key}))
                resp = json.loads(await self.ws.recv())

                if resp.get("type") == "auth_ok":
                    self.node_id = resp.get("node_id")
                    self.connected = True
                    self._reconnect_delay = 1
                    logger.info(f"Connected to signaling as {self.node_id}")

                    await self.ws.send(json.dumps({
                        "type": "register",
                        "models": self.models,
                        "max_sessions": self.max_sessions,
                        "visibility": self.visibility,
                        "allowed_keys": self.allowed_keys,
                    }))

                    if self._heartbeat_task and not self._heartbeat_task.done():
                        self._heartbeat_task.cancel()
                    if self._listen_task and not self._listen_task.done():
                        self._listen_task.cancel()
                    self._heartbeat_task = asyncio.create_task(self._heartbeat())
                    self._listen_task = asyncio.create_task(self._listen())
                    return

                elif resp.get("type") == "auth_error":
                    logger.error(f"Auth error: {resp.get('error')}")
                    await asyncio.sleep(5)
                    continue

            except (websockets.WebSocketException, ConnectionError) as e:
                logger.warning(f"Connection failed: {e}")
                self.connected = False
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, self._max_reconnect_delay)

    async def _heartbeat(self):
        while self.connected and self.ws:
            try:
                await self.ws.send(json.dumps({"type": "ping"}))
            except websockets.WebSocketException:
                break
            await asyncio.sleep(30)

    async def _listen(self):
        try:
            async for raw in self.ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                if msg.get("type") == "pong":
                    continue

                if self.on_message:
                    await self.on_message(msg)

        except websockets.WebSocketException:
            pass
        finally:
            self.connected = False
            if not self._stop:
                logger.info("Disconnected, reconnecting...")
                asyncio.create_task(self.connect())

    async def send(self, msg: dict):
        if self.ws and self.connected:
            try:
                await self.ws.send(json.dumps(msg))
            except websockets.WebSocketException:
                pass

    async def close(self):
        self._stop = True
        if self.ws:
            await self.ws.close()
        self.connected = False
