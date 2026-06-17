from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
import uuid
from typing import Any, AsyncIterator

import httpx
import websockets

from .errors import AuthError, ConnectionError, ModelNotFoundError, RoomCreationError, TimeoutError
from .types import Chunk, ModelInfo, Response

logger = logging.getLogger(__name__)


class LitertClient:
    def __init__(
        self,
        signaling_url: str,
        auth_token: str,
        model: str | None = None,
        timeout: int = 300,
        verify_fingerprint: bool = True,
    ):
        self.signaling_url = signaling_url
        self.auth_token = auth_token
        self.model = model
        self.timeout = timeout
        self.verify_fingerprint = verify_fingerprint

        self._ws = None
        self._connected = False
        self._user_id = None
        self._room_id = None
        self._node_id = None
        self._data_channel = None
        self._peer_connection = None
        self._pending_requests: dict[str, asyncio.Future] = {}
        self._recv_task = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def connect(self):
        try:
            self._ws = await websockets.connect(self.signaling_url)
        except Exception as e:
            raise ConnectionError(f"Could not connect to {self.signaling_url}: {e}")

        await self._ws.send(json.dumps({"type": "auth_jwt", "token": self.auth_token}))
        resp = json.loads(await self._ws.recv())

        if resp.get("type") == "auth_error":
            raise AuthError(resp.get("error", "Authentication failed"))

        self._user_id = resp.get("user_id")
        self._connected = True
        self._recv_task = asyncio.create_task(self._receive_loop())

    async def list_models(self) -> list[ModelInfo]:
        if not self._connected:
            raise ConnectionError("Not connected")

        await self._ws.send(json.dumps({"type": "list_nodes"}))
        resp = json.loads(await self._ws.recv())

        if resp.get("type") == "node_list":
            return [ModelInfo(**m) for m in resp.get("nodes", [])]
        return []

    async def chat(
        self,
        text: str,
        *,
        images: list[str | bytes] | None = None,
        tools: list[dict] | None = None,
        format: str | None = None,
    ) -> AsyncIterator[Chunk]:
        if not self._connected:
            raise ConnectionError("Not connected")

        request_id = uuid.uuid4().hex
        future = asyncio.get_event_loop().create_future()
        self._pending_requests[request_id] = future

        if not self._room_id:
            if not self.model:
                models = await self.list_models()
                if not models:
                    raise ModelNotFoundError("No models available")
                self.model = models[0].id

            await self._ws.send(json.dumps({"type": "create_room", "model": self.model}))

            resp_raw = await self._ws.recv()
            resp = json.loads(resp_raw)

            if resp.get("type") == "error":
                raise RoomCreationError(resp.get("error", "Room creation failed"))

            self._room_id = resp.get("room_id")
            self._node_id = resp.get("node")

            sdp_offer_raw = await self._ws.recv()
            sdp_msg = json.loads(sdp_offer_raw)

            if sdp_msg.get("type") == "sdp_offer":
                fingerprint = sdp_msg.get("fingerprint", "")
                if self.verify_fingerprint:
                    expected_fp = self._get_expected_fingerprint()
                    if expected_fp and fingerprint != expected_fp:
                        raise ConnectionError("DTLS fingerprint mismatch - possible MITM!")

                from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer

                pc = RTCPeerConnection(
                    configuration=RTCConfiguration([
                        RTCIceServer(urls="stun:stun.l.google.com:19302"),
                    ])
                )
                self._peer_connection = pc

                @pc.on("datachannel")
                def on_datachannel(channel):
                    self._data_channel = channel

                    @channel.on("message")
                    def on_message(raw):
                        try:
                            data = json.loads(raw)
                        except json.JSONDecodeError:
                            return
                        req_id = data.get("request_id", "")
                        if req_id in self._pending_requests:
                            fut = self._pending_requests[req_id]
                            if data.get("type") == "chunk":
                                if not fut.done():
                                    fut.set_result(data)
                            elif data.get("type") == "done":
                                if not fut.done():
                                    fut.set_result(data)
                            elif data.get("type") == "error":
                                if not fut.done():
                                    fut.set_exception(ConnectionError(data.get("error", "")))

                await pc.setRemoteDescription(
                    RTCSessionDescription(sdp=sdp_msg["sdp"]["sdp"], type=sdp_msg["sdp"]["type"])
                )
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)

                await self._ws.send(json.dumps({
                    "type": "sdp_answer",
                    "room_id": self._room_id,
                    "sdp": {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type},
                }))

        if not self._data_channel or (self._data_channel and self._data_channel.readyState != "open"):
            timeout = time.time() + 15
            while time.time() < timeout:
                if self._data_channel and self._data_channel.readyState == "open":
                    break
                await asyncio.sleep(0.1)
            if not self._data_channel or self._data_channel.readyState != "open":
                raise TimeoutError("DataChannel failed to open")

        payload = {"model": self.model, "messages": [{"role": "user", "content": text}], "stream": True}
        if images:
            payload["messages"][0]["images"] = []
            for img in images:
                if isinstance(img, bytes):
                    import base64
                    payload["messages"][0]["images"].append(base64.b64encode(img).decode("utf-8"))
                else:
                    import base64
                    with open(img, "rb") as f:
                        payload["messages"][0]["images"].append(base64.b64encode(f.read()).decode("utf-8"))
        if tools:
            payload["tools"] = tools
        if format:
            payload["format"] = format

        self._data_channel.send(json.dumps({
            "type": "infer",
            "request_id": request_id,
            "endpoint": "/api/chat",
            "payload": payload,
        }))

        while True:
            try:
                result = await asyncio.wait_for(future, timeout=self.timeout)
            except asyncio.TimeoutError:
                raise TimeoutError("Inference timeout")

            if result.get("type") == "done":
                done_data = result.get("data", {})
                yield Chunk(
                    text="",
                    done=True,
                    done_reason=done_data.get("done_reason", "stop"),
                    eval_count=done_data.get("eval_count"),
                    total_duration=done_data.get("total_duration"),
                )
                break
            elif result.get("type") == "chunk":
                chunk_data = result.get("data", {})
                yield Chunk(
                    text=chunk_data.get("message", {}).get("content", ""),
                    done=False,
                )
                future = asyncio.get_event_loop().create_future()
                self._pending_requests[request_id] = future

    def _get_expected_fingerprint(self) -> str:
        if not self.auth_token:
            return ""
        try:
            from jose import jwt
            payload = jwt.get_unverified_claims(self.auth_token)
            return payload.get("dtls_fingerprint", "")
        except Exception:
            return ""

    async def _receive_loop(self):
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                msg_type = msg.get("type", "")

                if msg_type == "ice_candidate":
                    if self._peer_connection:
                        from aiortc.sdp import candidate_from_sdp
                        cand_str = msg.get("candidate", {}).get("candidate", "")
                        try:
                            cand = candidate_from_sdp(cand_str)
                            if cand:
                                await self._peer_connection.addIceCandidate(cand)
                        except Exception:
                            pass

                elif msg_type == "room_closed":
                    self._room_id = None
                    self._data_channel = None
                    if self._peer_connection:
                        await self._peer_connection.close()
                        self._peer_connection = None

                elif msg_type == "pong":
                    pass

        except websockets.WebSocketException:
            pass

    async def close(self):
        if self._room_id:
            try:
                await self._ws.send(json.dumps({"type": "close_room", "room_id": self._room_id}))
            except Exception:
                pass
        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
        if self._peer_connection:
            await self._peer_connection.close()
        if self._ws:
            await self._ws.close()
        self._connected = False
