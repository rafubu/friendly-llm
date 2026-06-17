from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc.sdp import candidate_from_sdp
import httpx

logger = logging.getLogger(__name__)


class WebRTCResponder:
    def __init__(
        self,
        local_server_url: str = "http://127.0.0.1:11434",
        on_connection: Callable | None = None,
        on_disconnection: Callable | None = None,
        max_concurrent_requests: int = 3,
    ):
        self.local_server_url = local_server_url
        self.on_connection = on_connection
        self.on_disconnection = on_disconnection
        self._peers: dict[str, RTCPeerConnection] = {}
        self._data_channels: dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._request_sem = asyncio.Semaphore(max_concurrent_requests)
        self._http_client: httpx.AsyncClient | None = None

    async def start(self):
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=300)

    async def stop(self):
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def create_offer(self, room_id: str) -> tuple[dict, str]:
        pc = RTCPeerConnection(
            configuration=RTCConfiguration([
                RTCIceServer(urls="stun:stun.l.google.com:19302"),
                RTCIceServer(urls="stun:stun1.l.google.com:19302"),
            ])
        )

        @pc.on("datachannel")
        async def on_datachannel(channel):
            self._data_channels[room_id] = channel
            await self._start_listener(room_id, channel)

        @pc.on("iceconnectionstatechange")
        async def on_ice_change():
            logger.info(f"ICE state: {pc.iceConnectionState}")
            if pc.iceConnectionState in ("failed", "disconnected", "closed"):
                await self._remove_peer(room_id)

        @pc.on("connectionstatechange")
        async def on_conn_change():
            logger.info(f"Connection state: {pc.connectionState}")
            if pc.connectionState == "connected":
                if self.on_connection:
                    await self.on_connection(room_id)
            elif pc.connectionState in ("failed", "disconnected", "closed"):
                await self._remove_peer(room_id)

        await pc.setLocalDescription(await pc.createOffer())
        self._peers[room_id] = pc

        sdp = {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        dtls_fingerprint = self._extract_fingerprint(pc.localDescription.sdp)

        return {"sdp": sdp, "fingerprint": dtls_fingerprint}, room_id

    def _extract_fingerprint(self, sdp: str) -> str:
        for line in sdp.split("\n"):
            if line.strip().startswith("a=fingerprint:"):
                return line.strip().split(":", 1)[1].strip()
        return ""

    async def _start_listener(self, room_id: str, channel):
        @channel.on("message")
        async def on_message(raw):
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                return

            if msg.get("type") == "infer":
                asyncio.create_task(self._handle_infer(room_id, channel, msg))

    async def _handle_infer(self, room_id: str, channel, msg: dict):
        request_id = msg.get("request_id", "")
        endpoint = msg.get("endpoint", "/api/chat")
        payload = msg.get("payload", {})

        # Backpressure: acquire semaphore or wait
        async with self._request_sem:
            await self._proxy_request(room_id, channel, request_id, endpoint, payload)

    async def _proxy_request(
        self, room_id: str, channel, request_id: str, endpoint: str, payload: dict
    ):
        client = self._http_client
        if client is None:
            client = httpx.AsyncClient(timeout=300)
            try:
                await self._do_stream(client, channel, request_id, endpoint, payload)
            finally:
                await client.aclose()
            return

        await self._do_stream(client, channel, request_id, endpoint, payload)

    async def _do_stream(
        self, client: httpx.AsyncClient, channel, request_id: str, endpoint: str, payload: dict
    ):
        try:
            async with client.stream(
                "POST",
                f"{self.local_server_url}{endpoint}",
                json=payload,
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not data.get("done"):
                        try:
                            channel.send(json.dumps({
                                "type": "chunk",
                                "request_id": request_id,
                                "data": data,
                            }))
                        except Exception:
                            return
                    else:
                        channel.send(json.dumps({
                            "type": "done",
                            "request_id": request_id,
                            "data": data,
                        }))
        except Exception as e:
            try:
                channel.send(json.dumps({
                    "type": "error",
                    "request_id": request_id,
                    "error": str(e),
                }))
            except Exception:
                pass

    async def handle_offer(self, room_id: str, offer_sdp: dict) -> tuple[dict, str]:
        pc = RTCPeerConnection(
            configuration=RTCConfiguration([
                RTCIceServer(urls="stun:stun.l.google.com:19302"),
                RTCIceServer(urls="stun:stun1.l.google.com:19302"),
            ])
        )

        @pc.on("datachannel")
        async def on_datachannel(channel):
            self._data_channels[room_id] = channel
            await self._start_listener(room_id, channel)

        @pc.on("iceconnectionstatechange")
        async def on_ice_change():
            logger.info(f"ICE state: {pc.iceConnectionState}")
            if pc.iceConnectionState in ("failed", "disconnected", "closed"):
                await self._remove_peer(room_id)

        @pc.on("connectionstatechange")
        async def on_conn_change():
            logger.info(f"Connection state: {pc.connectionState}")
            if pc.connectionState == "connected":
                if self.on_connection:
                    await self.on_connection(room_id)
            elif pc.connectionState in ("failed", "disconnected", "closed"):
                await self._remove_peer(room_id)

        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=offer_sdp["sdp"], type=offer_sdp["type"])
        )
        await pc.setLocalDescription(await pc.createAnswer())
        self._peers[room_id] = pc

        sdp = {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        dtls_fingerprint = self._extract_fingerprint(pc.localDescription.sdp)

        return {"sdp": sdp, "fingerprint": dtls_fingerprint}, room_id

    async def set_remote_answer(self, room_id: str, answer_sdp: dict):
        pc = self._peers.get(room_id)
        if not pc:
            raise ValueError(f"No peer for room {room_id}")
        answer = RTCSessionDescription(sdp=answer_sdp["sdp"], type=answer_sdp["type"])
        await pc.setRemoteDescription(answer)

    async def add_ice_candidate(self, room_id: str, candidate: dict):
        pc = self._peers.get(room_id)
        if not pc:
            return
        try:
            cand = candidate_from_sdp(candidate.get("candidate", ""))
            if cand:
                await pc.addIceCandidate(cand)
        except Exception as e:
            logger.warning(f"ICE candidate error: {e}")

    async def _remove_peer(self, room_id: str):
        async with self._lock:
            pc = self._peers.pop(room_id, None)
            self._data_channels.pop(room_id, None)
            if pc:
                try:
                    await pc.close()
                except Exception:
                    pass
            if self.on_disconnection:
                await self.on_disconnection(room_id)

    async def send_to_client(self, room_id: str, msg: dict):
        channel = self._data_channels.get(room_id)
        if channel and channel.readyState == "open":
            try:
                channel.send(json.dumps(msg))
            except Exception as e:
                logger.warning(f"Send error: {e}")

    async def cleanup(self):
        for room_id in list(self._peers.keys()):
            await self._remove_peer(room_id)
        await self.stop()
