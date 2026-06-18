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
        on_ice_candidate: Callable | None = None,
        max_concurrent_requests: int = 3,
    ):
        self.local_server_url = local_server_url
        self.on_connection = on_connection
        self.on_disconnection = on_disconnection
        self.on_ice_candidate = on_ice_candidate
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

    def _setup_ice_handlers(self, pc, room_id: str):
        @pc.on("icecandidate")
        async def on_ice_candidate(event):
            if event.candidate and self.on_ice_candidate:
                await self.on_ice_candidate(room_id, {
                    "candidate": event.candidate.to_dict() if hasattr(event.candidate, "to_dict") else str(event.candidate),
                    "sdpMid": event.candidate.sdpMid,
                    "sdpMLineIndex": event.candidate.sdpMLineIndex,
                })

    async def create_offer(self, room_id: str) -> tuple[dict, str]:
        pc = RTCPeerConnection(
            configuration=RTCConfiguration([
                RTCIceServer(urls="stun:stun.l.google.com:19302"),
                RTCIceServer(urls="stun:stun1.l.google.com:19302"),
            ])
        )

        self._setup_ice_handlers(pc, room_id)

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
        logger.info(f"DataChannel listener started for room {room_id}, state={channel.readyState}")
        @channel.on("message")
        async def on_message(raw):
            logger.info(f"DataChannel raw message received: {len(raw)} bytes, room={room_id}")
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning(f"Non-JSON DataChannel message: {raw[:100]}")
                return

            msg_type = msg.get("type", "")
            logger.info(f"DataChannel message type: {msg_type} for room {room_id}")
            if msg_type == "infer":
                logger.info(f"Infer DETECTADO: endpoint={msg.get('endpoint')}, model={msg.get('payload', {}).get('model')}")
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
            logger.info(f"HTTP Request: POST {self.local_server_url}{endpoint}")
            async with client.stream(
                "POST",
                f"{self.local_server_url}{endpoint}",
                json=payload,
            ) as resp:
                logger.info(f"HTTP Response: {resp.status_code} {resp.reason_phrase}")
                if resp.status_code >= 400:
                    body = await resp.aread()
                    logger.error(f"Engine error body: {body.decode('utf-8', errors='replace')[:500]}")
                    try:
                        channel.send(json.dumps({
                            "type": "error",
                            "request_id": request_id,
                            "error": f"Engine HTTP {resp.status_code}: {body.decode('utf-8', errors='replace')[:200]}",
                        }))
                    except Exception:
                        pass
                    return
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    logger.info(f"Engine stream line ({len(line)} chars): {line[:80]}")
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning(f"Non-JSON streaming line: {line[:100]}")
                        continue
                    if not data.get("done"):
                        try:
                            channel.send(json.dumps({
                                "type": "chunk",
                                "request_id": request_id,
                                "data": data,
                            }))
                            logger.info(f"Chunk sent to DataChannel")
                        except Exception as e:
                            logger.error(f"Failed to send chunk via DataChannel: {e}")
                            return
                    else:
                        channel.send(json.dumps({
                            "type": "done",
                            "request_id": request_id,
                            "data": data,
                        }))
                        logger.info(f"Done sent to DataChannel")
        except Exception as e:
            logger.error(f"Stream error for room {id(self)}: {e}")
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

        self._setup_ice_handlers(pc, room_id)

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
