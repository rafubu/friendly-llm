from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litert_connector.webrtc_responder import WebRTCResponder


@pytest.fixture
def responder():
    r = WebRTCResponder(local_server_url="http://127.0.0.1:11434")
    r._http_client = AsyncMock()
    r._http_client.stream = AsyncMock()
    r._http_client.aclose = AsyncMock()
    return r


@pytest.mark.asyncio
@patch("litert_connector.webrtc_responder.RTCPeerConnection")
async def test_creates_answer_from_offer(mock_pc_class, responder):
    mock_pc = MagicMock()
    mock_pc.localDescription = MagicMock()
    mock_pc.localDescription.sdp = "mock-answer-sdp"
    mock_pc.localDescription.type = "answer"
    mock_pc.createAnswer = AsyncMock()
    mock_pc.setRemoteDescription = AsyncMock()
    mock_pc.setLocalDescription = AsyncMock()
    mock_pc_class.return_value = mock_pc

    result, room_id = await responder.handle_offer(
        "room1", {"sdp": "offer-sdp", "type": "offer"}
    )

    assert room_id == "room1"
    assert result["sdp"]["sdp"] == "mock-answer-sdp"
    mock_pc.setRemoteDescription.assert_called_once()
    mock_pc.createAnswer.assert_called_once()
    assert "room1" in responder._peers


@pytest.mark.asyncio
@patch("litert_connector.webrtc_responder.RTCPeerConnection")
async def test_creates_offer(mock_pc_class, responder):
    mock_pc = MagicMock()
    mock_pc.localDescription = MagicMock()
    mock_pc.localDescription.sdp = "mock-offer-sdp"
    mock_pc.localDescription.type = "offer"
    mock_pc.createOffer = AsyncMock()
    mock_pc.setLocalDescription = AsyncMock()
    mock_pc_class.return_value = mock_pc

    result, room_id = await responder.create_offer("room2")

    assert room_id == "room2"
    assert result["sdp"]["type"] == "offer"
    mock_pc.createOffer.assert_called_once()
    mock_pc.setLocalDescription.assert_called_once()


@pytest.mark.asyncio
async def test_streams_chunks_to_channel(responder, mock_channel):
    async def aiter_lines():
        yield json.dumps({"message": {"content": "Hello"}, "done": False})
        yield json.dumps({"message": {"content": " world"}, "done": False})
        yield json.dumps({"message": {"content": ""}, "done": True, "done_reason": "stop"})

    mock_resp = AsyncMock()
    mock_resp.status_code = 200
    mock_resp.aiter_lines = aiter_lines
    mock_cm = AsyncMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_cm.__aexit__ = AsyncMock(return_value=None)
    responder._http_client.stream = MagicMock(return_value=mock_cm)

    await responder._proxy_request("room1", mock_channel, "req1", "/api/chat", {})

    assert mock_channel.send.call_count >= 2
    sent = [json.loads(c[0][0]) for c in mock_channel.send.call_args_list]
    types = [s["type"] for s in sent]
    assert "chunk" in types
    assert "done" in types


@pytest.mark.asyncio
async def test_handles_http_error(responder, mock_channel):
    mock_resp = AsyncMock()
    mock_resp.status_code = 500
    mock_resp.reason_phrase = "Internal Server Error"
    mock_resp.aread = AsyncMock(return_value=b"Server error")
    mock_cm = AsyncMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_cm.__aexit__ = AsyncMock(return_value=None)
    responder._http_client.stream = MagicMock(return_value=mock_cm)

    await responder._proxy_request("room1", mock_channel, "req1", "/api/chat", {})

    sent = json.loads(mock_channel.send.call_args[0][0])
    assert sent["type"] == "error"
    assert "500" in sent["error"]


@pytest.mark.asyncio
async def test_handles_connection_error(responder, mock_channel):
    mock_cm = AsyncMock()
    mock_cm.__aenter__ = AsyncMock(side_effect=Exception("Connection refused"))
    mock_cm.__aexit__ = AsyncMock(return_value=None)
    responder._http_client.stream = MagicMock(return_value=mock_cm)

    await responder._proxy_request("room1", mock_channel, "req1", "/api/chat", {})

    sent = json.loads(mock_channel.send.call_args[0][0])
    assert sent["type"] == "error"


@pytest.mark.asyncio
async def test_creates_http_client_if_not_started(responder, mock_channel):
    responder._http_client = None

    async def aiter_lines():
        yield json.dumps({"message": {"content": "OK"}, "done": True})

    mock_resp = AsyncMock()
    mock_resp.status_code = 200
    mock_resp.aiter_lines = aiter_lines
    mock_cm = AsyncMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_cm.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value = mock_cm
        await responder._proxy_request("room1", mock_channel, "req1", "/api/chat", {})


class TestExtractFingerprint:
    def test_extracts_fingerprint(self, responder):
        sdp = (
            "v=0\n"
            "o=- 0 0 IN IP4 127.0.0.1\n"
            "s=-\n"
            "t=0 0\n"
            "a=fingerprint:SHA-256 AA:BB:CC:DD:EE:FF\n"
            "m=application 9 UDP/DTLS/SCTP webrtc-datachannel\n"
        )
        fp = responder._extract_fingerprint(sdp)
        assert fp == "SHA-256 AA:BB:CC:DD:EE:FF"

    def test_returns_empty_if_no_fingerprint(self, responder):
        sdp = "v=0\ns=-\nt=0 0\n"
        fp = responder._extract_fingerprint(sdp)
        assert fp == ""


@pytest.mark.asyncio
async def test_removes_all_peers(responder):
    responder._peers = {"r1": MagicMock(), "r2": MagicMock()}
    await responder.cleanup()
    assert len(responder._peers) == 0


@pytest.mark.asyncio
async def test_stops_http_client(responder):
    mock_client = responder._http_client
    await responder.cleanup()
    assert responder._http_client is None
    mock_client.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_removes_peer_and_channel(responder):
    responder._peers["r1"] = MagicMock()
    responder._data_channels["r1"] = MagicMock()
    await responder._remove_peer("r1")
    assert "r1" not in responder._peers
    assert "r1" not in responder._data_channels


@pytest.mark.asyncio
async def test_safe_if_missing(responder):
    await responder._remove_peer("nonexistent")
