from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import litert_lm
import pytest
from httpx import AsyncClient


def _make_mock_engine():
    engine = MagicMock(spec=litert_lm.Engine)
    engine.__enter__ = MagicMock(return_value=engine)
    engine.__exit__ = MagicMock(return_value=None)

    conv = MagicMock()
    conv.token_count = 10
    conv.__enter__ = MagicMock(return_value=conv)
    conv.__exit__ = MagicMock(return_value=None)

    chat_chunk = MagicMock()
    chat_chunk.get.return_value = [{"type": "text", "text": "Hello!"}]
    conv.send_message_async.return_value = [chat_chunk]

    engine.create_conversation.return_value = conv

    entry = MagicMock()
    entry.engine = engine
    entry.queue = AsyncMock()
    entry.queue.active = 0
    entry.queue.acquire = AsyncMock()
    entry.queue.release = MagicMock()

    return entry


@pytest.fixture
def mock_registry():
    entry = _make_mock_engine()
    with patch("litert_ollama.routers.chat.registry") as mock:
        mock.load_engine = AsyncMock(return_value=entry)
        mock.release_engine = AsyncMock()
        yield mock


async def test_chat_missing_model(client: AsyncClient):
    resp = await client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code in (400, 422)


async def test_chat_missing_messages(client: AsyncClient):
    resp = await client.post("/api/chat", json={"model": "test-model"})
    assert resp.status_code in (400, 422)


async def test_chat_empty_messages(client: AsyncClient):
    resp = await client.post("/api/chat", json={"model": "test-model", "messages": []})
    assert resp.status_code in (400, 422)


async def test_chat_streams_response(client: AsyncClient, mock_registry):
    resp = await client.post("/api/chat", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code == 200
    lines = resp.text.strip().split("\n")
    assert len(lines) >= 3

    statuses = []
    for line in lines:
        data = json.loads(line)
        if "status" in data:
            statuses.append(data["status"])
        else:
            assert "model" in data
            if "message" in data:
                assert "content" in data["message"]

    assert "loading" in statuses
    assert "loaded" in statuses

    last = json.loads(lines[-1])
    assert last.get("done") is True


async def test_chat_streams_content_tokens(client: AsyncClient, mock_registry):
    resp = await client.post("/api/chat", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
    })
    lines = resp.text.strip().split("\n")
    content_chunks = [
        json.loads(l)["message"]["content"]
        for l in lines
        if not json.loads(l).get("done") and "message" in json.loads(l) and "status" not in json.loads(l)
    ]
    combined = "".join(content_chunks)
    assert "Hello" in combined


async def test_chat_context_overflow(client: AsyncClient):
    with patch("litert_ollama.routers.chat.settings") as mock_settings:
        mock_settings.context_length = 10

        resp = await client.post("/api/chat", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "x" * 100}],
        })
        assert resp.status_code == 200
        lines = resp.text.strip().split("\n")
        last = json.loads(lines[-1]) if resp.text.strip() else {}
        assert last.get("done_reason") == "context_overflow"


async def test_chat_model_not_found(client: AsyncClient):
    with patch("litert_ollama.routers.chat.registry") as mock:
        mock.load_engine = AsyncMock(side_effect=FileNotFoundError("Model not found"))

        resp = await client.post("/api/chat", json={
            "model": "nonexistent",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 200
        lines = resp.text.strip().split("\n")
        last = json.loads(lines[-1])
        assert last.get("status") == "error"
        assert "not found" in last.get("error", "").lower()


async def test_chat_returns_done_reason(client: AsyncClient, mock_registry):
    resp = await client.post("/api/chat", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
    })
    lines = resp.text.strip().split("\n")
    last = json.loads(lines[-1])
    assert "done_reason" in last
    assert last["done_reason"] == "stop"


async def test_chat_preserves_model_name(client: AsyncClient, mock_registry):
    resp = await client.post("/api/chat", json={
        "model": "my-custom-model",
        "messages": [{"role": "user", "content": "hi"}],
    })
    lines = resp.text.strip().split("\n")
    for line in lines:
        data = json.loads(line)
        if "status" not in data:
            assert data["model"] == "my-custom-model"


async def test_chat_with_system_prompt(client: AsyncClient, mock_registry):
    resp = await client.post("/api/chat", json={
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ],
    })
    assert resp.status_code == 200
    mock_registry.load_engine.assert_awaited()
