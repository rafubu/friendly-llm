from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litert_connector.signaling_client import SignalingClient


@pytest.fixture
def client():
    return SignalingClient(
        url="ws://test:9876",
        api_key="sk-test",
        node_name="test-node",
        models=[{"name": "gemma-4", "max_sessions": 3}],
    )


@pytest.fixture
def mock_connect(mock_websocket):
    with patch("websockets.connect", new_callable=AsyncMock) as m:
        m.return_value = mock_websocket
        yield m, mock_websocket


@pytest.fixture
def fast_auth_ws(mock_websocket):
    async def recv_side():
        return json.dumps({"type": "auth_ok", "node_id": "n1"})
    mock_websocket.recv = recv_side
    return mock_websocket


@pytest.mark.asyncio
async def test_successful_connect(client, mock_connect, fast_auth_ws):
    _, ws = mock_connect
    await client.connect()
    assert client.connected is True
    assert client.node_id == "n1"


@pytest.mark.asyncio
async def test_auth_error_retries(client):
    with patch("websockets.connect", new_callable=AsyncMock) as mock:
        ws = AsyncMock()
        recv_responses = [
            json.dumps({"type": "auth_error", "error": "Invalid key"}),
            json.dumps({"type": "auth_ok", "node_id": "node_xyz"}),
        ]

        async def recv_side():
            return recv_responses.pop(0)

        ws.recv = recv_side
        mock.return_value = ws

        await client.connect()

    assert client.connected is True
    assert client.node_id == "node_xyz"


@pytest.mark.asyncio
async def test_connection_refused_retries(client):
    with patch("websockets.connect", new_callable=AsyncMock) as mock:
        ws = AsyncMock()

        async def recv_side():
            return json.dumps({"type": "auth_ok", "node_id": "retry"})

        ws.recv = recv_side
        call_count = 0

        async def connect_side(*a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Connection refused")
            return ws

        mock.side_effect = connect_side

        await client.connect()

    assert client.connected is True
    assert client.node_id == "retry"


@pytest.mark.asyncio
async def test_sends_register_after_auth(client, mock_connect, fast_auth_ws):
    _, ws = mock_connect
    await client.connect()

    register_calls = [
        c for c in ws.send.call_args_list
        if '"register"' in str(c)
    ]
    assert len(register_calls) >= 1
    call_arg = json.loads(register_calls[0][0][0])
    assert "models" in call_arg
    assert call_arg["models"][0]["name"] == "gemma-4"


@pytest.mark.asyncio
async def test_stop_flag_prevents_connect(client):
    client._stop = True
    await client.connect()
    assert client.connected is False


@pytest.mark.asyncio
async def test_sends_ping_heartbeat(client, mock_connect, fast_auth_ws):
    _, ws = mock_connect
    await client.connect()
    assert client._heartbeat_task is not None


@pytest.mark.asyncio
async def test_send_json(client, mock_connect, fast_auth_ws):
    _, ws = mock_connect
    await client.connect()
    await client.send({"type": "ping"})
    ws.send.assert_any_call(json.dumps({"type": "ping"}))


@pytest.mark.asyncio
async def test_send_when_not_connected(client):
    await client.send({"type": "ping"})


@pytest.mark.asyncio
async def test_send_after_close(client, mock_connect, fast_auth_ws):
    _, ws = mock_connect
    await client.connect()
    await client.close()
    assert client.connected is False
    ws.close.assert_called_once()


@pytest.mark.asyncio
async def test_listen_task_created_on_connect(client, mock_connect):
    """Verifies listen task exists (message forwarding tested via e2e)."""
    _, ws = mock_connect
    ws.recv = AsyncMock(return_value=json.dumps({"type": "auth_ok", "node_id": "n1"}))

    await client.connect()
    assert client._listen_task is not None
    assert not client._listen_task.done()
