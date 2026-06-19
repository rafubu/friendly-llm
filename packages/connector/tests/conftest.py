from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_websocket():
    ws = AsyncMock()
    ws.send = AsyncMock()
    ws.recv = AsyncMock()
    ws.close = AsyncMock()
    return ws


@pytest.fixture
def mock_websockets_connect(mock_websocket):
    with patch("websockets.connect", new_callable=AsyncMock) as mock:
        mock.return_value = mock_websocket
        yield mock, mock_websocket


@pytest.fixture
def mock_channel():
    ch = MagicMock()
    ch.readyState = "open"
    ch.send = MagicMock()
    ch.close = MagicMock()
    return ch


@pytest.fixture
def mock_httpx_client():
    client = AsyncMock()
    client.stream = AsyncMock()
    client.aclose = AsyncMock()
    return client
