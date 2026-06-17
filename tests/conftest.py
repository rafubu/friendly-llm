from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
try:
    import pytest_asyncio
except ImportError:
    pytest_asyncio = None


class MockEngine:
    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or ["Mock response"]
        self._response_index = 0
        self.tokenized = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def tokenize(self, text: str) -> list[int]:
        self.tokenized = True
        return [1, 2, 3, 4, 5]

    def detokenize(self, ids: list[int]) -> str:
        return "detokenized text"

    def create_conversation(self, **kwargs) -> MagicMock:
        conv = MagicMock()

        def send_message(msg: Any) -> dict:
            resp_index = self._response_index % len(self.responses)
            self._response_index += 1
            return {"role": "assistant", "content": [{"type": "text", "text": self.responses[resp_index]}]}

        def send_message_async(msg: Any):
            resp_index = self._response_index % len(self.responses)
            self._response_index += 1
            yield {"role": "assistant", "content": [{"type": "text", "text": self.responses[resp_index]}]}

        conv.send_message = MagicMock(side_effect=send_message)
        conv.send_message_async = MagicMock(side_effect=send_message_async)
        conv.__enter__ = MagicMock(return_value=conv)
        conv.__exit__ = MagicMock(return_value=None)
        conv.cancel_process = MagicMock()
        conv.token_count = 10

        return conv

    def create_session(self, **kwargs) -> MagicMock:
        session = MagicMock()
        session.run_prefill = MagicMock()
        session.run_decode_async = MagicMock(return_value=[])
        session.run_text_scoring = MagicMock(return_value=MagicMock(texts=[""], scores=[0.0]))
        session.__enter__ = MagicMock(return_value=session)
        session.__exit__ = MagicMock(return_value=None)
        return session


@pytest.fixture
def mock_engine():
    return MockEngine()


if pytest_asyncio:
    @pytest_asyncio.fixture
    async def app_client():
        from litert_ollama.app import app
        from httpx import AsyncClient, ASGITransport

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client
