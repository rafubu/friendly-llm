from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litert_sdk.local_client import LitertLocalClient
from litert_sdk.errors import ConnectionError, ContextOverflowError
from litert_sdk.types import Chunk, ModelInfo


class TestLocalClient:
    @pytest.fixture
    def client(self):
        return LitertLocalClient(base_url="http://test:11434", model="test-model")

    @pytest.fixture
    def mock_stream(self):
        """Returns [chunks_json, done_json] lines."""
        chunks = [
            json.dumps({"message": {"content": "Hello"}, "done": False}),
            json.dumps({"message": {"content": " world"}, "done": False}),
            json.dumps({"message": {"content": ""}, "done": True, "done_reason": "stop", "eval_count": 5, "total_duration": 1000}),
        ]

        async def aiter_lines():
            for line in chunks:
                yield line

        resp = AsyncMock()
        resp.aiter_lines = aiter_lines
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=resp)
        cm.__aexit__ = AsyncMock(return_value=None)
        return cm

    def test_chat_streams_tokens(self, client, mock_stream):
        client._client.stream = MagicMock(return_value=mock_stream)

        import asyncio
        results = []

        async def run():
            async for chunk in client.chat("hola"):
                results.append(chunk)

        asyncio.run(run())
        assert len(results) == 3
        assert results[0].text == "Hello"
        assert results[1].text == " world"
        assert results[2].done is True
        assert results[2].done_reason == "stop"
        assert results[2].eval_count == 5

    def test_chat_empty_response(self, client):
        async def aiter_lines():
            yield json.dumps({"message": {"content": ""}, "done": True, "done_reason": "stop"})

        resp = AsyncMock()
        resp.aiter_lines = aiter_lines
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=resp)
        cm.__aexit__ = AsyncMock(return_value=None)
        client._client.stream = MagicMock(return_value=cm)

        import asyncio
        results = []

        async def run():
            async for chunk in client.chat("hi"):
                results.append(chunk)

        asyncio.run(run())
        assert len(results) == 1
        assert results[0].done is True

    def test_chat_http_error(self, client):
        import httpx
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(side_effect=httpx.HTTPError("Connection refused"))
        cm.__aexit__ = AsyncMock(return_value=None)
        client._client.stream = MagicMock(return_value=cm)

        import asyncio

        async def run():
            with pytest.raises(ConnectionError):
                async for _ in client.chat("hi"):
                    pass

        asyncio.run(run())

    def test_chat_sync_returns_text(self, client):
        resp = MagicMock()
        resp.json.return_value = {"message": {"content": "Hello!"}, "done": True}
        client._client.post = AsyncMock(return_value=resp)

        import asyncio

        async def run():
            result = await client.chat_sync("hola")
            assert result.text == "Hello!"

        asyncio.run(run())

    def test_chat_sync_context_overflow(self, client):
        resp = MagicMock()
        resp.json.return_value = {
            "done": True,
            "done_reason": "context_overflow",
            "context_info": {
                "estimated_input_tokens": 50000,
                "context_limit": 32768,
                "overflow_by": 17232,
                "suggestion": "Reduce message count",
            },
        }
        client._client.post = AsyncMock(return_value=resp)

        import asyncio

        async def run():
            with pytest.raises(ContextOverflowError) as exc:
                await client.chat_sync("x" * 1000)
            assert exc.value.estimated_tokens == 50000
            assert exc.value.context_limit == 32768
            assert exc.value.overflow_by == 17232

        asyncio.run(run())

    def test_list_models(self, client):
        resp = MagicMock()
        resp.json.return_value = {
            "models": [
                {"name": "gemma-4-12b"},
                {"name": "gemma-4-27b"},
            ]
        }
        client._client.get = AsyncMock(return_value=resp)

        import asyncio

        async def run():
            models = await client.list_models()
            assert len(models) == 2
            assert models[0].id == "gemma-4-12b"
            assert models[1].id == "gemma-4-27b"

        asyncio.run(run())

    def test_embed(self, client):
        resp = MagicMock()
        resp.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
        client._client.post = AsyncMock(return_value=resp)

        import asyncio

        async def run():
            emb = await client.embed("test")
            assert emb == [0.1, 0.2, 0.3]

        asyncio.run(run())

    def test_generate_streams(self, client):
        chunks = [
            json.dumps({"response": "Hello", "done": False}),
            json.dumps({"response": "", "done": True, "done_reason": "stop", "eval_count": 3}),
        ]

        async def aiter_lines():
            for c in chunks:
                yield c

        resp = AsyncMock()
        resp.aiter_lines = aiter_lines
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=resp)
        cm.__aexit__ = AsyncMock(return_value=None)
        client._client.stream = MagicMock(return_value=cm)

        import asyncio
        results = []

        async def run():
            async for chunk in client.generate("tell me"):
                results.append(chunk)

        asyncio.run(run())
        assert len(results) == 2
        assert results[0].text == "Hello"
        assert results[1].done is True


class TestLitertChatCLI:
    def test_help(self):
        from litert_sdk.cli import main

        with pytest.raises(SystemExit):
            try:
                with patch("sys.argv", ["litert-chat", "--help"]):
                    main()
            except SystemExit as e:
                assert e.code == 0
                raise

    def test_default_local(self):
        """Without any args but with prompt, should default to localhost."""
        from litert_sdk.cli import main, _run_local

        with patch("litert_sdk.cli.asyncio.run") as mock_run:
            with patch("sys.argv", ["litert-chat", "hello"]):
                try:
                    main()
                except SystemExit:
                    pass
            assert mock_run.called

    def test_local_mode(self):
        from litert_sdk.cli import main

        with patch("litert_sdk.cli.asyncio.run") as mock_run:
            with patch("sys.argv", ["litert-chat", "--local", "hello"]):
                try:
                    main()
                except SystemExit:
                    pass
            assert mock_run.called
