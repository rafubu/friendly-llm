from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from litert_ollama.schemas import (
    ChatRequest, GenerateRequest, ChatChunk, GenerateChunk,
    Message,
)


class TestContextOverflowChat:
    """Tests for context overflow detection in /api/chat."""

    def test_normal_request_passes_preflight(self):
        """Short messages should pass pre-flight and be processed."""
        req = ChatRequest(
            model="test-model",
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there"),
            ],
        )
        from litert_ollama.routers.chat import _estimate_message_tokens
        tokens = _estimate_message_tokens([m.model_dump() for m in req.messages])
        assert tokens < 32768, f"Expected short conversation to fit, got {tokens}"

    def test_overflow_detected_for_huge_input(self):
        """Very long messages should trigger context overflow."""
        huge_text = "x" * 140000  # ~35K tokens
        req = ChatRequest(
            model="test-model",
            messages=[Message(role="user", content=huge_text)],
        )
        from litert_ollama.routers.chat import _estimate_message_tokens
        tokens = _estimate_message_tokens([m.model_dump() for m in req.messages])
        assert tokens > 32768, f"Expected overflow, got {tokens}"

    def test_overflow_with_num_ctx_override(self):
        """Setting options.num_ctx should use custom limit."""
        text = "y" * 40000  # ~10K tokens
        req = ChatRequest(
            model="test-model",
            messages=[Message(role="user", content=text)],
            options={"num_ctx": 4096},
        )
        from litert_ollama.routers.chat import _estimate_message_tokens
        tokens = _estimate_message_tokens([m.model_dump() for m in req.messages])
        assert tokens > 4096, f"Expected overflow with custom limit, got {tokens}"

    def test_multimodal_content_counted(self):
        """Text in multimodal content lists should be counted."""
        from litert_ollama.routers.chat import _estimate_message_tokens
        tokens = _estimate_message_tokens([
            {"role": "user", "content": [
                {"type": "text", "text": "What is in this image?" * 20},
                {"type": "image", "data": "base64data..."},
            ]}
        ])
        assert tokens > 8  # At least some tokens counted

    def test_empty_messages_returns_minimal(self):
        """Empty messages should still count some overhead."""
        from litert_ollama.routers.chat import _estimate_message_tokens
        tokens = _estimate_message_tokens([{"role": "user", "content": ""}])
        assert tokens >= 4, "Should count at least role overhead"


class TestContextOverflowGenerate:
    """Tests for context overflow detection in /api/generate."""

    def test_normal_prompt_passes(self):
        """Short prompt should fit easily."""
        from litert_ollama.routers.generate import _estimate_prompt_tokens
        tokens = _estimate_prompt_tokens("Hello world")
        assert tokens < 1000

    def test_huge_prompt_overflows(self):
        """Massive prompt should overflow."""
        from litert_ollama.routers.generate import _estimate_prompt_tokens
        huge = "abc " * 200000  # ~50K tokens
        tokens = _estimate_prompt_tokens(huge)
        assert tokens > 32768

    def test_dict_prompt_counted(self):
        """Dict-format prompts should be token-counted."""
        from litert_ollama.routers.generate import _estimate_prompt_tokens
        prompt = {"role": "user", "content": [{"type": "text", "text": "Hello world"}]}
        tokens = _estimate_prompt_tokens(prompt)
        assert tokens > 0


class TestChatChunkContextFields:
    """Verify ChatChunk has context fields."""

    def test_chunk_has_context_fields(self):
        chunk = ChatChunk(
            model="test",
            created_at="2026-01-01T00:00:00Z",
            context_used=1500,
            context_limit=32768,
        )
        assert chunk.context_used == 1500
        assert chunk.context_limit == 32768

    def test_chunk_context_info_for_overflow(self):
        chunk = ChatChunk(
            model="test",
            created_at="2026-01-01T00:00:00Z",
            done=True,
            done_reason="context_overflow",
            context_info={
                "estimated_input_tokens": 40000,
                "context_limit": 32768,
                "overflow_by": 7232,
                "suggestion": "Reduce message count",
            },
        )
        assert chunk.done_reason == "context_overflow"
        assert chunk.context_info["overflow_by"] == 7232

    def test_generate_chunk_has_context_fields(self):
        chunk = GenerateChunk(
            model="test",
            created_at="2026-01-01T00:00:00Z",
            context_used=2500,
            context_limit=8192,
        )
        assert chunk.context_used == 2500
        assert chunk.context_limit == 8192


class TestContextLimitPassthrough:
    """Verify context_limit flows through to load_engine."""

    @pytest.mark.asyncio
    async def test_load_engine_accepts_max_num_tokens(self):
        """load_engine should accept and use max_num_tokens parameter."""
        # This tests the signature change only — no engine needed
        import inspect
        from litert_ollama.engine_manager import ModelRegistry
        registry = ModelRegistry()
        sig = inspect.signature(registry.load_engine)
        params = sig.parameters
        assert "max_num_tokens" in params, "load_engine should accept max_num_tokens"
