from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from litert_ollama.engine_manager import registry


@pytest.mark.asyncio
async def test_discover_models_empty():
    models = registry.discover_models()
    assert isinstance(models, list)


def test_parse_keep_alive_default():
    assert registry._parse_keep_alive("5m") == 300
    assert registry._parse_keep_alive("10s") == 10
    assert registry._parse_keep_alive("1h") == 3600
    assert registry._parse_keep_alive("0") == 0


@pytest.mark.asyncio
async def test_get_loaded_models_empty():
    loaded = registry.get_loaded_models()
    assert isinstance(loaded, list)


@pytest.mark.asyncio
async def test_load_engine_not_found():
    try:
        await registry.load_engine("nonexistent-model")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass
