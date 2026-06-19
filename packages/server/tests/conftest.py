from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from litert_ollama.model_store import ModelStore


@pytest.fixture
def tmp_home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    old_home = os.environ.get("HOME") or os.environ.get("USERPROFILE")
    os.environ["HOME"] = str(home)
    os.environ["USERPROFILE"] = str(home)
    yield home
    if old_home:
        os.environ["HOME"] = old_home
        os.environ["USERPROFILE"] = old_home


@pytest.fixture
def models_dir(tmp_home: Path) -> Path:
    d = tmp_home / ".litert-lm" / "models"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def db_path(tmp_home: Path) -> Path:
    d = tmp_home / ".litert-ollama"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def store(models_dir: Path, db_path: Path) -> Generator[ModelStore, None, None]:
    ModelStore._instance = None
    store = ModelStore(db_path=str(db_path / "litert-ollama.db"))
    store._init_db()
    yield store
    ModelStore._instance = None


@pytest.fixture
def app(models_dir: Path) -> FastAPI:
    from litert_ollama.app import app
    from litert_ollama.config import settings
    settings.models_dir = str(models_dir)
    return app


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def mock_hf_api(mocker):
    mock = mocker.patch("huggingface_hub.HfApi")
    instance = mock.return_value
    instance.model_info.return_value = mocker.MagicMock()
    return instance
