from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient


@pytest.fixture
def mock_hf_model_info():
    with patch("huggingface_hub.HfApi") as mock:
        instance = mock.return_value
        repo_file = MagicMock()
        repo_file.rfilename = "model.litertlm"
        repo_file.size = 1_000_000_000
        model_info = MagicMock()
        model_info.siblings = [repo_file]
        instance.model_info.return_value = model_info
        yield instance


@contextmanager
def mock_httpx_download():
    """Mock httpx.AsyncClient to simulate a successful streaming download."""

    async def _aiter_bytes():
        for _ in range(5):
            yield b"x" * 65536
        yield b"x" * 65536

    response = AsyncMock()
    response.headers = {"content-length": "393216"}
    response.aiter_bytes = _aiter_bytes

    resp_cm = AsyncMock()
    resp_cm.__aenter__ = AsyncMock(return_value=response)
    resp_cm.__aexit__ = AsyncMock(return_value=None)

    client_instance = AsyncMock()
    client_instance.stream = MagicMock(return_value=resp_cm)

    client_cm = AsyncMock()
    client_cm.__aenter__ = AsyncMock(return_value=client_instance)
    client_cm.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=client_cm):
        yield client_instance


async def test_pull_missing_model(client: AsyncClient):
    resp = await client.post("/api/pull", json={})
    assert resp.status_code == 400


async def test_pull_missing_model_field(client: AsyncClient):
    resp = await client.post("/api/pull", json={"model": ""})
    assert resp.status_code == 400


async def test_pull_repo_not_found(client: AsyncClient, models_dir):
    with patch("huggingface_hub.HfApi") as mock:
        from huggingface_hub.utils import RepositoryNotFoundError
        import httpx as _httpx
        instance = mock.return_value
        resp_obj = MagicMock(spec=_httpx.Response)
        resp_obj.status_code = 404
        resp_obj.headers = {}
        resp_obj.json.return_value = {"error": "Not found"}
        instance.model_info.side_effect = RepositoryNotFoundError("Not found", response=resp_obj)

        resp = await client.post("/api/pull", json={"model": "unknown/model"})
        assert resp.status_code == 200
        lines = resp.text.strip().split("\n")
        last = json.loads(lines[-1])
        assert last["status"] == "error"


async def test_pull_streams_progress(client: AsyncClient, models_dir, mock_hf_model_info):
    with mock_httpx_download() as client_instance:
        resp = await client.post("/api/pull", json={"model": "test-model"})

    assert resp.status_code == 200
    lines = resp.text.strip().split("\n")
    assert len(lines) >= 2

    first = json.loads(lines[0])
    assert first["status"] == "pulling manifest"

    progress_lines = [l for l in lines if "total" in json.loads(l) and json.loads(l).get("total", 0) > 0]
    if progress_lines:
        pl = json.loads(progress_lines[0])
        assert "digest" in pl
        assert pl["total"] > 0
        assert pl["completed"] >= 0


async def test_pull_success_status(client: AsyncClient, models_dir, mock_hf_model_info):
    with mock_httpx_download():
        resp = await client.post("/api/pull", json={"model": "test-model"})

    assert resp.status_code == 200
    lines = resp.text.strip().split("\n")
    for line in lines:
        data = json.loads(line)
        if data["status"] == "error":
            print("ERROR:", data.get("error"))
    last = json.loads(lines[-1])
    assert last["status"] == "success"


async def test_pull_hf_token_forwarded(client: AsyncClient, models_dir):
    with patch("huggingface_hub.HfApi") as mock:
        instance = mock.return_value
        repo_file = MagicMock()
        repo_file.rfilename = "model.litertlm"
        repo_file.size = 100
        instance.model_info.return_value = MagicMock(siblings=[repo_file])

        with mock_httpx_download() as client_instance:
            resp = await client.post("/api/pull", json={
                "model": "gated/model",
                "huggingface_token": "hf_test123",
            })

        assert resp.status_code == 200
        assert instance.token == "hf_test123"
        call_headers = client_instance.stream.call_args[1].get("headers", {})
        assert call_headers.get("Authorization") == "Bearer hf_test123"


async def test_pull_without_slash_uses_default_repo(client: AsyncClient, models_dir):
    with patch("huggingface_hub.HfApi") as mock:
        instance = mock.return_value
        repo_file = MagicMock()
        repo_file.rfilename = "model.litertlm"
        repo_file.size = 100
        instance.model_info.return_value = MagicMock(siblings=[repo_file])

        with mock_httpx_download():
            resp = await client.post("/api/pull", json={"model": "gemma-4"})

        assert resp.status_code == 200
        called_repo = instance.model_info.call_args[0][0]
        assert "litert-community" in called_repo
