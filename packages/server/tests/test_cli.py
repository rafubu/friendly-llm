from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_rename_found_by_db_id(store, models_dir: Path, capsys):
    from litert_ollama.cli import _run_rename
    from litert_ollama.config import settings as cfg

    old_dir = models_dir / "old--model"
    old_dir.mkdir(parents=True)
    (old_dir / "model.litertlm").write_text("data")
    store.add_model("old--model", "old/model", str(old_dir), source="hf/original")

    old_mdir = cfg.models_dir
    cfg.models_dir = str(models_dir)
    try:
        with patch("litert_ollama.model_store.ModelStore.get_instance", return_value=store):
            args = MagicMock()
            args.model = "old/model"
            args.new_name = "new-model"
            _run_rename(args)
    finally:
        cfg.models_dir = old_mdir

    captured = capsys.readouterr()
    assert "Renamed to" in captured.out

    new_dir = models_dir / "new-model"
    assert new_dir.exists()
    assert not old_dir.exists()
    new_record = store.get_model("new-model")
    assert new_record is not None
    assert new_record["name"] == "new-model"
    assert new_record["source"] == "hf/original"


def test_rename_found_by_name(store, models_dir: Path, capsys):
    from litert_ollama.cli import _run_rename
    from litert_ollama.config import settings as cfg

    old_dir = models_dir / "my-model"
    old_dir.mkdir(parents=True)
    (old_dir / "model.litertlm").write_text("data")
    store.add_model("my-model", "display-name", str(old_dir))

    old_mdir = cfg.models_dir
    cfg.models_dir = str(models_dir)
    try:
        with patch("litert_ollama.model_store.ModelStore.get_instance", return_value=store):
            args = MagicMock()
            args.model = "display-name"
            args.new_name = "renamed"
            _run_rename(args)
    finally:
        cfg.models_dir = old_mdir

    captured = capsys.readouterr()
    assert "Renamed to" in captured.out
    assert store.get_model("renamed") is not None


def test_rename_not_found(store, capsys):
    from litert_ollama.cli import _run_rename

    with patch("litert_ollama.model_store.ModelStore.get_instance", return_value=store):
        args = MagicMock()
        args.model = "nonexistent"
        args.new_name = "new-name"
        _run_rename(args)

    captured = capsys.readouterr()
    assert "not found" in captured.out


def test_rename_sanitizes_colon(store, models_dir: Path, capsys):
    from litert_ollama.cli import _run_rename
    from litert_ollama.config import settings as cfg

    old_dir = models_dir / "valid-model"
    old_dir.mkdir(parents=True)
    (old_dir / "model.litertlm").write_text("data")
    store.add_model("valid-model", "valid/model", str(old_dir))

    old_mdir = cfg.models_dir
    cfg.models_dir = str(models_dir)
    try:
        with patch("litert_ollama.model_store.ModelStore.get_instance", return_value=store):
            args = MagicMock()
            args.model = "valid/model"
            args.new_name = "model:42b"
            _run_rename(args)
    finally:
        cfg.models_dir = old_mdir

    captured = capsys.readouterr()
    assert "not allowed" in captured.out
    assert store.get_model("model-42b") is not None


def test_rename_fallback_to_filesystem(store, models_dir: Path, capsys):
    from litert_ollama.cli import _run_rename
    from litert_ollama.config import settings as cfg

    old_dir = models_dir / "fs-only--model"
    old_dir.mkdir(parents=True)
    (old_dir / "model.litertlm").write_text("data")

    old_mdir = cfg.models_dir
    cfg.models_dir = str(models_dir)
    try:
        with patch("litert_ollama.model_store.ModelStore.get_instance", return_value=store):
            args = MagicMock()
            args.model = "fs-only/model"
            args.new_name = "from-fs"
            _run_rename(args)
    finally:
        cfg.models_dir = old_mdir

    captured = capsys.readouterr()
    assert "Renamed to" in captured.out

    new_dir = models_dir / "from-fs"
    assert new_dir.exists()


def test_pull_skips_if_source_exists(store, models_dir, capsys):
    from litert_ollama.cli import _run_pull

    store.add_model("existing--model", "existing/model", str(models_dir / "existing--model"), source="litert-community/existing-model")

    with patch("litert_ollama.model_store.ModelStore.get_instance", return_value=store):
        args = MagicMock()
        args.model = "litert-community/existing-model"
        args.hf_token = None
        _run_pull(args)

    captured = capsys.readouterr()
    assert "already exists" in captured.out


def test_pull_no_download_if_source_exists(store, models_dir, capsys):
    from litert_ollama.cli import _run_pull

    store.add_model("existing--model", "existing/model", str(models_dir / "existing--model"), source="litert-community/existing-model")

    with patch("litert_ollama.model_store.ModelStore.get_instance", return_value=store):
        args = MagicMock()
        args.model = "litert-community/existing-model"
        args.hf_token = None
        _run_pull(args)

    captured = capsys.readouterr()
    assert "already exists" in captured.out
