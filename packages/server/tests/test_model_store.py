from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from litert_ollama.model_store import ModelStore


def test_add_and_get_model(store: ModelStore):
    store.add_model("test--model", "test/model", "/some/path", size=1000, digest="abc123", source="hf/test-model")
    model = store.get_model("test--model")
    assert model is not None
    assert model["name"] == "test/model"
    assert model["path"] == "/some/path"
    assert model["size"] == 1000
    assert model["digest"] == "abc123"
    assert model["source"] == "hf/test-model"


def test_get_model_not_found(store: ModelStore):
    assert store.get_model("nonexistent") is None


def test_list_models(store: ModelStore):
    store.add_model("model-a", "Model A", "/a")
    store.add_model("model-b", "Model B", "/b")
    models = store.list_models()
    assert len(models) == 2


def test_remove_model(store: ModelStore):
    store.add_model("to-remove", "To Remove", "/path")
    store.remove_model("to-remove")
    assert store.get_model("to-remove") is None


def test_rename_model_creates_new_entry(store: ModelStore, tmp_path: Path):
    old_dir = tmp_path / "old--model"
    old_dir.mkdir(parents=True)
    model_file = old_dir / "model.litertlm"
    model_file.write_text("dummy")

    store.add_model("old--model", "old/model", str(old_dir), source="hf/original")
    ok = store.rename_model("old--model", "new-model")
    assert ok is True

    new_dir = tmp_path / "new-model"
    assert new_dir.exists()
    assert (new_dir / "model.litertlm").read_text() == "dummy"
    assert not old_dir.exists()

    new_record = store.get_model("new-model")
    assert new_record is not None
    assert new_record["name"] == "new-model"
    assert new_record["source"] == "hf/original"

    old_record = store.get_model("old--model")
    assert old_record is None


def test_rename_model_not_found(store: ModelStore):
    ok = store.rename_model("nonexistent", "new-name")
    assert ok is False


def test_rename_sanitizes_invalid_chars(store: ModelStore, tmp_path: Path):
    old_dir = tmp_path / "valid-model"
    old_dir.mkdir(parents=True)
    (old_dir / "model.litertlm").write_text("data")
    store.add_model("valid-model", "valid/model", str(old_dir))

    ok = store.rename_model("valid-model", "bad:name<with>chars")
    assert ok is True

    new_dir = tmp_path / "bad-name-with-chars"
    assert new_dir.exists()
    record = store.get_model("bad-name-with-chars")
    assert record is not None


def test_find_by_source(store: ModelStore):
    store.add_model("model-a", "Model A", "/a", source="hf/repo-a")
    store.add_model("model-b", "Model B", "/b", source="hf/repo-b")

    found = store.find_by_source("hf/repo-a")
    assert found is not None
    assert found["id"] == "model-a"

    not_found = store.find_by_source("hf/nonexistent")
    assert not_found is None


def test_find_by_name(store: ModelStore):
    store.add_model("test--id", "my-model", "/p")
    found = store.find_by_name("my-model")
    assert found is not None
    assert found["id"] == "test--id"

    not_found = store.find_by_name("unknown")
    assert not_found is None


def test_source_migration_on_existing_db(tmp_path: Path):
    db_file = tmp_path / "old.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute("""
        CREATE TABLE models (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            path TEXT NOT NULL,
            backend_constraint TEXT DEFAULT '',
            size INTEGER DEFAULT 0,
            digest TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now')),
            modified_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("INSERT INTO models (id, name, path) VALUES ('m1', 'M1', '/p')")
    conn.commit()
    conn.close()

    store = ModelStore(db_path=str(db_file))
    store._init_db()

    model = store.get_model("m1")
    assert model is not None
    assert "source" in model
    assert model["source"] == ""


def test_add_model_preserves_existing_source(store: ModelStore):
    store.add_model("m1", "M1", "/p", source="hf/original")
    store.add_model("m1", "M1-renamed", "/p", source="hf/different")
    model = store.get_model("m1")
    assert model is not None
    assert model["source"] == "hf/original"
    assert model["name"] == "M1-renamed"
