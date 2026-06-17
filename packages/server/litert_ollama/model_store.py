from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .config import settings


class ModelStore:
    _instance: ModelStore | None = None
    _lock = threading.Lock()

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            db_path = str(Path(settings.models_dir).parent / "litert-ollama.db")
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    @classmethod
    def get_instance(cls) -> ModelStore:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_path)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA foreign_keys=ON")
        return self._local.conn

    def _init_db(self):
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                path TEXT NOT NULL,
                backend_constraint TEXT DEFAULT '',
                size INTEGER DEFAULT 0,
                digest TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now')),
                modified_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS modelfiles (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                base_model TEXT NOT NULL,
                parameters TEXT DEFAULT '{}',
                system_prompt TEXT DEFAULT '',
                template TEXT DEFAULT '',
                messages TEXT DEFAULT '[]',
                license_text TEXT DEFAULT '',
                adapter TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (base_model) REFERENCES models(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key_hash TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                created_at TEXT DEFAULT (datetime('now')),
                disabled INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT DEFAULT '',
                model_id TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                message_count INTEGER DEFAULT 0,
                token_count INTEGER DEFAULT 0
            );
        """)
        conn.commit()
        conn.close()

    def add_model(self, model_id: str, name: str, path: str, size: int = 0, digest: str = ""):
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO models (id, name, path, size, digest, modified_at) VALUES (?, ?, ?, ?, ?, datetime('now'))",
            (model_id, name, path, size, digest),
        )
        conn.commit()

    def remove_model(self, model_id: str):
        conn = self._get_conn()
        conn.execute("DELETE FROM modelfiles WHERE base_model = ?", (model_id,))
        conn.execute("DELETE FROM models WHERE id = ?", (model_id,))
        conn.commit()

    def get_model(self, model_id: str) -> dict[str, Any] | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
        if row:
            return dict(row)
        return None

    def list_models(self) -> list[dict[str, Any]]:
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM models ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def add_modelfile(self, name: str, spec: dict[str, Any]) -> str:
        conn = self._get_conn()
        mf_id = f"mf_{name}"
        conn.execute(
            """INSERT OR REPLACE INTO modelfiles
               (id, name, base_model, parameters, system_prompt, template, messages, license_text, adapter)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                mf_id,
                name,
                spec.get("from_model", ""),
                json.dumps(spec.get("parameters", {})),
                spec.get("system", ""),
                spec.get("template", ""),
                json.dumps(spec.get("messages", [])),
                spec.get("license", ""),
                spec.get("adapter", ""),
            ),
        )
        conn.commit()
        return mf_id

    def list_modelfiles(self) -> list[dict[str, Any]]:
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM modelfiles ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def get_modelfile(self, name: str) -> dict[str, Any] | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM modelfiles WHERE name = ? OR id = ?", (name, name)
        ).fetchone()
        if row:
            return dict(row)
        return None

    def delete_modelfile(self, name: str):
        conn = self._get_conn()
        conn.execute("DELETE FROM modelfiles WHERE name = ? OR id = ?", (name, name))
        conn.commit()
