from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
from pathlib import Path
from typing import Generator
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient


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
def db_path(tmp_home: Path) -> Path:
    return tmp_home / ".litert-signaling" / "signaling.db"


@pytest.fixture
def app_and_settings(db_path: Path):
    from litert_signaling.config import settings
    from litert_signaling.app import app, node_registry
    settings.db_path = str(db_path)
    settings.jwt_secret = "test-secret-that-is-32-bytes-long!!"
    node_registry._nodes.clear()
    node_registry._rooms.clear()
    return app, settings


@pytest.fixture
def client(app_and_settings) -> Generator[TestClient, None, None]:
    app, _ = app_and_settings
    with TestClient(app) as c:
        yield c


@pytest.fixture
def initialized_db(db_path: Path):
    from litert_signaling.config import settings
    from litert_signaling.app import _init_db
    settings.db_path = str(db_path)
    _init_db()
    return db_path


@pytest.fixture
def admin_token(client: TestClient, initialized_db: Path) -> str:
    import bcrypt as _bcrypt
    import sqlite3
    import uuid

    conn = sqlite3.connect(str(initialized_db))
    conn.execute("UPDATE users SET password_hash = ?, role = 'admin' WHERE email = 'admin'",
                 (_bcrypt.hashpw(b"admin123", _bcrypt.gensalt()).decode(),))
    conn.commit()
    conn.close()

    resp = client.post("/auth/login", json={"email": "admin", "password": "admin123"})
    assert resp.status_code == 200, f"Admin login failed: {resp.text}"
    return resp.json()["access_token"]


@pytest.fixture
def client_jwt(client: TestClient, initialized_db: Path) -> str:
    """A valid JWT token for a regular client."""
    import bcrypt as _bcrypt
    import sqlite3
    import uuid

    conn = sqlite3.connect(str(initialized_db))
    pw_hash = _bcrypt.hashpw(b"testpass", _bcrypt.gensalt()).decode()
    conn.execute("INSERT OR IGNORE INTO invites (code, created_by) VALUES ('TESTCODE', 'test')")
    conn.execute("INSERT INTO users (id, email, password_hash, role) VALUES (?, ?, ?, 'user')",
                 (uuid.uuid4().hex, "client@test.com", pw_hash))
    conn.commit()
    conn.close()

    resp = client.post("/auth/login", json={"email": "client@test.com", "password": "testpass"})
    assert resp.status_code == 200, f"Client login failed: {resp.text}"
    return resp.json()["access_token"]


@pytest.fixture
def node_api_key(initialized_db: Path) -> str:
    import sqlite3
    import hashlib
    api_key = f"sk-{secrets.token_hex(16)}"
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    conn = sqlite3.connect(str(initialized_db))
    conn.execute("INSERT OR IGNORE INTO api_keys (id, key_hash, name, role) VALUES (?, ?, ?, 'node')",
                 (secrets.token_hex(8), key_hash, "test-node"))
    conn.commit()
    conn.close()
    return api_key


@pytest.fixture
def invite_code(initialized_db: Path) -> str:
    import sqlite3
    code = secrets.token_hex(6).upper()
    conn = sqlite3.connect(str(initialized_db))
    conn.execute("INSERT INTO invites (code, created_by) VALUES (?, 'test')", (code,))
    conn.commit()
    conn.close()
    return code


@pytest.fixture
def user_token(client: TestClient, initialized_db: Path, invite_code: str) -> str:
    resp = client.post("/auth/register", json={
        "email": "test@example.com",
        "password": "secure123",
        "invite_code": invite_code,
    })
    assert resp.status_code == 200
    resp = client.post("/auth/login", json={"email": "test@example.com", "password": "secure123"})
    assert resp.status_code == 200
    return resp.json()["access_token"]
