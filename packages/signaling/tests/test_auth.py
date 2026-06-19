from __future__ import annotations

import secrets
import sqlite3
import uuid

import bcrypt as _bcrypt
import pytest


def test_login_admin_default(initialized_db):
    import sqlite3
    conn = sqlite3.connect(str(initialized_db))
    row = conn.execute("SELECT id FROM users WHERE email = 'admin'").fetchone()
    conn.close()
    assert row is not None


def test_login_invalid_credentials(client, initialized_db):
    resp = client.post("/auth/login", json={"email": "admin", "password": "wrong"})
    assert resp.status_code == 401


def test_login_nonexistent_user(client, initialized_db):
    resp = client.post("/auth/login", json={"email": "nobody@test.com", "password": "x"})
    assert resp.status_code == 401


def test_login_returns_tokens(client, initialized_db):
    conn = sqlite3.connect(str(initialized_db))
    pw_hash = _bcrypt.hashpw(b"testpass", _bcrypt.gensalt()).decode()
    conn.execute("INSERT INTO users (id, email, password_hash, role) VALUES (?, ?, ?, 'user')",
                 (uuid.uuid4().hex, "user@test.com", pw_hash))
    conn.commit()
    conn.close()

    resp = client.post("/auth/login", json={"email": "user@test.com", "password": "testpass"})
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"
    assert data["expires_in"] == 3600


def test_login_case_sensitive(client, initialized_db):
    conn = sqlite3.connect(str(initialized_db))
    pw_hash = _bcrypt.hashpw(b"TestPass", _bcrypt.gensalt()).decode()
    conn.execute("INSERT INTO users (id, email, password_hash, role) VALUES (?, ?, ?, 'user')",
                 (uuid.uuid4().hex, "Case@test.com", pw_hash))
    conn.commit()
    conn.close()

    resp = client.post("/auth/login", json={"email": "Case@test.com", "password": "TestPass"})
    assert resp.status_code == 200
    resp2 = client.post("/auth/login", json={"email": "case@test.com", "password": "TestPass"})
    assert resp2.status_code == 401


def test_register_with_valid_invite(client, initialized_db, invite_code):
    resp = client.post("/auth/register", json={
        "email": "newuser@test.com",
        "password": "StrongPass1!",
        "invite_code": invite_code,
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "registered"


def test_register_with_invalid_invite(client, initialized_db):
    resp = client.post("/auth/register", json={
        "email": "newuser@test.com",
        "password": "StrongPass1!",
        "invite_code": "INVALID",
    })
    assert resp.status_code == 400


def test_register_with_used_invite(client, initialized_db, invite_code):
    client.post("/auth/register", json={
        "email": "first@test.com", "password": "x",
        "invite_code": invite_code,
    })
    resp = client.post("/auth/register", json={
        "email": "second@test.com", "password": "x",
        "invite_code": invite_code,
    })
    assert resp.status_code == 400


def test_register_duplicate_email(client, initialized_db, invite_code):
    code2 = secrets.token_hex(6).upper()
    conn = sqlite3.connect(str(initialized_db))
    conn.execute("INSERT INTO invites (code, created_by) VALUES (?, 'test')", (code2,))
    conn.commit()
    conn.close()

    client.post("/auth/register", json={
        "email": "dup@test.com", "password": "x",
        "invite_code": invite_code,
    })
    resp = client.post("/auth/register", json={
        "email": "dup@test.com", "password": "x",
        "invite_code": code2,
    })
    assert resp.status_code == 409


def test_jwt_token_works_for_websocket_auth(client, initialized_db):
    """Verify the JWT produced by login works for WebSocket auth."""
    conn = sqlite3.connect(str(initialized_db))
    pw_hash = _bcrypt.hashpw(b"testpass", _bcrypt.gensalt()).decode()
    conn.execute("INSERT INTO users (id, email, password_hash, role) VALUES (?, ?, ?, 'user')",
                 (uuid.uuid4().hex, "jwtuser@test.com", pw_hash))
    conn.commit()
    conn.close()

    resp = client.post("/auth/login", json={"email": "jwtuser@test.com", "password": "testpass"})
    token = resp.json()["access_token"]

    with client.websocket_connect("/signal") as ws:
        ws.send_json({"type": "auth_jwt", "token": token})
        resp = ws.receive_json()
        assert resp["type"] == "auth_ok"
        assert "user_id" in resp


def test_refresh_token_different_from_access(client, initialized_db):
    conn = sqlite3.connect(str(initialized_db))
    pw_hash = _bcrypt.hashpw(b"pass", _bcrypt.gensalt()).decode()
    conn.execute("INSERT INTO users (id, email, password_hash, role) VALUES (?, ?, ?, 'user')",
                 (uuid.uuid4().hex, "refresh@test.com", pw_hash))
    conn.commit()
    conn.close()

    resp = client.post("/auth/login", json={"email": "refresh@test.com", "password": "pass"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["access_token"] != data["refresh_token"]
