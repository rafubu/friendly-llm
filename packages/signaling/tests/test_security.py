from __future__ import annotations

import sqlite3
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litert_signaling.app import (
    RateLimiter,
    HeartbeatMonitor,
    _check_jwt_blacklist,
    _audit_log,
)
from litert_signaling.config import settings


def test_health_endpoint_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_health_endpoint_reports_node_count(client, node_api_key):
    with client.websocket_connect("/signal") as ws:
        ws.send_json({"type": "auth", "api_key": node_api_key})
        ws.receive_json()
        ws.send_json({"type": "register", "name": "health-node", "models": [{"name": "m"}]})
        ws.receive_json()

        resp = client.get("/health")
        data = resp.json()
        assert data["nodes"] >= 1


class TestRateLimiter:
    def test_message_limited(self):
        limiter = RateLimiter()
        ws = MagicMock()
        original = settings.rate_limit_per_minute
        settings.rate_limit_per_minute = 3
        try:
            assert limiter.check_message(ws) is True
            assert limiter.check_message(ws) is True
            assert limiter.check_message(ws) is True
            assert limiter.check_message(ws) is False
        finally:
            settings.rate_limit_per_minute = original

    def test_message_window_resets(self):
        limiter = RateLimiter()
        ws = MagicMock()
        original = settings.rate_limit_per_minute
        settings.rate_limit_per_minute = 2
        try:
            assert limiter.check_message(ws) is True
            assert limiter.check_message(ws) is True
            assert limiter.check_message(ws) is False

            limiter._conn_msgs[id(ws)] = []
            assert limiter.check_message(ws) is True
        finally:
            settings.rate_limit_per_minute = original

    def test_cleanup(self):
        limiter = RateLimiter()
        ws = MagicMock()
        limiter.check_message(ws)
        assert id(ws) in limiter._conn_msgs
        limiter.cleanup_connection(ws)
        assert id(ws) not in limiter._conn_msgs

    def test_connection_limited_per_ip(self):
        limiter = RateLimiter()
        original = settings.max_connections_per_ip
        settings.max_connections_per_ip = 2
        try:
            assert limiter.check_connection("1.2.3.4") is True
            assert limiter.check_connection("1.2.3.4") is True
            assert limiter.check_connection("1.2.3.4") is False
            assert limiter.check_connection("5.6.7.8") is True
        finally:
            settings.max_connections_per_ip = original


class TestHeartbeat:
    def test_register_and_unregister(self):
        hb = HeartbeatMonitor()
        ws = MagicMock()
        hb.register(ws)
        assert id(ws) in hb._connections
        hb.unregister(ws)
        assert id(ws) not in hb._connections

    def test_refresh_updates_last_seen(self):
        hb = HeartbeatMonitor()
        ws = MagicMock()
        hb.register(ws)
        old = hb._last_seen[id(ws)]
        time.sleep(0.01)
        hb.refresh(ws)
        assert hb._last_seen[id(ws)] > old

    def test_stale_connection_gets_closed(self):
        hb = HeartbeatMonitor()
        ws = AsyncMock()
        hb.register(ws)
        hb._last_seen[id(ws)] = time.time() - 60
        original_timeout = settings.heartbeat_timeout
        settings.heartbeat_timeout = 30
        try:
            import asyncio
            asyncio.get_event_loop().run_until_complete(hb._check_stale())
            ws.close.assert_called_once()
        finally:
            settings.heartbeat_timeout = original_timeout


class TestJwtBlacklist:
    def test_blacklist_revoked_token(self, initialized_db):
        conn = sqlite3.connect(str(initialized_db))
        conn.execute("INSERT INTO token_blacklist (jti) VALUES ('revoked-jti-123')")
        conn.commit()
        conn.close()

        assert _check_jwt_blacklist("revoked-jti-123") is True

    def test_blacklist_valid_token(self, initialized_db):
        assert _check_jwt_blacklist("unknown-jti") is False

    def test_blacklist_nonexistent_table(self, initialized_db):
        conn = sqlite3.connect(str(initialized_db))
        conn.execute("DROP TABLE IF EXISTS token_blacklist")
        conn.commit()
        conn.close()
        assert _check_jwt_blacklist("anything") is False


class TestAuditLog:
    def test_audit_log_writes_to_db(self, initialized_db):
        _audit_log("test_action", "test details", ip="1.2.3.4", user_id="user1")

        conn = sqlite3.connect(str(initialized_db))
        rows = conn.execute(
            "SELECT user_id, action, details, ip FROM audit_log WHERE action = 'test_action'"
        ).fetchall()
        conn.close()
        assert len(rows) >= 1
        assert rows[0][0] == "user1"
        assert rows[0][1] == "test_action"
        assert rows[0][3] == "1.2.3.4"
