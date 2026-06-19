"""End-to-end local test for signaling control plane.

Tests the full signaling protocol flow without WebRTC (aiortc is not needed):
  node auth → register → client auth → list_nodes → create_room → SDP relay → close_room

Uses Starlette TestClient (no subprocesses, no real server port).
"""

from __future__ import annotations

import json

import pytest


def test_e2e_signaling_flow(client, node_api_key, client_jwt):
    """Full signaling control plane: node + client over WebSocket."""

    # ── 1. Node connects and authenticates ──
    with client.websocket_connect("/signal") as node_ws:
        node_ws.send_json({"type": "auth", "api_key": node_api_key})
        resp = node_ws.receive_json()
        assert resp["type"] == "auth_ok", f"Node auth failed: {resp}"
        node_id = resp["node_id"]
        assert node_id is not None

        # ── 2. Node registers its models ──
        node_ws.send_json({
            "type": "register",
            "name": "e2e-node",
            "models": [
                {"name": "gemma-4-12b", "max_sessions": 3},
                {"name": "gemma-4-27b", "max_sessions": 2},
            ],
            "visibility": "public",
            "max_sessions": 5,
        })
        resp = node_ws.receive_json()
        assert resp["type"] == "registered", f"Node register failed: {resp}"

        # ── 3. Client connects and authenticates ──
        with client.websocket_connect("/signal") as client_ws:
            client_ws.send_json({"type": "auth_jwt", "token": client_jwt})
            resp = client_ws.receive_json()
            assert resp["type"] == "auth_ok", f"Client auth failed: {resp}"
            user_id = resp.get("user_id")
            assert user_id is not None

            # ── 4. Client lists available nodes ──
            client_ws.send_json({"type": "list_nodes"})
            resp = client_ws.receive_json()
            assert resp["type"] == "node_list"
            assert len(resp["nodes"]) >= 1, "No nodes listed"

            models_available = {n["model"] for n in resp["nodes"]}
            assert "gemma-4-12b" in models_available, f"Expected gemma-4-12b, got {models_available}"
            assert "gemma-4-27b" in models_available, f"Expected gemma-4-27b, got {models_available}"

            # ── 5. Client creates a room ──
            client_ws.send_json({"type": "create_room", "model": "gemma-4-12b@e2e-node"})
            resp = client_ws.receive_json()
            assert resp["type"] == "room_created", f"Room creation failed: {resp}"
            room_id = resp["room_id"]
            node_name = resp["node"]
            assert node_name == "e2e-node"

            # ── 6. Node receives new_room notification ──
            node_notif = node_ws.receive_json()
            assert node_notif["type"] == "new_room"
            assert node_notif["room_id"] == room_id
            assert node_notif["model"] == "gemma-4-12b"

            # ── 7. Client sends SDP offer (relayed through signaling) ──
            fake_offer = {
                "type": "offer",
                "sdp": "v=0\no=client 0 0 IN IP4 127.0.0.1\ns=-\nt=0 0\nm=application 9 UDP/DTLS/SCTP webrtc-datachannel\nc=IN IP4 127.0.0.1\na=fingerprint:SHA-256 AA:BB:CC:DD:EE:FF\n",
            }
            client_ws.send_json({
                "type": "sdp_offer",
                "room_id": room_id,
                "sdp": fake_offer,
                "fingerprint": "SHA-256 AA:BB:CC:DD:EE:FF",
            })

            node_offer = node_ws.receive_json()
            assert node_offer["type"] == "sdp_offer"
            assert node_offer["room_id"] == room_id
            assert node_offer["sdp"]["type"] == "offer"
            assert "fingerprint" in node_offer
            assert node_offer["fingerprint"] == "SHA-256 AA:BB:CC:DD:EE:FF"

            # ── 8. Node sends SDP answer ──
            fake_answer = {
                "type": "answer",
                "sdp": "v=0\no=node 0 0 IN IP4 127.0.0.1\ns=-\nt=0 0\nm=application 9 UDP/DTLS/SCTP webrtc-datachannel\nc=IN IP4 127.0.0.1\n",
            }
            node_ws.send_json({
                "type": "sdp_answer",
                "room_id": room_id,
                "sdp": fake_answer,
            })

            client_answer = client_ws.receive_json()
            assert client_answer["type"] == "sdp_answer"
            assert client_answer["room_id"] == room_id
            assert client_answer["sdp"]["type"] == "answer"

            # ── 9. Client sends ICE candidates ──
            client_ws.send_json({
                "type": "ice_candidate",
                "room_id": room_id,
                "from": "client",
                "candidate": {"candidate": "candidate:1 1 UDP 127.0.0.1 5000 typ host", "sdpMid": "0"},
            })
            node_ice = node_ws.receive_json()
            assert node_ice["type"] == "ice_candidate"
            assert node_ice["room_id"] == room_id

            # ── 10. Node sends ICE candidates ──
            node_ws.send_json({
                "type": "ice_candidate",
                "room_id": room_id,
                "from": "node",
                "candidate": {"candidate": "candidate:2 1 UDP 127.0.0.1 5001 typ host", "sdpMid": "0"},
            })
            client_ice = client_ws.receive_json()
            assert client_ice["type"] == "ice_candidate"
            assert client_ice["room_id"] == room_id

            # ── 11. Client closes the room ──
            client_ws.send_json({"type": "close_room", "room_id": room_id})

            node_close = node_ws.receive_json()
            assert node_close["type"] == "room_closed"
            assert node_close["room_id"] == room_id

            # ── 12. Verify the room was cleaned up ──
            from litert_signaling.app import node_registry
            import asyncio

            async def check_room_gone():
                r = await node_registry.get_room(room_id)
                return r is None

            room_gone = asyncio.get_event_loop().run_until_complete(check_room_gone())
            assert room_gone, "Room was not cleaned up after close"


def test_e2e_select_model_resolves_node(client, node_api_key, client_jwt):
    """select_model with only model name (no @node) resolves via select_best_node."""
    with client.websocket_connect("/signal") as node_ws:
        node_ws.send_json({"type": "auth", "api_key": node_api_key})
        node_ws.receive_json()
        node_ws.send_json({
            "type": "register",
            "name": "auto-node",
            "models": [{"name": "gemma-4-12b"}],
        })
        node_ws.receive_json()

        with client.websocket_connect("/signal") as client_ws:
            client_ws.send_json({"type": "auth_jwt", "token": client_jwt})
            client_ws.receive_json()

            client_ws.send_json({"type": "select_model", "model": "gemma-4-12b"})
            resp = client_ws.receive_json()
            assert resp["type"] == "room_created", f"select_model failed: {resp}"


def test_e2e_room_full_rejected(client, node_api_key, client_jwt):
    """Node at capacity should reject additional rooms."""
    with client.websocket_connect("/signal") as node_ws:
        node_ws.send_json({"type": "auth", "api_key": node_api_key})
        node_ws.receive_json()
        node_ws.send_json({
            "type": "register",
            "name": "full-node",
            "models": [{"name": "m"}],
            "max_sessions": 1,
        })
        node_ws.receive_json()

        with client.websocket_connect("/signal") as c1:
            c1.send_json({"type": "auth_jwt", "token": client_jwt})
            c1.receive_json()
            c1.send_json({"type": "create_room", "model": "m@full-node"})
            c1.receive_json()
            node_ws.receive_json()

            with client.websocket_connect("/signal") as c2:
                c2.send_json({"type": "auth_jwt", "token": client_jwt})
                c2.receive_json()
                c2.send_json({"type": "create_room", "model": "m@full-node"})
                resp = c2.receive_json()
                assert resp["type"] == "error"
                assert "capacity" in resp["error"].lower() or "not available" in resp["error"].lower()


def test_e2e_node_disconnect_cleans_rooms(client, node_api_key, client_jwt):
    """When a node disconnects, all its rooms should be cleaned up."""
    with client.websocket_connect("/signal") as node_ws:
        node_ws.send_json({"type": "auth", "api_key": node_api_key})
        node_ws.receive_json()
        node_ws.send_json({"type": "register", "name": "disco-node", "models": [{"name": "m"}]})
        node_ws.receive_json()

        with client.websocket_connect("/signal") as client_ws:
            client_ws.send_json({"type": "auth_jwt", "token": client_jwt})
            client_ws.receive_json()
            client_ws.send_json({"type": "create_room", "model": "m@disco-node"})
            client_ws.receive_json()
            node_ws.receive_json()

        # client_ws closed here, node_ws still open

    # node_ws closed here → node disconnected → rooms should be cleaned
    from litert_signaling.app import node_registry
    import asyncio
    import time

    async def check():
        return len(node_registry._rooms) == 0 and node_registry._nodes.get("disco-node") is None

    time.sleep(0.3)
    cleaned = asyncio.get_event_loop().run_until_complete(check())
    assert cleaned, "Rooms or node were not cleaned up after disconnect"


def test_e2e_health_endpoint(client):
    """Health endpoint works and returns status."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "nodes" in data
    assert "rooms" in data


def test_e2e_private_node_not_listed(client, node_api_key, client_jwt):
    """Private nodes don't appear in list_nodes for other clients."""
    with client.websocket_connect("/signal") as node_ws:
        node_ws.send_json({"type": "auth", "api_key": node_api_key})
        node_ws.receive_json()
        node_ws.send_json({
            "type": "register",
            "name": "stealth-node",
            "models": [{"name": "hidden-model"}],
            "visibility": "private",
        })
        node_ws.receive_json()

        with client.websocket_connect("/signal") as client_ws:
            client_ws.send_json({"type": "auth_jwt", "token": client_jwt})
            client_ws.receive_json()
            client_ws.send_json({"type": "list_nodes"})
            resp = client_ws.receive_json()
            models = [n["model"] for n in resp["nodes"]]
            assert "hidden-model" not in models


def test_e2e_audit_log_created(client, initialized_db, node_api_key):
    """Connections and auth should create audit log entries."""
    import sqlite3

    with client.websocket_connect("/signal") as ws:
        ws.send_json({"type": "auth", "api_key": node_api_key})
        ws.receive_json()

    conn = sqlite3.connect(str(initialized_db))
    rows = conn.execute(
        "SELECT action FROM audit_log ORDER BY id DESC LIMIT 3"
    ).fetchall()
    conn.close()

    actions = [r[0] for r in rows]
    assert "ws_connected" in actions
    assert "auth_ok" in actions
    assert "ws_disconnected" in actions
