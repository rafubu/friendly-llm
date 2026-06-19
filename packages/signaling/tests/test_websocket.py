from __future__ import annotations

import pytest


def test_websocket_auth_required(client):
    with client.websocket_connect("/signal") as ws:
        ws.send_json({"type": "list_nodes"})
        resp = ws.receive_json()
        assert resp["type"] == "error"
        assert "Authenticate first" in resp["error"]


def test_websocket_auth_with_api_key(client, node_api_key):
    with client.websocket_connect("/signal") as ws:
        ws.send_json({"type": "auth", "api_key": node_api_key})
        resp = ws.receive_json()
        assert resp["type"] == "auth_ok"
        assert "node_id" in resp


def test_websocket_auth_with_invalid_api_key(client):
    with client.websocket_connect("/signal") as ws:
        ws.send_json({"type": "auth", "api_key": "sk-invalid"})
        resp = ws.receive_json()
        assert resp["type"] == "auth_error"


def test_websocket_auth_with_jwt(client, client_jwt):
    with client.websocket_connect("/signal") as ws:
        ws.send_json({"type": "auth_jwt", "token": client_jwt})
        resp = ws.receive_json()
        assert resp["type"] == "auth_ok"
        assert "user_id" in resp


def test_websocket_auth_with_invalid_jwt(client):
    with client.websocket_connect("/signal") as ws:
        ws.send_json({"type": "auth_jwt", "token": "invalid.jwt.token"})
        resp = ws.receive_json()
        assert resp["type"] == "auth_error"


def test_websocket_ping_pong(client, node_api_key):
    with client.websocket_connect("/signal") as ws:
        ws.send_json({"type": "auth", "api_key": node_api_key})
        ws.receive_json()
        ws.send_json({"type": "ping"})
        resp = ws.receive_json()
        assert resp["type"] == "pong"


def test_node_register_after_auth(client, node_api_key):
    with client.websocket_connect("/signal") as ws:
        ws.send_json({"type": "auth", "api_key": node_api_key})
        ws.receive_json()
        ws.send_json({
            "type": "register",
            "name": "test-node",
            "models": [{"name": "gemma-4-12b"}],
            "max_sessions": 3,
            "visibility": "public",
        })
        resp = ws.receive_json()
        assert resp["type"] == "registered"


def test_node_register_updates_existing(client, node_api_key):
    with client.websocket_connect("/signal") as ws:
        ws.send_json({"type": "auth", "api_key": node_api_key})
        ws.receive_json()
        ws.send_json({"type": "register", "name": "update-node", "models": [{"name": "model-a"}]})
        ws.receive_json()

        ws.send_json({"type": "register", "models": [{"name": "model-b"}]})
        resp = ws.receive_json()
        assert resp["type"] == "registered"

        ws.send_json({"type": "list_nodes"})
        node_list = ws.receive_json()
        models = [n["model"] for n in node_list["nodes"]]
        assert "model-b" in models


def test_create_room_and_sdp_flow(client, node_api_key, client_jwt):
    """Full room creation + SDP offer/answer using nested connections."""
    with client.websocket_connect("/signal") as node_ws:
        node_ws.send_json({"type": "auth", "api_key": node_api_key})
        node_ws.receive_json()
        node_ws.send_json({
            "type": "register", "name": "flow-node",
            "models": [{"name": "gemma-4-12b"}],
        })
        node_ws.receive_json()

        with client.websocket_connect("/signal") as client_ws:
            client_ws.send_json({"type": "auth_jwt", "token": client_jwt})
            client_ws.receive_json()

            client_ws.send_json({"type": "create_room", "model": "gemma-4-12b@flow-node"})
            resp = client_ws.receive_json()
            assert resp["type"] == "room_created"
            room_id = resp["room_id"]

            node_notif = node_ws.receive_json()
            assert node_notif["type"] == "new_room"
            assert node_notif["room_id"] == room_id

            client_ws.send_json({
                "type": "sdp_offer", "room_id": room_id,
                "sdp": {"type": "offer", "sdp": "v=0..."},
                "fingerprint": "SHA-256 AA:BB:CC",
            })
            sdp_notif = node_ws.receive_json()
            assert sdp_notif["type"] == "sdp_offer"
            assert sdp_notif["fingerprint"] == "SHA-256 AA:BB:CC"

            node_ws.send_json({
                "type": "sdp_answer", "room_id": room_id,
                "sdp": {"type": "answer", "sdp": "v=0..."},
            })
            answer = client_ws.receive_json()
            assert answer["type"] == "sdp_answer"

            client_ws.send_json({
                "type": "ice_candidate", "room_id": room_id,
                "from": "client",
                "candidate": {"candidate": "candidate:1 1 UDP", "sdpMid": "0"},
            })
            node_ws.receive_json()

            node_ws.send_json({
                "type": "ice_candidate", "room_id": room_id,
                "from": "node",
                "candidate": {"candidate": "candidate:2 1 UDP", "sdpMid": "0"},
            })
            client_ws.receive_json()


def test_close_room(client, node_api_key, client_jwt):
    with client.websocket_connect("/signal") as node_ws:
        node_ws.send_json({"type": "auth", "api_key": node_api_key})
        node_ws.receive_json()
        node_ws.send_json({"type": "register", "name": "close-node", "models": [{"name": "m"}]})
        node_ws.receive_json()

        with client.websocket_connect("/signal") as client_ws:
            client_ws.send_json({"type": "auth_jwt", "token": client_jwt})
            client_ws.receive_json()
            client_ws.send_json({"type": "create_room", "model": "m@close-node"})
            resp = client_ws.receive_json()
            room_id = resp["room_id"]
            node_ws.receive_json()

            client_ws.send_json({"type": "close_room", "room_id": room_id})
            node_notif = node_ws.receive_json()
            assert node_notif["type"] == "room_closed"
            assert node_notif["room_id"] == room_id


def test_room_at_capacity(client, node_api_key, client_jwt):
    with client.websocket_connect("/signal") as node_ws:
        node_ws.send_json({"type": "auth", "api_key": node_api_key})
        node_ws.receive_json()
        node_ws.send_json({"type": "register", "name": "cap-node", "models": [{"name": "m"}], "max_sessions": 1})
        node_ws.receive_json()

        with client.websocket_connect("/signal") as c1:
            c1.send_json({"type": "auth_jwt", "token": client_jwt})
            c1.receive_json()
            c1.send_json({"type": "create_room", "model": "m@cap-node"})
            c1.receive_json()
            node_ws.receive_json()

            with client.websocket_connect("/signal") as c2:
                c2.send_json({"type": "auth_jwt", "token": client_jwt})
                c2.receive_json()
                c2.send_json({"type": "create_room", "model": "m@cap-node"})
                resp = c2.receive_json()
                assert resp["type"] == "error"
                assert "capacity" in resp["error"].lower() or "not available" in resp["error"].lower()
