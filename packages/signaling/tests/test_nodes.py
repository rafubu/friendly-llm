from __future__ import annotations

import asyncio

import pytest  # noqa: F401 (used for xfail mark)


def test_list_nodes_empty(client, client_jwt):
    with client.websocket_connect("/signal") as ws:
        ws.send_json({"type": "auth_jwt", "token": client_jwt})
        ws.receive_json()
        ws.send_json({"type": "list_nodes"})
        resp = ws.receive_json()
        assert resp["type"] == "node_list"
        assert resp["nodes"] == []


def test_list_nodes_shows_public(client, node_api_key, client_jwt):
    with client.websocket_connect("/signal") as node_ws:
        node_ws.send_json({"type": "auth", "api_key": node_api_key})
        node_ws.receive_json()
        node_ws.send_json({
            "type": "register",
            "name": "public-node",
            "models": [{"name": "gemma-4-12b"}, {"name": "gemma-4-27b"}],
            "visibility": "public",
        })
        node_ws.receive_json()

        with client.websocket_connect("/signal") as c:
            c.send_json({"type": "auth_jwt", "token": client_jwt})
            c.receive_json()
            c.send_json({"type": "list_nodes"})
            resp = c.receive_json()
            assert resp["type"] == "node_list"
            models = {n["model"] for n in resp["nodes"]}
            assert "gemma-4-12b" in models
            assert "gemma-4-27b" in models


def test_private_nodes_hidden(client, node_api_key, client_jwt):
    with client.websocket_connect("/signal") as node_ws:
        node_ws.send_json({"type": "auth", "api_key": node_api_key})
        node_ws.receive_json()
        node_ws.send_json({
            "type": "register",
            "name": "private-node",
            "models": [{"name": "secret-model"}],
            "visibility": "private",
        })
        node_ws.receive_json()

        with client.websocket_connect("/signal") as c:
            c.send_json({"type": "auth_jwt", "token": client_jwt})
            c.receive_json()
            c.send_json({"type": "list_nodes"})
            resp = c.receive_json()
            assert resp["type"] == "node_list"
            models = [n["model"] for n in resp["nodes"]]
            assert "secret-model" not in models


def test_select_model_returns_none_if_no_nodes(client, client_jwt):
    with client.websocket_connect("/signal") as c:
        c.send_json({"type": "auth_jwt", "token": client_jwt})
        c.receive_json()
        c.send_json({"type": "select_model", "model": "nonexistent"})
        resp = c.receive_json()
        assert resp["type"] == "error"
        assert "No nodes available" in resp["error"]


@pytest.mark.xfail(reason="select_model does not pass client_key to select_best_node, so friends/allowed_keys check is dead code (audit finding #16)")
def test_friends_visibility_hidden_from_unauthorized(client, node_api_key, client_jwt):
    with client.websocket_connect("/signal") as node_ws:
        node_ws.send_json({"type": "auth", "api_key": node_api_key})
        node_ws.receive_json()
        node_ws.send_json({
            "type": "register",
            "name": "friends-node",
            "models": [{"name": "friends-model"}],
            "visibility": "friends",
            "allowed_keys": ["known-key"],
        })
        node_ws.receive_json()

        with client.websocket_connect("/signal") as c:
            c.send_json({"type": "auth_jwt", "token": client_jwt})
            c.receive_json()
            c.send_json({"type": "select_model", "model": "friends-model"})
            resp = c.receive_json()
            assert resp["type"] == "error"
            assert "No nodes available" in resp["error"]


def test_node_load_updates_on_room_create(client, node_api_key, client_jwt):
    from litert_signaling.app import node_registry
    async def check_load(exp):
        n = await node_registry.get_node("load-node")
        return n and n.load == exp

    with client.websocket_connect("/signal") as node_ws:
        node_ws.send_json({"type": "auth", "api_key": node_api_key})
        node_ws.receive_json()
        node_ws.send_json({"type": "register", "name": "load-node", "models": [{"name": "m"}], "max_sessions": 5})
        node_ws.receive_json()

        assert asyncio.get_event_loop().run_until_complete(check_load(0))

        with client.websocket_connect("/signal") as c:
            c.send_json({"type": "auth_jwt", "token": client_jwt})
            c.receive_json()
            c.send_json({"type": "create_room", "model": "m@load-node"})
            resp = c.receive_json()
            assert resp["type"] == "room_created"

            assert asyncio.get_event_loop().run_until_complete(check_load(1))


def test_rest_list_nodes_endpoint(client, node_api_key, client_jwt):
    with client.websocket_connect("/signal") as ws:
        ws.send_json({"type": "auth", "api_key": node_api_key})
        ws.receive_json()
        ws.send_json({"type": "register", "name": "rest-node", "models": [{"name": "rest-model"}]})
        ws.receive_json()

        resp = client.get("/nodes")
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data
        models = [n["model"] for n in data["nodes"]]
        assert "rest-model" in models
