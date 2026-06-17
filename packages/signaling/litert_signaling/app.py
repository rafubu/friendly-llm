from __future__ import annotations

import asyncio
import json
import hashlib
import logging
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from jose import JWTError, jwt

from . import config as cfg

logger = logging.getLogger(__name__)

JWT_SECRET = cfg.settings.jwt_secret
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 1
REFRESH_EXPIRY_DAYS = 30


@dataclass
class NodeInfo:
    node_id: str
    ws: WebSocket
    name: str
    api_key_hash: str
    models: list[dict[str, Any]] = field(default_factory=list)
    load: int = 0
    max_load: int = 5
    vram_free_mb: int = 0
    last_seen: float = field(default_factory=time.time)
    visibility: str = "public"
    allowed_keys: list[str] = field(default_factory=list)


@dataclass
class Room:
    room_id: str
    node_id: str
    model_id: str
    client_ws: WebSocket | None = None
    client_id: str | None = None
    sdp_offer: dict | None = None
    sdp_answer: dict | None = None
    ice_candidates: list[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


class NodeRegistry:
    def __init__(self):
        self._nodes: dict[str, NodeInfo] = {}
        self._rooms: dict[str, Room] = {}
        self._lock = asyncio.Lock()

    async def register_node(self, ws: WebSocket, api_key: str, info: dict) -> NodeInfo:
        async with self._lock:
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            node_id = info.get("name", f"node_{uuid.uuid4().hex[:8]}")

            node = NodeInfo(
                node_id=node_id,
                ws=ws,
                name=info.get("name", node_id),
                api_key_hash=key_hash,
                models=info.get("models", []),
                load=0,
                max_load=info.get("max_sessions", 5),
                visibility=info.get("visibility", "public"),
                allowed_keys=info.get("allowed_keys", []),
            )
            self._nodes[node_id] = node
            return node

    async def unregister_node(self, node_id: str):
        async with self._lock:
            self._nodes.pop(node_id, None)
            for rid, room in list(self._rooms.items()):
                if room.node_id == node_id:
                    self._rooms.pop(rid, None)

    async def get_node(self, node_id: str) -> NodeInfo | None:
        async with self._lock:
            return self._nodes.get(node_id)

    async def get_node_by_ws(self, ws: WebSocket) -> NodeInfo | None:
        async with self._lock:
            for node in self._nodes.values():
                if node.ws == ws:
                    return node
            return None

    async def list_available(self) -> list[dict]:
        async with self._lock:
            result = []
            for node in self._nodes.values():
                if node.visibility == "private":
                    continue
                for m in node.models:
                    result.append({
                        "id": f'{m.get("name", "unknown")}@{node.node_id}',
                        "node": node.node_id,
                        "model": m.get("name"),
                        "load": node.load,
                        "max_load": node.max_load,
                        "vram_free_mb": node.vram_free_mb,
                        "visibility": node.visibility,
                    })
            return result

    async def get_room(self, room_id: str) -> Room | None:
        async with self._lock:
            return self._rooms.get(room_id)

    async def get_room_by_node(self, node_id: str) -> list[Room]:
        async with self._lock:
            return [r for r in self._rooms.values() if r.node_id == node_id]

    async def update_node_load(self, node_id: str):
        async with self._lock:
            rooms = [r for r in self._rooms.values() if r.node_id == node_id]
            node = self._nodes.get(node_id)
            if node:
                node.load = len(rooms)

    async def select_best_node(self, model_name: str, client_key: str | None = None) -> str | None:
        """Find the best node for a given model name.
        Returns node_id or None if no node has this model available.
        Selection: least loaded node that has the model.
        """
        async with self._lock:
            candidates = []
            for node_id, node in self._nodes.items():
                if node.visibility == "private":
                    continue
                if node.visibility == "friends" and client_key and client_key not in node.allowed_keys:
                    continue
                has_model = any(
                    m.get("name") == model_name
                    for m in node.models
                )
                if has_model and node.load < node.max_load:
                    candidates.append((node.load, node_id))

            if not candidates:
                return None
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]

    async def create_room(self, model_spec: str, client_ws: WebSocket, client_id: str) -> Room | None:
        async with self._lock:
            if "@" not in model_spec:
                return None
            model_name, node_id = model_spec.rsplit("@", 1)
            node = self._nodes.get(node_id)
            if not node:
                return None
            if node.load >= node.max_load:
                return None

            room_id = f"room_{uuid.uuid4().hex[:12]}"
            room = Room(
                room_id=room_id,
                node_id=node_id,
                model_id=model_name,
                client_ws=client_ws,
                client_id=client_id,
            )
            self._rooms[room_id] = room
            node.load += 1
            return room

    async def close_room(self, room_id: str):
        async with self._lock:
            room = self._rooms.pop(room_id, None)
            if room:
                node = self._nodes.get(room.node_id)
                if node:
                    node.load = max(0, node.load - 1)

    async def authenticate_node(self, api_key: str) -> str | None:
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        async with self._lock:
            for node in self._nodes.values():
                if node.api_key_hash == key_hash:
                    return node.node_id
        import sqlite3
        db_path = cfg.settings.db_path
        try:
            conn = sqlite3.connect(db_path)
            row = conn.execute("SELECT id, name FROM api_keys WHERE key_hash = ? AND disabled = 0", (key_hash,)).fetchone()
            conn.close()
            if row:
                return row[0]
        except Exception:
            pass
        return None

    async def authenticate(self, ws: WebSocket, msg: dict) -> str | None:
        msg_type = msg.get("type", "")
        if msg_type == "auth":
            api_key = msg.get("api_key", "")
            return await self.authenticate_node(api_key)
        if msg_type == "auth_jwt":
            token = msg.get("token", "")
            try:
                payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
                return payload.get("sub", "")
            except JWTError:
                return None
        return None


node_registry = NodeRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="LiteRT-Ollama Signaling",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoginRequest(BaseModel):
    email: str
    password: str


class InviteRequest(BaseModel):
    code: str


class RegisterClientRequest(BaseModel):
    email: str
    password: str
    invite_code: str


@app.post("/auth/login")
async def login(req: LoginRequest):
    from passlib.hash import bcrypt
    import sqlite3

    db_path = cfg.settings.db_path
    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT id, password_hash, role FROM users WHERE email = ?", (req.email,)).fetchone()
    conn.close()

    user = None
    if row and bcrypt.verify(req.password, row[1]):
        user = {"id": row[0], "role": row[2]}

    if not user:
        raise HTTPException(401, "Invalid credentials")

    now = datetime.now(timezone.utc)
    access_payload = {
        "sub": user["id"],
        "role": user["role"],
        "iat": now,
        "exp": now + timedelta(hours=JWT_EXPIRY_HOURS),
        "jti": uuid.uuid4().hex,
    }
    access_token = jwt.encode(access_payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    refresh_payload = {
        "sub": user["id"],
        "type": "refresh",
        "iat": now,
        "exp": now + timedelta(days=REFRESH_EXPIRY_DAYS),
        "jti": uuid.uuid4().hex,
    }
    refresh_token = jwt.encode(refresh_payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": JWT_EXPIRY_HOURS * 3600,
    }


@app.post("/auth/register")
async def register(req: RegisterClientRequest):
    from passlib.hash import bcrypt
    import sqlite3

    db_path = cfg.settings.db_path
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")

    row = conn.execute("SELECT id FROM invites WHERE code = ? AND used_at IS NULL", (req.invite_code,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(400, "Invalid or used invite code")

    existing = conn.execute("SELECT id FROM users WHERE email = ?", (req.email,)).fetchone()
    if existing:
        conn.close()
        raise HTTPException(409, "Email already registered")

    pw_hash = bcrypt.hash(req.password)
    conn.execute(
        "INSERT INTO users (id, email, password_hash, role) VALUES (?, ?, ?, 'user')",
        (uuid.uuid4().hex, req.email, pw_hash),
    )
    conn.execute("UPDATE invites SET used_at = datetime('now'), used_by = ? WHERE code = ?", (req.email, req.invite_code))
    conn.commit()
    conn.close()

    return {"status": "registered"}


security = HTTPBearer(auto_error=False)


def _init_db():
    import sqlite3
    db_path = cfg.settings.db_path
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            created_at TEXT DEFAULT (datetime('now')),
            disabled INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS nodes (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            api_key_hash TEXT NOT NULL,
            owner_id TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            last_seen TEXT,
            status TEXT DEFAULT 'offline',
            FOREIGN KEY (owner_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS invites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT UNIQUE NOT NULL,
            created_by TEXT NOT NULL,
            used_by TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            used_at TEXT
        );
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            action TEXT NOT NULL,
            details TEXT,
            ip TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS api_keys (
            id TEXT PRIMARY KEY,
            key_hash TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            role TEXT DEFAULT 'node',
            created_at TEXT DEFAULT (datetime('now')),
            disabled INTEGER DEFAULT 0
        );
    """)

    admin_row = conn.execute("SELECT id FROM users WHERE email = 'admin'").fetchone()
    if not admin_row:
        from passlib.hash import bcrypt
        admin_pw = secrets.token_hex(8)
        admin_hash = bcrypt.hash(admin_pw)
        conn.execute(
            "INSERT INTO users (id, email, password_hash, role) VALUES (?, ?, ?, 'admin')",
            (uuid.uuid4().hex, "admin", admin_hash),
        )
        print(f"[INIT] Created admin user: admin / {admin_pw}")
        print(f"[INIT] ⚠ Change this password immediately via the signaling API.")

    api_key_row = conn.execute("SELECT id FROM api_keys WHERE name = 'default-node'").fetchone()
    if not api_key_row:
        node_key = f"sk-{secrets.token_hex(16)}"
        key_hash = hashlib.sha256(node_key.encode()).hexdigest()
        conn.execute(
            "INSERT INTO api_keys (id, key_hash, name, role) VALUES (?, ?, ?, 'node')",
            (uuid.uuid4().hex, key_hash, "default-node"),
        )
        print(f"[INIT] Created default node API key: {node_key}")
        print(f"[INIT] Save this key — it will not be shown again.")

    conn.commit()
    conn.close()


@app.on_event("startup")
async def startup():
    _init_db()
    logger.info("Signaling server initialized. Admin and default node API key created.")


@app.get("/nodes")
async def list_nodes():
    available = await node_registry.list_available()
    return {"nodes": available}


@app.websocket("/signal")
async def signal_websocket(ws: WebSocket):
    await ws.accept()
    authenticated = False
    node_id = None
    client_id = None

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type", "")

            if not authenticated:
                if msg_type == "auth":
                    nid = await node_registry.authenticate(ws, msg)
                    if nid:
                        authenticated = True
                        node_id = nid
                        await ws.send_json({"type": "auth_ok", "node_id": nid})
                        await node_registry.update_node_load(nid)
                    else:
                        await ws.send_json({"type": "auth_error", "error": "Invalid credentials"})
                    continue

                elif msg_type == "auth_jwt":
                    nid = await node_registry.authenticate(ws, msg)
                    if nid:
                        authenticated = True
                        client_id = nid
                        await ws.send_json({"type": "auth_ok", "user_id": nid})
                    else:
                        await ws.send_json({"type": "auth_error", "error": "Invalid token"})
                    continue
                else:
                    await ws.send_json({"type": "error", "error": "Authenticate first"})
                    continue

            if msg_type == "register":
                node = await node_registry.get_node(node_id) if node_id else None
                if node:
                    node.models = msg.get("models", [])
                    node.max_load = msg.get("max_sessions", node.max_load)
                    node.visibility = msg.get("visibility", node.visibility)
                    node.allowed_keys = msg.get("allowed_keys", [])
                    await ws.send_json({"type": "registered"})
                elif authenticated and node_id:
                    api_key = msg.get("api_key", "")
                    new_node = await node_registry.register_node(ws, api_key, msg)
                    if new_node:
                        node_id = new_node.node_id
                        await ws.send_json({"type": "registered", "node_id": node_id})
                        logger.info(f"Node registered: {new_node.name} ({node_id})")

            elif msg_type == "list_nodes":
                available = await node_registry.list_available()
                await ws.send_json({"type": "node_list", "nodes": available})

            elif msg_type == "select_model":
                model_name = msg.get("model", "")
                if "@" in model_name:
                    node_id_from_spec = model_name.rsplit("@", 1)[1]
                    spec = model_name
                else:
                    node_id_from_spec = await node_registry.select_best_node(model_name)
                    if not node_id_from_spec:
                        await ws.send_json({"type": "error", "error": f"No nodes available for model {model_name}"})
                        continue
                    spec = f"{model_name}@{node_id_from_spec}"
                room = await node_registry.create_room(spec, ws, client_id or "anonymous")
                if room:
                    node = await node_registry.get_node(room.node_id)
                    if node:
                        await ws.send_json({"type": "room_created", "room_id": room.room_id, "node": room.node_id})
                        await node.ws.send_json({
                            "type": "new_room",
                            "room_id": room.room_id,
                            "client_id": room.client_id,
                            "model": room.model_id,
                        })
                else:
                    await ws.send_json({"type": "error", "error": "Node not available or at capacity"})

            elif msg_type == "create_room":
                model_spec = msg.get("model", "")
                room = await node_registry.create_room(model_spec, ws, client_id or "anonymous")
                if room:
                    node = await node_registry.get_node(room.node_id)
                    if node:
                        await ws.send_json({"type": "room_created", "room_id": room.room_id, "node": room.node_id})
                        await node.ws.send_json({
                            "type": "new_room",
                            "room_id": room.room_id,
                            "client_id": room.client_id,
                            "model": room.model_id,
                        })
                else:
                    await ws.send_json({"type": "error", "error": "Node not available or at capacity"})

            elif msg_type == "sdp_offer":
                room_id = msg.get("room_id", "")
                room = await node_registry.get_room(room_id)
                if room:
                    room.sdp_offer = msg.get("sdp", {})
                    node = await node_registry.get_node(room.node_id)
                    if node:
                        await node.ws.send_json({
                            "type": "sdp_offer",
                            "room_id": room_id,
                            "sdp": msg.get("sdp", {}),
                            "fingerprint": msg.get("fingerprint", ""),
                        })

            elif msg_type == "sdp_answer":
                room_id = msg.get("room_id", "")
                room = await node_registry.get_room(room_id)
                if room and room.client_ws:
                    room.sdp_answer = msg.get("sdp", {})
                    await room.client_ws.send_json({
                        "type": "sdp_answer",
                        "room_id": room_id,
                        "sdp": msg.get("sdp", {}),
                    })

            elif msg_type == "ice_candidate":
                room_id = msg.get("room_id", "")
                room = await node_registry.get_room(room_id)
                if room:
                    candidate = msg.get("candidate", {})
                    room.ice_candidates.append(candidate)
                    target_ws = room.client_ws if msg.get("from") == "node" else (
                        (await node_registry.get_node(room.node_id)).ws if room.node_id else None
                    )
                    if target_ws:
                        try:
                            await target_ws.send_json({
                                "type": "ice_candidate",
                                "room_id": room_id,
                                "candidate": candidate,
                            })
                        except Exception:
                            pass

            elif msg_type == "close_room":
                room_id = msg.get("room_id", "")
                room = await node_registry.get_room(room_id)
                if room:
                    if room.client_ws:
                        try:
                            await room.client_ws.send_json({"type": "room_closed", "room_id": room_id})
                        except Exception:
                            pass
                    node = await node_registry.get_node(room.node_id)
                    if node:
                        try:
                            await node.ws.send_json({"type": "room_closed", "room_id": room_id})
                        except Exception:
                            pass
                    await node_registry.close_room(room_id)

            elif msg_type == "ping":
                await ws.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: node={node_id}")
    except Exception as e:
        logger.exception(f"WebSocket error for node={node_id}: {e}")
    finally:
        if node_id:
            await node_registry.unregister_node(node_id)
        for rid, room in list(node_registry._rooms.items()):
            if room.client_ws == ws or room.node_id == node_id:
                await node_registry.close_room(rid)
