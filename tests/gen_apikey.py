"""Generate a signaling API key for testing the connector."""
import hashlib, sqlite3, secrets, uuid
from pathlib import Path

db_path = str(Path.home() / ".litert-signaling" / "signaling.db")
Path(db_path).parent.mkdir(parents=True, exist_ok=True)
conn = sqlite3.connect(db_path)
conn.execute("PRAGMA journal_mode=WAL")

conn.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL, role TEXT DEFAULT 'user',
        created_at TEXT DEFAULT (datetime('now')), disabled INTEGER DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS nodes (
        id TEXT PRIMARY KEY, name TEXT NOT NULL, api_key_hash TEXT NOT NULL,
        owner_id TEXT, created_at TEXT DEFAULT (datetime('now')),
        last_seen TEXT, status TEXT DEFAULT 'offline',
        FOREIGN KEY (owner_id) REFERENCES users(id)
    );
    CREATE TABLE IF NOT EXISTS invites (
        id INTEGER PRIMARY KEY AUTOINCREMENT, code TEXT UNIQUE NOT NULL,
        created_by TEXT NOT NULL, used_by TEXT,
        created_at TEXT DEFAULT (datetime('now')), used_at TEXT
    );
    CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, action TEXT NOT NULL,
        details TEXT, ip TEXT, created_at TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS api_keys (
        id TEXT PRIMARY KEY, key_hash TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL, role TEXT DEFAULT 'node',
        created_at TEXT DEFAULT (datetime('now')), disabled INTEGER DEFAULT 0
    );
""")

row = conn.execute("SELECT id FROM api_keys WHERE name = 'default-node'").fetchone()
if not row:
    node_key = "sk-" + secrets.token_hex(16)
    key_hash = hashlib.sha256(node_key.encode()).hexdigest()
    conn.execute(
        "INSERT INTO api_keys (id, key_hash, name, role) VALUES (?, ?, ?, ?)",
        (uuid.uuid4().hex, key_hash, "default-node", "node"),
    )
    conn.commit()
    print(f"[INIT] Created default node API key: {node_key}")
else:
    # Create a new key with a unique name
    suffix = secrets.token_hex(4)
    node_key = "sk-" + secrets.token_hex(16)
    key_hash = hashlib.sha256(node_key.encode()).hexdigest()
    conn.execute(
        "INSERT INTO api_keys (id, key_hash, name, role) VALUES (?, ?, ?, ?)",
        (uuid.uuid4().hex, key_hash, f"node-{suffix}", "node"),
    )
    conn.commit()
    print(f"[NEW] Additional API key: {node_key}")

conn.close()
