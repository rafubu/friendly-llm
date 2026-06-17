from __future__ import annotations

import argparse
import sys

from .config import settings


def main():
    parser = argparse.ArgumentParser(description="LiteRT-Ollama Signaling Server")
    sub = parser.add_subparsers(dest="command")

    serve_p = sub.add_parser("serve", help="Start signaling server")
    serve_p.add_argument("--host", default=settings.host, help="Host to bind")
    serve_p.add_argument("--port", type=int, default=settings.port, help="Port to bind")
    serve_p.add_argument("--jwt-secret", default=settings.jwt_secret, help="JWT secret")
    serve_p.add_argument("--db-path", default=settings.db_path, help="Database path")

    admin_p = sub.add_parser("admin", help="Admin commands")
    admin_sub = admin_p.add_subparsers(dest="admin_cmd")
    inv_p = admin_sub.add_parser("invite", help="Create invite code")
    inv_p.add_argument("--count", type=int, default=1, help="Number of codes")

    args = parser.parse_args()

    if args.command == "serve":
        settings.host = args.host
        settings.port = args.port
        settings.jwt_secret = args.jwt_secret
        settings.db_path = args.db_path
        _run_serve(args)
    elif args.command == "admin":
        if args.admin_cmd == "invite":
            _run_invite(args)
        else:
            admin_p.print_help()
    else:
        parser.print_help()


def _run_serve(args):
    import uvicorn
    from .app import app

    print(f"Starting LiteRT-Ollama Signaling on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def _run_invite(args):
    import sqlite3
    import uuid
    from pathlib import Path

    db_path = settings.db_path
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS invites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT UNIQUE NOT NULL,
            created_by TEXT NOT NULL,
            used_by TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            used_at TEXT
        )
    """)

    for _ in range(args.count):
        code = uuid.uuid4().hex[:12].upper()
        conn.execute("INSERT INTO invites (code, created_by) VALUES (?, 'admin')", (code,))
        print(f"Invite code: {code}")

    conn.commit()
    conn.close()


if __name__ == "__main__":
    main()
