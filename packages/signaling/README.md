# LiteRT-Ollama Signaling Server (VPS)

Lightweight WebRTC signaling relay for P2P connections between client and inference nodes.

## Installation

```bash
pip install litert-signaling
```

## Usage

```bash
# Start server
litert-signaling serve --port 9876

# Create invite codes for user registration
litert-signaling admin invite --count 5
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/signal` | WS | WebSocket signaling |
| `/auth/login` | POST | Login with email+password |
| `/auth/register` | POST | Register with invite code |
| `/nodes` | GET | List available nodes |

## Docker

```bash
docker-compose up -d
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LITERT_SIGNALING_JWT_SECRET` | `change-me` | JWT signing secret |
| `LITERT_SIGNALING_DB_PATH` | `~/.litert-signaling/signaling.db` | SQLite database path |
| `LITERT_SIGNALING_RATE_LIMIT_PER_MINUTE` | `60` | Max messages per minute |
