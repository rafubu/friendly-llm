# LiteRT-Ollama Connector (PC Gamer)

WebRTC connector that bridges the local LiteRT-Ollama server to the signaling server,
allowing remote clients to connect directly via P2P.

## Installation

```bash
pip install litert-connector
```

## Usage

```bash
# Connect to signaling server
litert-ollama-connect \
  --relay wss://mi-vps.com:9876 \
  --api-key sk-my-key \
  --name "rtx4090-gamer" \
  --model gemma4-12b:3 \
  --visibility public
```

## Options

| Flag | Description |
|------|-------------|
| `--relay` | Signaling server URL (required) |
| `--api-key` | Authentication key (required) |
| `--name` | Node display name |
| `--model` | Model to serve (format: `name:max_sessions`) |
| `--visibility` | `public`, `friends`, or `private` |
| `--max-sessions` | Max concurrent P2P sessions (default: 5) |
| `--server-port` | Local server port (default: 11434) |

## Architecture

```
Remote Client ──P2P WebRTC──► Connector ──HTTP──► litert-ollama server (:11434)
```
