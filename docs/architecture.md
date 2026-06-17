# Architecture

## Overview

LiteRT-Ollama is a distributed LLM inference system built on top of the `litert-lm` Python API for Gemma 4 models. It provides an Ollama-compatible REST API on each node, plus a P2P WebRTC relay layer for remote clients.

## Components

```
┌─────────────────────────────────────────────────────────────┐
│                      PC Gamer Node                           │
│                                                              │
│  ┌──────────────────────────┐  ┌──────────────────────────┐ │
│  │  LiteRT-Ollama Server     │  │  LiteRT Connector        │ │
│  │  (FastAPI :11434)         │  │  (WebRTC responder)      │ │
│  │                           │  │                          │ │
│  │  /api/chat                │  │  SignalingClient ←──WSS──│─●──→ VPS
│  │  /api/generate            │  │  WebRTCResponder         │ │
│  │  /api/tags                │  │  Session Pool            │ │
│  │  /v1/chat/completions     │  └──────────────────────────┘ │
│  └──────────┬───────────────┘                                │
│             │ localhost                                       │
│  ┌──────────▼───────────────┐                                │
│  │  litert-lm Engine         │                                │
│  │  (native C++ / ctypes)    │                                │
│  └──────────────────────────┘                                │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

### Local (same network)
```
Client ──HTTP──► LiteRT-Ollama (localhost:11434) ──ctypes──► litert-lm Engine
```

### Remote (P2P WebRTC)
```
Phone ──WSS──► VPS ──WSS──► PC Gamer   (signaling, SDP/ICE exchange)
Phone ═══WebRTC═══► PC Gamer (encrypted P2P data, no VPS touch)
```

## Protocol

### Signaling (VPS — WebSocket)
- Minimal relay: only SDP, ICE candidates, and room management
- Message types on /signal endpoint:
  - `auth` / `auth_jwt` — authentication
  - `register` — node registers its models
  - `create_room` — client requests a model
  - `sdp_offer` / `sdp_answer` — WebRTC handshake
  - `ice_candidate` — NAT traversal
  - `close_room` — end session
  - `ping` / `pong` — keepalive

### Data (P2P — WebRTC DataChannel)
- Only encrypted data flows: no VPS is involved after the handshake
- DTLS-SRTP with AES-256-GCM (mandatory in WebRTC spec)
- JSON messages:
  - `{type: "infer", request_id, endpoint, payload}` — inference request
  - `{type: "chunk", request_id, data}` — streaming chunk
  - `{type: "done", request_id, data}` — final response
  - `{type: "error", request_id, error}` — error

## Security

See [security-model.md](security-model.md) for the full threat model.

Key points:
- WebRTC DTLS encryption is mandatory — no data travels in plaintext
- DTLS fingerprint pinning via JWT prevents MITM on the signaling channel
- API keys are stored as SHA-256 hashes
- Passwords use bcrypt
- Rate limiting on all endpoints
- TURN server only used as fallback when P2P fails
