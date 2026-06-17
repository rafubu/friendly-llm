# LiteRT-Ollama

**Ollama-compatible local LLM server** using the LiteRT-LM engine and Gemma 4 models.
Multi-node distributed inference with P2P WebRTC relay — no data passes through the VPS.

## Features

- **Ollama API** – `/api/generate`, `/api/chat`, `/api/tags`, `/api/show`, `/api/create`, `/api/embed`, `/api/pull`, `/api/ps`, `/api/version`
- **OpenAI API** – `/v1/chat/completions`, `/v1/models`, `/v1/embeddings`
- **Multimodal** – Images + Audio (natively via `litert_lm.Content`)
- **Tool calling** – Function calling with proxy relay
- **JSON mode** – Constrained decoding (`enable_constrained_decoding`)
- **Modelfile** – Create custom model configurations (params, system, template)
- **P2P Relay** – WebRTC DataChannel connects clients directly to inference nodes
- **Multi-node** – Up to 20+ PCs can serve models simultaneously
- **SDKs** – Python + JavaScript clients for easy integration

## Packages

| Package | Install | Purpose |
|---------|---------|---------|
| `litert-ollama` | `pip install litert-ollama` | Server + CLI |
| `litert-signaling` | `pip install litert-signaling` | VPS signaling server |
| `litert-sdk` | `pip install litert-sdk` | Python client SDK |
| `litert-sdk-js` | `npm install litert-sdk` | JavaScript client SDK |

## Quickstart

```bash
# 1. Start the server (PC Gamer)
litert-ollama serve --port 11434

# 2. Start signaling server (VPS)
litert-signaling serve --port 9876

# 3. Connect PC to signaling (PC Gamer)
litert-ollama connect --relay wss://mi-vps.com:9876 --api-key sk-xxx

# 4. Chat from anywhere using the SDK
litert-chat --signaling wss://mi-vps.com:9876
```

## Architecture

```
        Phone                          VPS                         PC Gamer
     (litert-sdk JS)              (litert-signaling)           (litert-connector)
          │                              │                           │
          │  1. WS connect + auth       │                           │
          │─────────────────────────────►│                           │
          │  2. ask: available models   │                           │
          │◄── list: ["g4-12b@pc1"] ────│                           │
          │                              │  3. WS connect + auth    │
          │                              │◄──────────────────────────│
          │  4. join_room(g4-12b@pc1)   │  4'. register models      │
          │─────────────────────────────►│──────────────────────────►│
          │◄── SDP offer ───────────────│◄── SDP offer ─────────────│
          │── SDP answer ──────────────►│── SDP answer ────────────►│
          │◄── ICE candidates ──────────│◄── ICE candidates ────────│
          │── ICE candidates ──────────►│── ICE candidates ────────►│
          │                              │                           │
          │  ═══ WEBRTC P2P (DTLS encrypted) ═══                  │
          │────────────────────────────────────────────────────────►│
          │◄────────────────────────────────────────────────────────│
          │  (data never touches the VPS)                           │
```

## Documentation

- [Architecture](docs/architecture.md)
- [Security Model](docs/security-model.md)
- [Protocol](docs/protocol.md)
- Package READMEs inside `packages/*/`

## License

MIT
