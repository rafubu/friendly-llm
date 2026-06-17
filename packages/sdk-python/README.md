# LiteRT-Ollama Python SDK

Client SDK for connecting to LiteRT-Ollama servers via P2P WebRTC or direct HTTP.

## Installation

```bash
pip install litert-sdk
# With CLI tools
pip install litert-sdk[cli]
```

## Quickstart

### Direct HTTP (local network)

```python
from litert_sdk import LitertLocalClient

async with LitertLocalClient("http://192.168.1.50:11434") as client:
    async for chunk in client.chat("Hello!"):
        print(chunk.text, end="")
```

### P2P WebRTC (anywhere)

```python
from litert_sdk import LitertClient

async with LitertClient(
    signaling_url="wss://mi-vps.com/signal",
    auth_token="jwt-token",
    model="gemma4-12b@rtx4090",
) as client:
    async for chunk in client.chat("Tell me a joke"):
        print(chunk.text, end="")
```

### Sync mode

```python
from litert_sdk import LitertLocalClient

client = LitertLocalClient()
response = client.chat_sync("Hello!")
print(response.text)
```

### CLI

```bash
# HTTP mode
litert-chat --local http://192.168.1.50:11434

# P2P mode
litert-chat --signaling wss://mi-vps.com/signal --token jwt-token --model gemma4-12b@rtx4090

# One-shot
litert-chat --local http://localhost:11434 "Explain Python"
```

## API Reference

### LitertClient (P2P)

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `chat(text, *, images, tools, format)` | `text: str`, `images: list[File/bytes]`, `tools: list[dict]`, `format: str` | `AsyncIterator[Chunk]` | Stream chat |
| `list_models()` | — | `list[ModelInfo]` | Available models |

### LitertLocalClient (HTTP)

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `chat(text, *, images, tools, format)` | ídem | `AsyncIterator[Chunk]` | Stream chat |
| `chat_sync(text, ...)` | ídem | `Response` | Full response |
| `generate(prompt, *, images, options, format)` | — | `AsyncIterator[Chunk]` | Generate endpoint |
| `embed(text)` | `text: str` | `list[float]` | Embedding |
| `list_models()` | — | `ModelInfo[]` | Local models |

### Chunk

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | Text delta |
| `done` | `bool` | Is this the final chunk? |
| `tool_calls` | `list[dict]` | Any tool calls returned |
| `eval_count` | `int` | Tokens generated (final only) |

## Security

- P2P data is encrypted via DTLS-SRTP (AES-256-GCM) — mandatory WebRTC
- DTLS fingerprint verified against JWT to prevent MITM
- HTTP mode is intended for trusted local networks only
