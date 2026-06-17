# LiteRT-Ollama Server

Ollama-compatible local LLM server powered by the Gemma 4 and LiteRT-LM.

## Installation

```bash
pip install litert-ollama
```

## Quickstart

```bash
# 1. Start the server
litert-ollama serve

# 2. Chat via API
curl http://localhost:11434/api/chat \
  -d '{"model": "gemma4-12b", "messages": [{"role": "user", "content": "Hello!"}]}'

# 3. Interactive CLI
litert-ollama run gemma4-12b
```

## API Endpoints

### Ollama-compatible

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/generate` | POST | Single prompt completion |
| `/api/chat` | POST | Multi-turn chat with tools |
| `/api/tags` | GET | List local models |
| `/api/show` | POST | Model details |
| `/api/create` | POST | Create model from Modelfile |
| `/api/embed` | POST | Generate embeddings |
| `/api/pull` | POST | Download from HuggingFace |
| `/api/delete` | DELETE | Remove a model |
| `/api/copy` | POST | Copy/rename model |
| `/api/ps` | GET | Loaded models |
| `/api/version` | GET | Server version |

### OpenAI-compatible

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat with streaming |
| `/v1/models` | GET | List models |
| `/v1/embeddings` | POST | Embeddings |

## Modelfile

Create custom model configurations:

```dockerfile
FROM gemma4-12b
PARAMETER temperature 0.7
SYSTEM "You are a coding assistant."
```

```bash
curl http://localhost:11434/api/create \
  -d '{"model": "gemma4-code", "modelfile": "FROM gemma4-12b\nPARAMETER temperature 0.7\nSYSTEM You are a coding assistant."}'
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `litert-ollama serve` | Start server |
| `litert-ollama run <model>` | Interactive chat |
| `litert-ollama list` | List models |
| `litert-ollama show <model>` | Model details |
| `litert-ollama pull <model>` | Download model |
| `litert-ollama delete <model>` | Delete model |

## Security

- `/api/create` rejects invalid Modelfiles
- Payload size limits: 10MB `/api/chat`, 20MB multimodal
- Context length capped at `max_output_tokens` (default: 8192)
- Rate limiting via configurable `rate_limit_per_minute`
