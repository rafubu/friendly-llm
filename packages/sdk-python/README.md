# LiteRT SDK Python

SDK para conectar con servidores LiteRT-Ollama. Soporta modo **local** (HTTP directo) y **P2P** (WebRTC vía signaling).

## Instalación

```bash
pip install litert-sdk

# Para modo P2P (WebRTC):
pip install litert-sdk[webrtc]

# Para CLI con autocompletado:
pip install litert-sdk[cli]
```

## Uso como SDK

### Modo local (misma máquina / LAN)

```python
import asyncio
from litert_sdk import LitertLocalClient

async def main():
    async with LitertLocalClient(
        base_url="http://127.0.0.1:11434",
        model="gemma-4-12b"
    ) as client:
        # Chat streaming
        async for chunk in client.chat("¿Qué es Python?"):
            if not chunk.done:
                print(chunk.text, end="")
            else:
                print(f"\n[hecho: {chunk.done_reason}]")

        # Chat síncrono (respuesta completa)
        response = await client.chat_sync("Dime un chiste")
        print(response.text)

        # Generar (endpoint /api/generate)
        async for chunk in client.generate("Cuento corto"):
            print(chunk.text, end="")

        # Listar modelos disponibles
        models = await client.list_models()
        for m in models:
            print(f"  {m.id}")

        # Embeddings
        vector = await client.embed("texto a vectorizar")
        print(f"Embedding: {len(vector)} dimensiones")

asyncio.run(main())
```

### Modo P2P (vía signaling server)

Requiere `pip install litert-sdk[webrtc]`.

```python
import asyncio
from litert_sdk import LitertClient

async def main():
    async with LitertClient(
        signaling_url="wss://signal.app-re.online",
        auth_token="<jwt-token>",
        model="gemma-4-12b@rtx4090",
    ) as client:
        # Listar nodos disponibles
        models = await client.list_models()
        print("Modelos disponibles:")
        for m in models:
            print(f"  {m.id} (carga: {m.load}/{m.max_load})")

        # Chat streaming vía WebRTC P2P
        async for chunk in client.chat("Hola mundo"):
            if not chunk.done:
                print(chunk.text, end="")
            else:
                print(f"\n[hecho: {chunk.done_reason}]")

asyncio.run(main())
```

### Manejo de errores

```python
from litert_sdk import LitertLocalClient, LitertClient
from litert_sdk.errors import (
    ConnectionError,
    AuthError,
    ModelNotFoundError,
    RoomCreationError,
    TimeoutError,
    ContextOverflowError,
)

try:
    async with LitertLocalClient("http://127.0.0.1:11434") as client:
        async for chunk in client.chat("Hola"):
            pass
except ConnectionError as e:
    print(f"No se pudo conectar: {e}")
except ContextOverflowError as e:
    print(f"Contexto excedido: {e.estimated_tokens} > {e.context_limit}")
```

### Con imágenes

```python
async with LitertLocalClient("http://127.0.0.1:11434") as client:
    async for chunk in client.chat(
        "¿Qué hay en esta imagen?",
        images=["ruta/a/imagen.jpg"],
    ):
        print(chunk.text, end="")
```

## CLI

```bash
# Consulta rápida
litert-chat "Hola mundo" --model gemma-4

# Especificar host
litert-chat --host http://192.168.1.100:11434 "consulta"

# Usar variable de entorno
set LITERT_HOST=http://192.168.1.100:11434
litert-chat "consulta"

# Chat interactivo
litert-chat

# Modo P2P
litert-chat --signaling wss://signal.app-re.online --token <jwt> --model m

# Ayuda
litert-chat --help
```

## API de tipos

```python
from litert_sdk.types import Chunk, Response, ModelInfo, NodeInfo

# Chunk — cada fragmento del stream
chunk = Chunk(
    text="Hello",          # texto generado
    done=False,            # True si es el último
    done_reason="stop",    # stop, length, context_overflow, error
    eval_count=42,         # tokens generados
    total_duration=1_234_567_890,  # nanosegundos
)

# Response — respuesta completa síncrona
resp = Response(
    text="Hello world",
    tool_calls=...,
    usage={"prompt_tokens": 10, "completion_tokens": 20},
)
```

## Desarrollo

```bash
git clone https://github.com/rafubu/friendly-llm.git
cd friendly-llm/packages/sdk-python
pip install -e ".[dev]"
pytest
```
