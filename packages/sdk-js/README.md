# litert-sdk (JavaScript)

Client SDK for LiteRT-Ollama. Supports P2P WebRTC and direct HTTP (LAN).

## Installation

```bash
npm install litert-sdk
# or
<script src="https://unpkg.com/litert-sdk/dist/litert-sdk.min.js"></script>
```

## Usage

### P2P via Signaling Server (anywhere, no VPS data relay)

```javascript
import { LitertClient } from 'litert-sdk';

const client = new LitertClient({
  signalingUrl: 'wss://mi-vps.com/signal',
  authToken: 'jwt-token',
  model: 'gemma4-12b@rtx4090',
});

await client.connect();

for await (const chunk of client.chat('Hello!')) {
  console.log(chunk.text);
}

client.close();
```

### Direct HTTP (same LAN)

```javascript
import { LitertLocalClient } from 'litert-sdk';

const client = new LitertLocalClient('http://192.168.1.50:11434');
const models = await client.listModels();
console.log(models);

for await (const chunk of client.chat('Write a poem')) {
  document.getElementById('output').textContent += chunk.text;
}
```

### Browser with \<script\> tag

```html
<script src="litert-sdk.min.js"></script>
<script>
  const client = new LitertSDK.LitertLocalClient();
  client.listModels().then(models => console.log(models));
</script>
```

## Security

- WebRTC DataChannels use mandatory DTLS-SRTP encryption (AES-256-GCM).
- Data never passes through the VPS — only signaling (SDP/ICE).
- DTLS fingerprint verification prevents MITM even with a compromised VPS.

## API

### LitertClient

| Method | Returns | Description |
|--------|---------|-------------|
| `connect()` | `Promise<void>` | Connect to signaling, authenticate |
| `listModels()` | `Promise<ModelInfo[]>` | Available models across all nodes |
| `chat(text, opts?)` | `AsyncGenerator<Chunk>` | Stream chat response |
| `close()` | `void` | Disconnect |

### LitertLocalClient

| Method | Returns | Description |
|--------|---------|-------------|
| `listModels()` | `Promise<ModelInfo[]>` | Local models |
| `chat(text, opts?)` | `AsyncGenerator<Chunk>` | Stream response |
| `chatSync(text, opts?)` | `Promise<Response>` | Full response (no stream) |

## License

MIT
