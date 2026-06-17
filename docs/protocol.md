# Signaling Protocol

## WebSocket Endpoint

```
wss://<signaling-server>/signal
wss://<signaling-server>/signal
```

## Authentication

### Node (API Key)

```json
{"type": "auth", "api_key": "sk-<node-api-key>"}
```
Response:
```json
{"type": "auth_ok", "node_id": "node_xyz"}
```

### Client (JWT)

```json
{"type": "auth_jwt", "token": "<jwt-access-token>"}
```
Response:
```json
{"type": "auth_ok", "user_id": "user_abc"}
```

## Node Messages

### Register (node → signaling)

```json
{
  "type": "register",
  "name": "rtx4090-gamer",
  "models": [{"name": "gemma4-12b", "max_sessions": 3}],
  "max_sessions": 5,
  "visibility": "public",
  "allowed_keys": []
}
```

### Status report (node → signaling)

```json
{"type": "status", "load": 2, "vram_free_mb": 6144}
```

## Client Messages

### List nodes (client → signaling)

```json
{"type": "list_nodes"}
```
Response:
```json
{
  "type": "node_list",
  "nodes": [
    {"id": "gemma4-12b@rtx4090", "node": "rtx4090", "model": "gemma4-12b",
     "load": 2, "max_load": 5, "visibility": "public"}
  ]
}
```

### Create room (client → signaling)

```json
{"type": "create_room", "model": "gemma4-12b@rtx4090"}
```
Response:
```json
{"type": "room_created", "room_id": "room_abc123", "node": "rtx4090"}
```

### SDP Offer (client → signaling)

```json
{
  "type": "sdp_offer",
  "room_id": "room_abc123",
  "sdp": {"sdp": "...", "type": "offer"},
  "fingerprint": "SHA-256 AA:BB:CC:..."
}
```

### SDP Answer (node → signaling)

```json
{
  "type": "sdp_answer",
  "room_id": "room_abc123",
  "sdp": {"sdp": "...", "type": "answer"}
}
```

### ICE Candidate (both → signaling)

```json
{
  "type": "ice_candidate",
  "room_id": "room_abc123",
  "from": "client",
  "candidate": {"candidate": "candidate:... sdpMid:0"}
}
```

## Data Protocol (WebRTC DataChannel)

### Inference request (client → node)

```json
{
  "type": "infer",
  "request_id": "req_abc123",
  "endpoint": "/api/chat",
  "payload": {
    "model": "gemma4-12b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }
}
```

### Streaming chunk (node → client)

```json
{
  "type": "chunk",
  "request_id": "req_abc123",
  "data": {
    "model": "gemma4-12b",
    "message": {"role": "assistant", "content": "Hello"},
    "done": false
  }
}
```

### Final response (node → client)

```json
{
  "type": "done",
  "request_id": "req_abc123",
  "data": {
    "model": "gemma4-12b",
    "message": {"role": "assistant", "content": ""},
    "done": true,
    "done_reason": "stop",
    "total_duration": 123456789,
    "eval_count": 42
  }
}
```

### Error (node → client)

```json
{
  "type": "error",
  "request_id": "req_abc123",
  "error": "Model not loaded"
}
```
