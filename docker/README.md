# LiteRT-Ollama Docker Deploy

## VPS Deployment (Signaling + TURN)

```bash
# 1. Set secrets
export JWT_SECRET=$(openssl rand -hex 32)
export TURN_PASSWORD=$(openssl rand -hex 16)

# 2. Start services
docker-compose up -d

# 3. Create invite codes
docker-compose exec signaling litert-signaling admin invite --count 5

# 4. Check logs
docker-compose logs -f signaling
```

## Architecture

```
Internet (WSS :9876) ──► Docker Host (VPS)
                              │
                    ┌──────── ┴────────┐
                    │  signaling:9876  │  WebSocket SDP/ICE relay
                    │  turn:3478       │  TURN fallback (UDP/TCP)
                    └─────────────────┘
```

## Scaling

- 1 vCPU, 1GB RAM is enough for 20+ nodes and 100+ clients.
- TURN is only used when P2P fails (~5-8% of connections).
- For production: set up nginx reverse proxy in front of signaling for TLS termination.

## TLS

For production, set up nginx to terminate TLS:

```nginx
server {
    listen 443 ssl;
    server_name signal.ejemplo.com;

    ssl_certificate /etc/letsencrypt/live/signal.ejemplo.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/signal.ejemplo.com/privkey.pem;

    location /signal {
        proxy_pass http://127.0.0.1:9876;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```
