# Security Policy

## Encryption

- **WebRTC P2P**: DTLS 1.2 + AES-256-GCM. Mandatory per WebRTC spec.
- **Signaling VPS → Client**: WSS (TLS 1.3) for WebSocket.
- **Client → VPS API**: HTTPS (TLS 1.3) for REST.
- **Fingerprint pinning**: SHA-256 hash of the DTLS certificate is embedded in the signed JWT.
  Clients verify this fingerprint during the P2P handshake to prevent MITM
  even if the VPS is compromised.
- **API Keys**: Stored as SHA-256 hashes. Never in plaintext.
- **Passwords**: Bcrypt (cost factor 12).

## Threat Model

See docs/security-model.md for the full threat model covering:
- Network: MITM, STUN spoofing, TURN abuse, DoS
- Authentication: JWT theft, API key compromise, brute force
- Application: prompt injection, SSRF, path traversal, race conditions

## Reporting Vulnerabilities

security@litert-ollama.dev (PGP: awaiting)
No public GitHub issues for security vulnerabilities.

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.x.x   | Yes       |
| 0.x.x   | No        |
