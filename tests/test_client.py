"""Quick test: connect to signaling, find a model, send a query via WebRTC P2P.

Usage:
    python test_client.py --signaling ws://127.0.0.1:9876/signal
                          --model gemma4-e2b
                          --message "Hola, ¿cómo estás?"
                          --jwt-secret test-secret-123

This simulates what the browser does:
  1. Connect to signaling server
  2. Authenticate via auth_jwt
  3. Ask for available nodes
  4. Select a model → create room
  5. Create WebRTC offer → exchange SDP/ICE
  6. Send message via DataChannel
  7. Print response from gamer's GPU
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test-client")

# Try to use jose for JWT generation, or generate a simple one
try:
    from jose import jwt as jose_jwt
except ImportError:
    jose_jwt = None

HAS_AIORTC = False
try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
    HAS_AIORTC = True
except ImportError:
    pass


def create_jwt(email: str, secret: str) -> str:
    """Create a JWT compatible with the signaling server."""
    if jose_jwt:
        now = datetime.now(timezone.utc)
        payload = {
            "sub": email,
            "role": "user",
            "iat": now,
            "exp": now + timedelta(hours=1),
            "jti": os.urandom(12).hex(),
        }
        return jose_jwt.encode(payload, secret, algorithm="HS256")
    else:
        # Simple base64-encoded JWT simulation
        # This won't work with real JWT validation, but useful for testing
        import base64
        import hashlib
        import hmac

        header = base64.urlsafe_b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode()).rstrip(b"=").decode()
        now_ts = int(time.time())
        payload = base64.urlsafe_b64encode(
            json.dumps({"sub": email, "role": "user", "iat": now_ts, "exp": now_ts + 3600}).encode()
        ).rstrip(b"=").decode()
        sig = hmac.new(secret.encode(), f"{header}.{payload}".encode(), hashlib.sha256).digest()
        signature = base64.urlsafe_b64encode(sig).rstrip(b"=").decode()
        return f"{header}.{payload}.{signature}"


class SignalingConnector:
    """Minimal signaling client that simulates a browser's WebRTC flow."""

    def __init__(self, url: str, email: str = "test@local.dev", jwt_secret: str = "test-secret-123"):
        self.url = url
        self.email = email
        self.jwt_secret = jwt_secret
        self.ws = None
        self.pc = None
        self.dc = None
        self.room_id = None
        self.node_id = None
        self.response_buffer = ""
        self.connected = asyncio.Event()

    async def connect(self):
        import websockets

        token = create_jwt(self.email, self.jwt_secret)
        logger.info(f"Connecting to {self.url}")

        self.ws = await websockets.connect(self.url)

        # Auth
        await self.ws.send(json.dumps({"type": "auth_jwt", "token": token}))
        resp = json.loads(await self.ws.recv())
        if resp.get("type") != "auth_ok":
            raise RuntimeError(f"Auth failed: {resp}")
        logger.info(f"Connected as {resp.get('user_id')}")

    async def list_nodes(self) -> list[dict]:
        await self.ws.send(json.dumps({"type": "list_nodes"}))
        resp = json.loads(await self.ws.recv())
        nodes = resp.get("nodes", [])
        logger.info(f"Available nodes: {len(nodes)}")
        for n in nodes:
            logger.info(f"  {n['model']} on {n['node']} (load: {n['load']}/{n['max_load']})")
        return nodes

    async def select_model(self, model_name: str) -> dict:
        await self.ws.send(json.dumps({"type": "select_model", "model": model_name}))
        resp = json.loads(await self.ws.recv())
        if resp.get("type") == "error":
            raise RuntimeError(f"Model selection failed: {resp.get('error')}")
        if resp.get("type") != "room_created":
            raise RuntimeError(f"Unexpected response: {resp}")
        self.room_id = resp["room_id"]
        self.node_id = resp["node"]
        logger.info(f"Room {self.room_id} created on node {self.node_id}")
        return resp

    async def setup_webrtc(self):
        if not HAS_AIORTC:
            logger.error("aiortc not installed. Install with: pip install aiortc")
            logger.error("Falling back to simulated chat...")
            return False

        config = RTCConfiguration([
            RTCIceServer(urls="stun:stun.l.google.com:19302"),
        ])
        self.pc = RTCPeerConnection(configuration=config)

        # Handle ICE candidates
        @self.pc.on("icecandidate")
        async def on_icecandidate(candidate):
            if candidate:
                await self.ws.send(json.dumps({
                    "type": "ice_candidate",
                    "room_id": self.room_id,
                    "candidate": {
                        "candidate": str(candidate),
                        "sdpMid": candidate.sdpMid,
                        "sdpMLineIndex": candidate.sdpMLineIndex,
                    },
                }))

        # Handle DataChannel
        @self.pc.on("datachannel")
        async def on_datachannel(channel):
            self.dc = channel
            logger.info(f"DataChannel received: {channel.label}")

            @channel.on("message")
            async def on_message(msg):
                try:
                    data = json.loads(msg)
                    if data.get("type") == "chunk":
                        content = data.get("data", {}).get("message", {}).get("content", "")
                        if content:
                            print(content, end="", flush=True)
                            self.response_buffer += content
                    elif data.get("type") == "done":
                        print()
                        logger.info(f"Response complete ({len(self.response_buffer)} chars)")
                        self.connected.set()
                    elif data.get("type") == "error":
                        logger.error(f"Model error: {data.get('error')}")
                        self.connected.set()
                except json.JSONDecodeError:
                    print(msg, end="", flush=True)
                    self.response_buffer += msg

        # Create DataChannel (offerer)
        self.dc = self.pc.createDataChannel("chat")
        logger.info("DataChannel created")

        @self.dc.on("open")
        async def on_open():
            logger.info("DataChannel open - ready to send!")

        # Create offer
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)

        # Extract fingerprint
        fingerprint = ""
        for line in offer.sdp.split("\n"):
            if line.strip().startswith("a=fingerprint:"):
                fingerprint = line.strip().split(":", 1)[1].strip()

        # Send offer to signaling
        await self.ws.send(json.dumps({
            "type": "sdp_offer",
            "room_id": self.room_id,
            "sdp": {"sdp": offer.sdp, "type": offer.type},
            "fingerprint": fingerprint,
        }))
        logger.info("SDP offer sent")

        # Handle SDP answer + ICE candidates via signaling
        async def listen():
            try:
                async for raw in self.ws:
                    msg = json.loads(raw)
                    if msg.get("type") == "sdp_answer":
                        desc = RTCSessionDescription(
                            sdp=msg["sdp"]["sdp"],
                            type=msg["sdp"]["type"],
                        )
                        await self.pc.setRemoteDescription(desc)
                        logger.info("Remote description set (SDP answer received)")
                    elif msg.get("type") == "ice_candidate":
                        cand = msg.get("candidate", {})
                        if cand and self.pc.remoteDescription:
                            try:
                                from aiortc import candidate_from_sdp
                                c = candidate_from_sdp(cand.get("candidate", ""))
                                if c:
                                    await self.pc.addIceCandidate(c)
                            except Exception as e:
                                logger.warning(f"ICE error: {e}")
                    elif msg.get("type") == "room_closed":
                        logger.info("Room closed")
                        break
                    elif msg.get("type") == "error":
                        logger.warning(f"Signaling error: {msg.get('error')}")
            except Exception as e:
                logger.warning(f"Listen error: {e}")
            finally:
                await self.cleanup()

        asyncio.create_task(listen())
        return True

    async def send_message(self, text: str):
        if self.dc and self.dc.readyState == "open":
            msg = json.dumps({
                "type": "infer",
                "request_id": f"req_{int(time.time())}_{os.urandom(4).hex()}",
                "endpoint": "/api/chat",
                "payload": {
                    "model": "gemma4-e2b",
                    "messages": [{"role": "user", "content": text}],
                    "stream": True,
                },
            })
            self.dc.send(msg)
            logger.info("Message sent, waiting for response...")
            print("Response: ", end="", flush=True)
            # Wait for response
            try:
                await asyncio.wait_for(self.connected.wait(), timeout=60)
            except asyncio.TimeoutError:
                logger.warning("Response timeout after 60s")
            print()
            return self.response_buffer
        else:
            logger.error("DataChannel not open")
            return None

    async def cleanup(self):
        if self.pc:
            await self.pc.close()
        if self.ws:
            await self.ws.close()
        self.pc = None
        self.ws = None


async def test_direct_query(url: str, model: str = "gemma4-e2b", message: str = "Hola", jwt_secret: str = "test-secret-123"):
    """Full test: connect → select model → chat via WebRTC."""
    client = SignalingConnector(url, jwt_secret=jwt_secret)

    try:
        await client.connect()
        nodes = await client.list_nodes()

        if not nodes:
            logger.error("No nodes available. Is the connector running?")
            return False

        await client.select_model(model)
        await client.setup_webrtc()

        # Wait for DataChannel to open
        await asyncio.sleep(2)

        response = await client.send_message(message)
        if response:
            logger.info(f"Full response length: {len(response)} chars")
            return True
        return False

    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False
    finally:
        await client.cleanup()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Friendly LLM Test Client")
    parser.add_argument("--signaling", default="ws://127.0.0.1:9876/signal", help="Signaling server URL")
    parser.add_argument("--model", default="gemma4-e2b", help="Model to use")
    parser.add_argument("--message", default="Hola, cuentame una historia corta", help="Message to send")
    parser.add_argument("--jwt-secret", default="test-secret-123", help="JWT secret")

    args = parser.parse_args()

    success = asyncio.run(test_direct_query(
        url=args.signaling,
        model=args.model,
        message=args.message,
        jwt_secret=args.jwt_secret,
    ))

    sys.exit(0 if success else 1)
