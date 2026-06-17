from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys

from .signaling_client import SignalingClient
from .webrtc_responder import WebRTCResponder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


async def run_connector(args):
    models = []
    if args.model:
        for m in args.model:
            parts = m.split(":", 1)
            model_info = {"name": parts[0], "max_sessions": int(parts[1]) if len(parts) > 1 else 3}
            models.append(model_info)
    else:
        models = [{"name": "gemma4-12b", "max_sessions": 3}]

    responder = WebRTCResponder(
        local_server_url=f"http://127.0.0.1:{args.server_port}",
    )

    async def handle_message(msg):
        msg_type = msg.get("type", "")

        if msg_type == "new_room":
            room_id = msg.get("room_id")
            model = msg.get("model", "gemma4-12b")

            logger.info(f"New room {room_id} for model {model}")

            offer_data, _ = await responder.create_offer(room_id)
            await client.send({
                "type": "sdp_offer",
                "room_id": room_id,
                "sdp": offer_data["sdp"],
                "fingerprint": offer_data["fingerprint"],
            })

        elif msg_type == "sdp_answer":
            room_id = msg.get("room_id")
            sdp = msg.get("sdp")
            if sdp:
                await responder.set_remote_answer(room_id, sdp)

        elif msg_type == "ice_candidate":
            room_id = msg.get("room_id")
            candidate = msg.get("candidate")
            if candidate:
                await responder.add_ice_candidate(room_id, candidate)

        elif msg_type == "room_closed":
            room_id = msg.get("room_id")
            logger.info(f"Room {room_id} closed")

    client = SignalingClient(
        url=args.relay,
        api_key=args.api_key,
        node_name=args.name,
        models=models,
        visibility=args.visibility,
        max_sessions=args.max_sessions,
        on_message=handle_message,
    )

    try:
        await client.connect()
        logger.info(f"Connected to {args.relay} as {args.name}")
        logger.info(f"Models: {[m['name'] for m in models]}")

        while True:
            await asyncio.sleep(1)

    except asyncio.CancelledError:
        pass
    finally:
        await client.close()
        await responder.cleanup()


def main():
    parser = argparse.ArgumentParser(description="LiteRT-Ollama Connector")
    parser.add_argument("--relay", required=True, help="Signaling server URL (ws:// or wss://)")
    parser.add_argument("--api-key", required=True, help="API key for authentication")
    parser.add_argument("--name", default=os.uname().nodename if hasattr(os, "uname") else "gamer-pc", help="Node name")
    parser.add_argument("--model", action="append", help="Model to serve (format: name:max_sessions)")
    parser.add_argument("--visibility", default="public", choices=["public", "friends", "private"], help="Node visibility")
    parser.add_argument("--max-sessions", type=int, default=5, help="Maximum concurrent sessions")
    parser.add_argument("--server-port", type=int, default=11434, help="Local litert-ollama server port")

    args = parser.parse_args()

    try:
        asyncio.run(run_connector(args))
    except KeyboardInterrupt:
        print("\nShutdown")


if __name__ == "__main__":
    main()
