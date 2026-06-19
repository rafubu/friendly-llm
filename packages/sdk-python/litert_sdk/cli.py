from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from .client import LitertClient
from .local_client import LitertLocalClient
from .errors import LitertError

DEFAULT_LOCAL = "http://127.0.0.1:11434"


def main():
    parser = argparse.ArgumentParser(
        description="LiteRT-Ollama Chat CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            f"  litert-chat --local                  Chat interactivo con servidor local\n"
            f"  litert-chat --local --model gemma-4  Usar modelo específico\n"
            f"  litert-chat \"Hola\"                    Consulta directa a localhost:11434\n"
            f"  litert-chat --signaling wss://... --token <jwt> --model m  Chat vía P2P\n"
        ),
    )
    parser.add_argument("--signaling", help="Signaling server URL (ws:// or wss://)")
    parser.add_argument("--token", help="JWT auth token for P2P mode")
    parser.add_argument("--model", help="Model ID to use")
    parser.add_argument("--local", action="store_true", help=f"Connect to local server (default: {DEFAULT_LOCAL})")
    parser.add_argument("--host", default=os.getenv("LITERT_HOST", DEFAULT_LOCAL), help="Server host/URL")
    parser.add_argument("--image", action="append", default=[], help="Image attachment path")
    parser.add_argument("--format", choices=["json"], default=None, help="Response format")
    parser.add_argument("prompt", nargs="*", help="Single prompt (modo no interactivo)")

    args = parser.parse_args()
    prompt = " ".join(args.prompt) if args.prompt else None

    if args.signaling:
        asyncio.run(_run_p2p(args, prompt))
    elif args.local or not args.signaling:
        args.local = args.host
        asyncio.run(_run_local(args, prompt))
    else:
        parser.print_help()


async def _run_local(args, prompt: str | None):
    async with LitertLocalClient(args.local, model=args.model or "") as client:
        if prompt:
            async for chunk in client.chat(prompt, images=args.image or None, format=args.format):
                if not chunk.done:
                    print(chunk.text, end="", flush=True)
            print()
            return

        print("Chat mode (Ctrl+C to exit)\n")
        history = []
        while True:
            try:
                text = input("> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text:
                continue
            history.append({"role": "user", "content": text})
            full = ""
            async for chunk in client.chat(text, images=args.image or None):
                if not chunk.done:
                    print(chunk.text, end="", flush=True)
                    full += chunk.text
            print()
            history.append({"role": "assistant", "content": full})


async def _run_p2p(args, prompt: str | None):
    if not args.token:
        print("Error: --token is required for P2P mode")
        return

    async with LitertClient(signaling_url=args.signaling, auth_token=args.token, model=args.model) as client:
        if not args.model:
            models = await client.list_models()
            if not models:
                print("No models available")
                return
            args.model = models[0].id
            print(f"Using model: {args.model}")

        if prompt:
            async for chunk in client.chat(prompt, images=args.image or None, format=args.format):
                if not chunk.done:
                    print(chunk.text, end="", flush=True)
            print()
            return

        print(f"Chat with {args.model} (Ctrl+C to exit)\n")
        while True:
            try:
                text = input("> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text:
                continue
            async for chunk in client.chat(text):
                if not chunk.done:
                    print(chunk.text, end="", flush=True)
            print()


if __name__ == "__main__":
    main()
