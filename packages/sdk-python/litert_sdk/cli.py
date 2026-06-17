from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from .client import LitertClient
from .local_client import LitertLocalClient
from .errors import LitertError


def main():
    parser = argparse.ArgumentParser(description="LiteRT-Ollama Chat CLI")
    parser.add_argument("--signaling", help="Signaling server URL")
    parser.add_argument("--token", help="JWT auth token")
    parser.add_argument("--model", help="Model ID to use")
    parser.add_argument("--local", help="Local server URL (e.g., http://127.0.0.1:11434)")
    parser.add_argument("--image", action="append", default=[], help="Image attachment")
    parser.add_argument("--format", choices=["json", None], default=None)
    parser.add_argument("prompt", nargs="*", help="Single prompt")

    args = parser.parse_args()
    prompt = " ".join(args.prompt) if args.prompt else None

    if args.local:
        asyncio.run(_run_local(args, prompt))
    elif args.signaling:
        asyncio.run(_run_p2p(args, prompt))
    else:
        parser.print_help()


async def _run_local(args, prompt: str | None):
    async with LitertLocalClient(args.local) as client:
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
