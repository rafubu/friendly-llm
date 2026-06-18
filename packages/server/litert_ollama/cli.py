from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from .config import settings

logger = logging.getLogger(__name__)


def main():
    # CLI entry point using argparse
    import argparse

    parser = argparse.ArgumentParser(description="LiteRT-Ollama CLI")
    sub = parser.add_subparsers(dest="command")

    serve_p = sub.add_parser("serve", help="Start the Ollama-compatible server")
    serve_p.add_argument("--host", default=settings.host, help="Host to bind")
    serve_p.add_argument("--port", type=int, default=settings.port, help="Port to bind")
    serve_p.add_argument("--backend", default=settings.backend, choices=["cpu", "gpu", "auto"], help="Inference backend")
    serve_p.add_argument("--keep-alive", default=settings.keep_alive, help="Keep alive duration")
    serve_p.add_argument("--models-dir", default=settings.models_dir, help="Models directory")
    serve_p.add_argument("--enable-speculative-decoding", action="store_true", help="Enable spec decoding for ~2x speedup")
    serve_p.add_argument("--benchmark-on-startup", action="store_true", help="Run benchmark on startup to find best config")
    serve_p.add_argument("--force-benchmark", action="store_true", help="Re-run benchmark even if cached results exist")

    pull_p = sub.add_parser("pull", help="Download a model from HuggingFace")
    pull_p.add_argument("model", help="Model ID (e.g., gemma-4-12B-it-litert-lm)")
    pull_p.add_argument("--hf-token", help="HuggingFace token")

    list_p = sub.add_parser("list", help="List available models")
    show_p = sub.add_parser("show", help="Show model details")
    show_p.add_argument("model", help="Model ID")

    delete_p = sub.add_parser("delete", help="Delete a model")
    delete_p.add_argument("model", help="Model ID")

    run_p = sub.add_parser("run", help="Run a model interactively")
    run_p.add_argument("model", help="Model ID")
    run_p.add_argument("--prompt", help="Single prompt (non-interactive)")
    run_p.add_argument("--image", action="append", default=[], help="Image attachment path")
    run_p.add_argument("--format", choices=["json", None], default=None, help="Response format")
    run_p.add_argument("--system", help="System prompt")

    args = parser.parse_args()

    if args.command == "serve":
        _run_serve(args)
    elif args.command == "pull":
        _run_pull(args)
    elif args.command == "list":
        _run_list()
    elif args.command == "show":
        _run_show(args)
    elif args.command == "delete":
        _run_delete(args)
    elif args.command == "run":
        _run_interactive(args)
    else:
        parser.print_help()


def _run_serve(args):
    settings.host = args.host
    settings.port = args.port
    settings.backend = args.backend
    settings.keep_alive = args.keep_alive
    settings.models_dir = args.models_dir
    settings.enable_speculative_decoding = args.enable_speculative_decoding
    settings.benchmark_on_startup = args.benchmark_on_startup
    settings.force_benchmark = getattr(args, "force_benchmark", False)

    if settings.benchmark_on_startup:
        from .benchmark import run_model_benchmarks, find_model_paths, load_all_results

        logger.info("Benchmark mode — checking for cached results...")

        model_paths = find_model_paths(Path(settings.models_dir))
        if not model_paths:
            logger.warning("No models found for benchmarking")
        else:
            results_path = Path(settings.benchmark_results_path)
            cached = load_all_results(results_path) if not settings.force_benchmark else None
            missing_models = []

            if cached:
                for mid in model_paths:
                    if mid not in cached:
                        missing_models.append(mid)
                if not missing_models:
                    logger.info("All models have cached benchmark results — skipping benchmark")
                    for mid, mp in model_paths.items():
                        best = cached[mid].get("best_settings", {})
                        tps = cached[mid].get("best_decode_tps", 0)
                        tags = [best.get("backend", "cpu").upper()]
                        if best.get("spec_decoding"):
                            tags.append("SPEC")
                        logger.info(f"  {mid}: {'+'.join(tags)} @ {tps:.1f} t/s (cached)")
                else:
                    logger.info(f"Cached results miss {missing_models} — benchmarking those only")
            else:
                if settings.force_benchmark:
                    logger.info("--force-benchmark: re-running all benchmarks")
                else:
                    logger.info("No cached results — running benchmark...")
                missing_models = list(model_paths.keys())

            for mid in (missing_models or model_paths.keys()):
                mp = model_paths[mid]
                logger.info(f"Benchmarking {mid} ({mp})")
                results = run_model_benchmarks(mid, mp, results_path)
                best = results.get("best_settings", {})
                tps = results.get("best_decode_tps", 0)
                tags = [best.get("backend", "cpu").upper()]
                if best.get("spec_decoding"):
                    tags.append("SPEC")
                logger.info(f"  → {mid}: {'+'.join(tags)} @ {tps:.1f} t/s")

            logger.info(f"Results saved to {results_path}")

    import uvicorn
    from .app import app

    print(f"Starting LiteRT-Ollama on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def _run_pull(args):
    import json
    import requests

    url = f"http://{settings.host}:{settings.port}/api/pull"
    payload = {"model": args.model, "model_file": ""}
    if args.hf_token:
        payload["huggingface_token"] = args.hf_token

    try:
        resp = requests.post(url, json=payload, stream=True)
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                status = data.get("status", "")
                if status == "success":
                    print(f"Downloaded {args.model}")
                    return
                print(f"  {status}")
    except requests.exceptions.ConnectionError:
        print(f"Error: Cannot connect to server at {url}")


def _run_list():
    from .engine_manager import registry
    models = registry.discover_models()
    if not models:
        print("No models found")
        return
    print(f"{'ID':<30} {'SIZE':<12} {'MODIFIED'}")
    print("-" * 60)
    for m in sorted(models, key=lambda x: x.get("modified", 0), reverse=True):
        size = m.get("size", 0)
        if size > 1_000_000_000:
            size_str = f"{size / 1_000_000_000:.1f} GB"
        elif size > 1_000_000:
            size_str = f"{size / 1_000_000:.1f} MB"
        else:
            size_str = f"{size / 1_000:.0f} KB"
        print(f"{m['id']:<30} {size_str:<12}")


def _run_show(args):
    from .engine_manager import registry
    path = registry.find_model_path(args.model)
    if not path:
        print(f"Model {args.model!r} not found")
        return
    print(f"Model: {args.model}")
    print(f"Path: {path}")
    size = os.path.getsize(path)
    print(f"Size: {size / 1_000_000_000:.2f} GB")
    print(f"Format: litertlm")
    import subprocess
    print(f"Command: litert-ollama serve --model {args.model}")


def _run_delete(args):
    from .engine_manager import registry
    confirm = input(f"Delete model {args.model!r}? [y/N] ")
    if confirm.lower() != "y":
        print("Cancelled")
        return
    import asyncio
    asyncio.run(registry.unload_engine(args.model))
    path = registry.find_model_path(args.model)
    if path:
        os.remove(path)
        print(f"Deleted {args.model}")
    else:
        print(f"Model {args.model!r} not found")


def _run_interactive(args):
    import requests

    url = f"http://{settings.host}:{settings.port}/api/chat"

    system_prompt = args.system
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    print(f"Chatting with {args.model}. Ctrl+C to exit.\n")

    if args.prompt:
        messages.append({"role": "user", "content": args.prompt})
        payload = {"model": args.model, "messages": messages, "stream": True}
        try:
            resp = requests.post(url, json=payload, stream=True)
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    if not data.get("done"):
                        print(data.get("message", {}).get("content", ""), end="", flush=True)
            print()
        except requests.exceptions.ConnectionError:
            print(f"Error: Cannot connect to server at {url}")
        return

    while True:
        try:
            user_input = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        payload = {"model": args.model, "messages": messages[-10:], "stream": True}

        try:
            resp = requests.post(url, json=payload, stream=True)
            full_response = ""
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if not data.get("done") and content:
                        print(content, end="", flush=True)
                        full_response += content
            print()
            messages.append({"role": "assistant", "content": full_response})
        except requests.exceptions.ConnectionError:
            print(f"\nError: Cannot connect to server at {url}")
            break
        except KeyboardInterrupt:
            print("\n[Interrupted]")
            break


if __name__ == "__main__":
    main()
