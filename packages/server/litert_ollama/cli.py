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

    rename_p = sub.add_parser("rename", help="Rename a local model")
    rename_p.add_argument("model", help="Current model name or ID")
    rename_p.add_argument("new_name", help="New name for the model")

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
    elif args.command == "rename":
        _run_rename(args)
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
    import asyncio
    import shutil
    import sys
    import time
    from pathlib import Path

    import httpx
    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError

    hf_token = args.hf_token or os.getenv("HF_TOKEN")

    repo_id = args.model
    if "/" not in repo_id:
        repo_id = f"litert-community/{repo_id}-litert-lm"

    models_dir = Path(settings.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    termsize = shutil.get_terminal_size((80, 20))
    max_w = termsize.columns - 1

    def fmt_size(n: float) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if n < 1024:
                return f"{n:.1f} {unit}"
            n /= 1024
        return f"{n:.1f} TB"

    def write_line(line: str):
        line = line[:max_w]
        sys.stdout.write("\033[2K\r" + line)
        sys.stdout.flush()

    def done_line():
        sys.stdout.write("\033[2K\r")
        sys.stdout.flush()

    def draw_progress_bar(pct, downloaded, total, speed, elapsed):
        bar_w = 20
        filled = int(bar_w * pct / 100) if total > 0 else 0
        bar = "=" * filled + ">" * (1 if filled < bar_w else 0) + " " * (bar_w - filled - (1 if filled < bar_w else 0))
        dl = fmt_size(downloaded)
        if total > 0:
            tl = fmt_size(total)
            pct_str = f"{pct:5.1f}%"
            sp = fmt_size(speed) + "/s"
            write_line(f"  [{bar}] {pct_str}  {dl:>8}/{tl:<8}  {sp:>9}  {elapsed:4.0f}s")
        else:
            sp = fmt_size(speed) + "/s"
            write_line(f"  {dl:>8}  {sp:>9}  {elapsed:4.0f}s")

    async def do_pull():
        from .model_store import ModelStore
        store = ModelStore.get_instance()

        existing = store.find_by_source(repo_id)
        if existing:
            done_line()
            print(f"Model already exists as '{existing['name']}' — use 'litert-ollama rename' to change its name")
            return

        api = HfApi(token=hf_token) if hf_token else HfApi()

        headers = {}
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"

        write_line("  Pulling " + repo_id)

        spinner_stop = False

        async def spin(msg):
            chars = "-\\|/"
            i = 0
            while not spinner_stop:
                write_line(f"  {chars[i % 4]} {msg}...")
                await asyncio.sleep(0.12)
                i += 1

        spin_task = asyncio.create_task(spin("fetching model info"))
        try:
            model_info = await asyncio.to_thread(api.model_info, repo_id)
        except RepositoryNotFoundError:
            spinner_stop = True
            await spin_task
            done_line()
            print(f"Error: Model {repo_id} not found on HuggingFace")
            return
        finally:
            spinner_stop = True
            await spin_task

        files = [f for f in model_info.siblings if f.rfilename.endswith(".litertlm") or f.rfilename.endswith(".bin")]
        if not files:
            files = model_info.siblings

        model_id = args.model.replace("/", "--")
        dest_dir = models_dir / model_id
        dest_dir.mkdir(parents=True, exist_ok=True)

        total_size = sum(f.size or 0 for f in files)
        total_done = 0
        start_time = time.monotonic()

        for file_info in files:
            filename = file_info.rfilename
            file_size = file_info.size or 0

            if filename.endswith(".litertlm"):
                local_path = dest_dir / "model.litertlm"
            else:
                local_path = dest_dir / filename

            hf_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"

            downloaded = 0
            last_report = 0
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0)) as client:
                    async with client.stream("GET", hf_url, headers=headers, follow_redirects=True) as response:
                        response.raise_for_status()
                        if file_size == 0:
                            file_size = int(response.headers.get("content-length", 0))

                        local_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(local_path, "wb") as f:
                            async for chunk in response.aiter_bytes():
                                f.write(chunk)
                                downloaded += len(chunk)
                                if downloaded - last_report >= 1024 * 1024 or downloaded >= file_size:
                                    last_report = downloaded
                                    now = time.monotonic()
                                    done = total_done + downloaded
                                    total_size = max(total_size, done)
                                    pct = done / total_size * 100 if total_size > 0 else 0
                                    speed = done / (now - start_time) if (now - start_time) > 0 else 0
                                    draw_progress_bar(pct, done, total_size, speed, now - start_time)
            except Exception as e:
                done_line()
                print(f"Error: Failed to download {filename}: {e}")
                return

            total_done += downloaded

        elapsed = time.monotonic() - start_time
        done_line()
        print(f"Downloaded {args.model} in {elapsed:.1f}s")

        store.add_model(model_id, args.model, str(dest_dir), source=repo_id)

    asyncio.run(do_pull())


def _run_list():
    from .engine_manager import registry
    from .model_store import ModelStore
    store = ModelStore.get_instance()
    fs_models = registry.discover_models()
    if not fs_models:
        print("No models found")
        return

    store_models = {m["id"]: m for m in store.list_models()}

    print(f"{'NAME':<35} {'SIZE':<12} {'SOURCE'}")
    print("-" * 80)
    for m in sorted(fs_models, key=lambda x: x.get("modified", 0), reverse=True):
        size = m.get("size", 0)
        if size > 1_000_000_000:
            size_str = f"{size / 1_000_000_000:.1f} GB"
        elif size > 1_000_000:
            size_str = f"{size / 1_000_000:.1f} MB"
        else:
            size_str = f"{size / 1_000:.0f} KB"

        fs_id = m["id"]
        store_id = fs_id.replace("/", "--")
        record = store_models.get(store_id)
        name = record["name"] if record else fs_id
        source = record.get("source", "") if record else ""
        print(f"{name:<35} {size_str:<12} {source}")


def _run_show(args):
    from .engine_manager import registry
    from .model_store import ModelStore
    path = registry.find_model_path(args.model)
    if not path:
        print(f"Model {args.model!r} not found")
        return
    model_id = args.model.replace("/", "--")
    store = ModelStore.get_instance()
    record = store.get_model(model_id)
    print(f"Model: {args.model}")
    if record and record.get("source"):
        print(f"Source: {record['source']}")
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


def _run_rename(args):
    import re
    from pathlib import Path
    from .model_store import ModelStore
    from .config import settings
    store = ModelStore.get_instance()
    old_db_id = args.model.replace("/", "--")
    new_id = re.sub(r'[<>:"/\\|?*]', '-', args.new_name.replace("/", "--"))

    model = store.get_model(old_db_id)
    if not model:
        model = store.find_by_name(args.model)

    old_dir = None
    models_dir = Path(settings.models_dir)
    for d in models_dir.iterdir():
        if d.is_dir() and (d / "model.litertlm").exists():
            if d.name == old_db_id or d.name == args.model:
                old_dir = d
                break

    if not model and not old_dir:
        print(f"Model '{args.model}' not found")
        return

    if old_dir:
        new_dir = old_dir.parent / new_id
        if old_dir != new_dir:
            old_dir.rename(new_dir)
    elif model:
        old_path = Path(model["path"])
        new_path = old_path.parent / new_id
        if old_path.exists():
            old_path.rename(new_path)
        elif not new_path.exists():
            new_path.mkdir(parents=True, exist_ok=True)
        new_dir = new_path
    else:
        new_dir = old_dir.parent / new_id

    source = model.get("source", "") if model else ""
    store.add_model(new_id, args.new_name, str(new_dir), source=source)
    if model and model["id"] != new_id:
        if store.get_model(model["id"]):
            store.remove_model(model["id"])

    print(f"Renamed to '{args.new_name}'")
    if new_id != args.new_name.replace("/", "--"):
        print(f"  (stored as '{new_id}' — char(s) like ':' not allowed in filenames)")


def _run_interactive(args):
    import json as _json
    import shutil
    import sys
    import threading
    import time

    import requests

    url = f"http://{settings.host}:{settings.port}/api/chat"
    system_prompt = args.system
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    termsize = shutil.get_terminal_size((80, 20))
    max_w = termsize.columns - 1

    def write_line(line: str):
        sys.stdout.write("\033[2K\r" + line[:max_w])
        sys.stdout.flush()

    def done_line():
        sys.stdout.write("\033[2K\r")
        sys.stdout.flush()

    def fmt_size(n: float) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if n < 1024:
                return f"{n:.1f} {unit}"
            n /= 1024
        return f"{n:.1f} TB"

    spinner_running = False
    spinner_msg = ""

    def start_spinner(msg: str):
        nonlocal spinner_running, spinner_msg
        spinner_running = True
        spinner_msg = msg

        def _spin():
            chars = "-\\|/"
            i = 0
            while spinner_running:
                write_line(f"  {chars[i % 4]} {spinner_msg}")
                time.sleep(0.12)
                i += 1
            done_line()

        t = threading.Thread(target=_spin, daemon=True)
        t.start()
        return t

    def stop_spinner(t: threading.Thread | None):
        nonlocal spinner_running
        spinner_running = False
        if t:
            t.join(0.5)
        done_line()

    def send_and_stream(payload: dict) -> str | None:
        spin_thread = None
        try:
            resp = requests.post(url, json=payload, stream=True, timeout=(5, 300))

            if resp.status_code != 200:
                done_line()
                try:
                    err_data = resp.json()
                    err_msg = err_data.get("detail", err_data.get("error", str(resp.status_code)))
                except Exception:
                    err_msg = resp.reason or str(resp.status_code)
                if resp.status_code == 404:
                    print(f"\nError: Model '{args.model}' not found on server")
                    print(f"  Pull it: litert-ollama pull {args.model}")
                elif resp.status_code == 503:
                    print(f"\nError: Server is busy. Try again later.")
                else:
                    print(f"\nError: Server returned HTTP {resp.status_code} ({err_msg})")
                return None

            spin_thread = start_spinner("waiting for response")

            full_response = ""
            first_token = True
            got_any_data = False
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = _json.loads(line)
                except _json.JSONDecodeError:
                    continue

                got_any_data = True

                status = data.get("status")
                if status == "loading":
                    spinner_msg = data.get("message", "Loading model...")
                    continue
                if status == "loaded":
                    if spin_thread:
                        stop_spinner(spin_thread)
                        spin_thread = None
                    continue
                if status == "error":
                    stop_spinner(spin_thread)
                    spin_thread = None
                    print(f"\nError: {data.get('error', 'Unknown error')}")
                    return None

                if data.get("done"):
                    continue

                content = data.get("message", {}).get("content", "")
                if not content:
                    continue

                if first_token:
                    stop_spinner(spin_thread)
                    spin_thread = None
                    first_token = False

                print(content, end="", flush=True)
                full_response += content

            if first_token:
                stop_spinner(spin_thread)
                spin_thread = None

            if not got_any_data:
                print("\n  (no response from server)")
            else:
                print()
            return full_response

        except requests.exceptions.ConnectionError:
            stop_spinner(spin_thread)
            spin_thread = None
            done_line()
            print(f"Error: Cannot connect to server at {url}")
            print(f"  Start the server: litert-ollama serve")
            return None
        except requests.exceptions.Timeout:
            stop_spinner(spin_thread)
            spin_thread = None
            done_line()
            print(f"\nError: Request timed out")
            return None
        except Exception as e:
            stop_spinner(spin_thread)
            spin_thread = None
            done_line()
            print(f"\nError: {e}")
            return None
        finally:
            if spin_thread:
                spinner_running = False
                spin_thread.join(0.5)

    print(f"  Chatting with {args.model}. Ctrl+C to exit.\n")

    if args.prompt:
        messages.append({"role": "user", "content": args.prompt})
        payload = {"model": args.model, "messages": messages, "stream": True}
        send_and_stream(payload)
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
        response = send_and_stream(payload)
        if response is None:
            break
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
