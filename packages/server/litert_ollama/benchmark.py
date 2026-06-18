from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from litert_lm import Backend, Benchmark, BenchmarkInfo

logger = logging.getLogger(__name__)


def run_model_benchmarks(
    model_id: str,
    model_path: str,
    results_path: Path,
    prefill_tokens: int = 256,
    decode_tokens: int = 256,
) -> dict:
    """Test all backend configs for a model, return best one."""
    configs = [
        {"name": "cpu", "backend": Backend.CPU(), "spec": False},
        {"name": "cpu_spec", "backend": Backend.CPU(), "spec": True},
        {"name": "gpu", "backend": Backend.GPU(), "spec": False},
        {"name": "gpu_spec", "backend": Backend.GPU(), "spec": True},
    ]

    all_results = {}
    best_name = "cpu"
    best_decode_tps = -1.0
    best_config = {"backend": "cpu", "spec_decoding": False}

    for cfg in configs:
        name = cfg["name"]
        header = f"  {name}"
        try:
            b = Benchmark(
                model_path=model_path,
                backend=cfg["backend"],
                prefill_tokens=prefill_tokens,
                decode_tokens=decode_tokens,
                enable_speculative_decoding=cfg["spec"] if cfg["spec"] else None,
            )
            info: BenchmarkInfo = b.run()

            result = {
                "supported": True,
                "init_time_s": round(info.init_time_in_second, 3),
                "ttft_ms": round(info.time_to_first_token_in_second * 1000, 1),
                "prefill_tokens": info.last_prefill_token_count,
                "prefill_tps": round(info.last_prefill_tokens_per_second, 1),
                "decode_tokens": info.last_decode_token_count,
                "decode_tps": round(info.last_decode_tokens_per_second, 1),
            }
            all_results[name] = result
            decode_tps = info.last_decode_tokens_per_second
            logger.info(f"{header}: {decode_tps:.1f} t/s decode, {info.last_prefill_tokens_per_second:.1f} t/s prefill (TTFT {info.time_to_first_token_in_second*1000:.0f}ms)")

            if decode_tps > best_decode_tps:
                best_decode_tps = decode_tps
                best_name = name
                best_config = {
                    "backend": name.split("_")[0],
                    "spec_decoding": "spec" in name,
                }

        except Exception as e:
            err = str(e)[:200]
            all_results[name] = {"supported": False, "error": err}
            logger.warning(f"{header}: UNSUPPORTED — {err}")

    this_result = {
        "model_path": model_path,
        "best_config": best_name,
        "best_decode_tps": round(best_decode_tps, 1),
        "best_settings": best_config,
        "prefill_tokens_used": prefill_tokens,
        "decode_tokens_used": decode_tokens,
        "benchmarked_at": time.time(),
        "all_results": all_results,
    }

    # Load existing results, update this model's entry
    results_path.parent.mkdir(parents=True, exist_ok=True)
    all_models = {}
    if results_path.exists():
        try:
            all_models = json.loads(results_path.read_text())
        except Exception:
            pass
    all_models[model_id] = this_result
    results_path.write_text(json.dumps(all_models, indent=2))

    best_display = (
        f"  Best: {best_name} ({best_decode_tps:.1f} t/s decode)"
        if best_decode_tps > 0
        else "  ALL CONFIGS FAILED — using CPU fallback"
    )
    logger.info(best_display)

    return this_result


def get_model_config(model_id: str, results_path: Path) -> dict | None:
    """Get the best config for a specific model from cached results."""
    data = load_all_results(results_path)
    if data and model_id in data:
        return data[model_id].get("best_settings")
    return None


def get_model_decode_tps(model_id: str, results_path: Path) -> float | None:
    """Get decode tokens/second for a model from cached results."""
    data = load_all_results(results_path)
    if data and model_id in data:
        return data[model_id].get("best_decode_tps")
    return None


def load_all_results(results_path: Path) -> dict | None:
    if results_path.exists():
        try:
            return json.loads(results_path.read_text())
        except Exception:
            pass
    return None


def find_model_paths(models_dir: Path) -> dict[str, str]:
    """Discover all model paths."""
    model_paths = {}
    if not models_dir.exists():
        return model_paths
    for model_dir in models_dir.iterdir():
        if not model_dir.is_dir():
            continue
        model_file = model_dir / "model.litertlm"
        if model_file.exists():
            model_id = model_dir.name.replace("--", "/")
            model_paths[model_id] = str(model_file)
    return model_paths
