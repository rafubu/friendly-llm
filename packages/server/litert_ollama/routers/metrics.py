from __future__ import annotations

import time

from fastapi import APIRouter

from ..engine_manager import registry
from ..gpu_monitor import gpu_monitor

router = APIRouter()


@router.get("/api/metrics")
async def get_metrics():
    """Returns server metrics: GPU state, queue status, request stats."""
    loaded = registry.get_loaded_models()
    gpu_info = await gpu_monitor.get_info()

    models = []
    for m in loaded:
        models.append({
            "id": m["id"],
            "ref_count": m["ref_count"],
            "idle_seconds": m["idle_seconds"],
            "queue": {
                "active": m["queue_active"],
                "waiting": m["queue_waiting"],
                "available_slots": m["queue_available"],
            },
        })

    return {
        "server": {
            "uptime_seconds": 0,
            "loaded_models_count": len(loaded),
        },
        "gpu": gpu_info or {"available": False, "message": "No GPU detected via nvidia-smi"},
        "models": models,
        "queue": {
            "total_active": sum(m["queue_active"] for m in loaded),
            "total_waiting": sum(m["queue_waiting"] for m in loaded),
        },
    }
