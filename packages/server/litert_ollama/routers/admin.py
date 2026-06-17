from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

router = APIRouter()

from ..config import settings
from ..engine_manager import registry
from ..gpu_monitor import gpu_monitor
from ..model_store import ModelStore

ADMIN_HTML = Path(__file__).resolve().parent.parent / "static" / "admin.html"

_active_sessions: dict[str, dict[str, Any]] = {}


def track_session(session_id: str, user: str, model: str, req_type: str = "http"):
    _active_sessions[session_id] = {
        "id": session_id,
        "user": user,
        "model": model,
        "type": req_type,
        "started_at": time.time(),
        "tokens": 0,
        "status": "active",
    }


def untrack_session(session_id: str):
    _active_sessions.pop(session_id, None)


@router.get("/admin")
async def admin_index():
    if ADMIN_HTML.exists():
        return HTMLResponse(ADMIN_HTML.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Admin panel not found</h1>")


@router.get("/admin/api/status")
async def admin_status():
    loaded = registry.get_loaded_models()
    gpu_info = await gpu_monitor.get_info()

    # Filter out internal fields for display
    models_list = []
    for m in loaded:
        models_list.append({
            "id": m["id"],
            "ref_count": m["ref_count"],
            "idle_seconds": m["idle_seconds"],
            "queue_active": m["queue_active"],
            "queue_waiting": m["queue_waiting"],
            "queue_available": m["queue_available"],
        })

    # Count active sessions by model
    sessions_by_model: dict[str, int] = {}
    for s in _active_sessions.values():
        if s["status"] == "active":
            sessions_by_model[s["model"]] = sessions_by_model.get(s["model"], 0) + 1

    return {
        "server": {
            "uptime_seconds": int(time.time() - __import__("os").path.getctime(__file__)),
            "version": "0.1.0",
            "mode": settings.backend,
            "loaded_models_count": len(loaded),
        },
        "gpu": gpu_info or {"available": False, "message": "Not detected (nvidia-smi not found)"},
        "models": models_list,
        "sessions": {
            "total": len(_active_sessions),
            "active": sum(1 for s in _active_sessions.values() if s["status"] == "active"),
            "by_model": sessions_by_model,
        },
    }


@router.get("/admin/api/sessions")
async def admin_list_sessions():
    return {
        "sessions": [
            {
                "id": s["id"],
                "user": s["user"],
                "model": s["model"],
                "type": s["type"],
                "elapsed_seconds": int(time.time() - s["started_at"]),
                "tokens": s.get("tokens", 0),
                "status": s["status"],
            }
            for s in sorted(
                _active_sessions.values(),
                key=lambda x: x["started_at"],
                reverse=True,
            )
        ]
    }


@router.post("/admin/api/sessions/{session_id}/kick")
async def admin_kick_session(session_id: str):
    session = _active_sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    session["status"] = "kicked"
    # In a real implementation, we would cancel the conversation task here
    # For now, mark it as kicked
    return {"status": "kicked", "session_id": session_id}


@router.get("/admin/api/models")
async def admin_list_models():
    discovered = registry.discover_models()
    loaded = {m["id"]: m for m in registry.get_loaded_models()}
    store = ModelStore.get_instance()
    modelfiles = store.list_modelfiles()

    models = []
    for d in discovered:
        mid = d["id"]
        info = {
            "id": mid,
            "loaded": mid in loaded,
            "path": d["path"],
            "size_gb": round(d["size"] / 1_000_000_000, 2) if d["size"] else 0,
            "queue_active": loaded[mid]["queue_active"] if mid in loaded else 0,
            "queue_waiting": loaded[mid]["queue_waiting"] if mid in loaded else 0,
            "ref_count": loaded[mid]["ref_count"] if mid in loaded else 0,
        }
        models.append(info)

    for mf in modelfiles:
        models.append({
            "id": mf["name"],
            "loaded": False,
            "path": f"modelfile:{mf['base_model']}",
            "size_gb": 0,
            "queue_active": 0,
            "queue_waiting": 0,
            "ref_count": 0,
            "modelfile": True,
            "base_model": mf["base_model"],
        })

    return {"models": models}


@router.post("/admin/api/models/{model_id}/unload")
async def admin_unload_model(model_id: str):
    try:
        await registry.unload_engine(model_id)
        return {"status": "unloaded", "model_id": model_id}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/admin/api/metrics")
async def admin_metrics():
    loaded = registry.get_loaded_models()
    gpu_info = await gpu_monitor.get_info()

    total_active = sum(m.get("queue_active", 0) for m in loaded)
    total_waiting = sum(m.get("queue_waiting", 0) for m in loaded)

    return {
        "gpu": gpu_info,
        "queue": {
            "total_active": total_active,
            "total_waiting": total_waiting,
            "total_loaded_models": len(loaded),
        },
        "sessions": {
            "active": sum(1 for s in _active_sessions.values() if s["status"] == "active"),
            "total_today": len(_active_sessions),
        },
    }


@router.get("/admin/api/config")
async def admin_config():
    return {
        "host": settings.host,
        "port": settings.port,
        "backend": settings.backend,
        "keep_alive": settings.keep_alive,
        "context_length": settings.context_length,
        "max_loaded_models": settings.max_loaded_models,
        "max_concurrent_per_model": settings.max_concurrent_per_model,
        "max_queue_size": settings.max_queue_size,
        "inference_timeout": settings.inference_timeout,
        "rate_limit_per_minute": settings.rate_limit_per_minute,
        "models_dir": settings.models_dir,
    }
