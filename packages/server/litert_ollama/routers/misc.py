from __future__ import annotations

from fastapi import APIRouter

from ..engine_manager import registry

router = APIRouter()


@router.get("/api/version")
async def version():
    import litert_lm
    return {"version": "0.1.0", "litert_lm_version": litert_lm.__version__ if hasattr(litert_lm, "__version__") else "0.13.1"}


@router.get("/api/ps")
async def ps():
    loaded = registry.get_loaded_models()
    return {
        "models": [
            {
                "name": m["id"],
                "model": m["id"],
                "size": 0,
                "digest": "",
                "details": {
                    "parent_model": "",
                    "format": "litertlm",
                    "family": "",
                    "parameter_size": "",
                    "quantization_level": "",
                },
                "expires_at": "",
                "size_vram": 0,
            }
            for m in loaded
        ]
    }
