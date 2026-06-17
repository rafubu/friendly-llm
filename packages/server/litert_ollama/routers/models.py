from __future__ import annotations

import os
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from ..config import settings
from ..engine_manager import registry
from ..model_store import ModelStore

router = APIRouter()


@router.get("/api/tags")
async def list_models():
    discovered = registry.discover_models()
    store = ModelStore.get_instance()
    stored = store.list_models()

    merged = {}
    for d in discovered:
        merged[d["id"]] = d
    for s in stored:
        if s["id"] not in merged:
            merged[s["id"]] = s

    tags = []
    for mid, info in merged.items():
        modified_ts = info.get("modified", info.get("modified_at"))
        if isinstance(modified_ts, (int, float)):
            modified_at = datetime.fromtimestamp(modified_ts, tz=timezone.utc).isoformat()
        else:
            modified_at = str(modified_ts or "")

        size = info.get("size", 0)
        path = info.get("path", "")
        backend_constraint = ""

        if path:
            try:
                from litert_lm_builder import litertlm_peek
                import io
                with io.StringIO() as buf:
                    meta = litertlm_peek.read_litertlm_header(path, buf)
            except Exception:
                meta = None
            if meta:
                backend_constraint = "gpu" if meta.SectionMetadata() and any(
                    litertlm_peek.get_model_type(meta.SectionMetadata().Objects(i)) == "tf_lite_artisan_text_decoder"
                    for i in range(meta.SectionMetadata().ObjectsLength())
                ) else "cpu"

        capabilities = ["completion"]
        if backend_constraint == "gpu":
            capabilities.append("vision")

        details = {
            "parent_model": "",
            "format": "litertlm",
            "family": "gemma",
            "parameter_size": "",
            "quantization_level": "",
            "backend_constraint": backend_constraint,
        }

        tags.append({
            "name": mid,
            "model": mid,
            "modified_at": modified_at,
            "size": size,
            "digest": info.get("digest", ""),
            "details": details,
            "capabilities": capabilities,
        })

    return {"models": tags}


@router.post("/api/show")
async def show_model(req_body: dict):
    model_id = req_body.get("model", "")
    if not model_id:
        raise HTTPException(400, "Missing model")

    path = registry.find_model_path(model_id)
    if not path:
        store = ModelStore.get_instance()
        mf = store.get_modelfile(model_id)
        if mf:
            return {
                "modelfile": "",
                "parameters": mf.get("parameters", "{}"),
                "template": mf.get("template", ""),
                "details": {"parent_model": mf.get("base_model", ""), "format": "modelfile", "family": "custom"},
                "model_info": {},
                "capabilities": ["completion"],
            }
        raise HTTPException(404, f"Model {model_id!r} not found")

    try:
        from litert_lm_builder import litertlm_peek
        import io
        with io.StringIO() as buf:
            metadata = litertlm_peek.read_litertlm_header(path, buf)
    except Exception as e:
        metadata = None

    capabilities = ["completion"]
    details = {
        "parent_model": "",
        "format": "litertlm",
        "family": "gemma",
        "parameter_size": "",
        "quantization_level": "",
    }

    if metadata:
        try:
            section = metadata.SectionMetadata()
            if section:
                for i in range(section.ObjectsLength()):
                    obj = section.Objects(i)
                    if obj and obj.ItemsLength():
                        for j in range(obj.ItemsLength()):
                            item = litertlm_peek.kvp_to_dict(obj.Items(j))
                            if item.get("key") == "backend_constraint":
                                capabilities.append("vision")
        except Exception:
            pass

    return {
        "modelfile": "",
        "parameters": "",
        "template": "",
        "details": details,
        "model_info": {},
        "capabilities": capabilities,
    }


@router.delete("/api/delete")
async def delete_model(req_body: dict):
    model_id = req_body.get("model", "")
    if not model_id:
        raise HTTPException(400, "Missing model")

    path = registry.find_model_path(model_id)
    if not path:
        raise HTTPException(404, f"Model {model_id!r} not found")

    await registry.unload_engine(model_id)

    model_file = path
    if os.path.exists(model_file):
        os.remove(model_file)

    model_dir = os.path.dirname(model_file)
    for f in os.listdir(model_dir):
        if f.endswith(".bin") or f.endswith(".cache"):
            try:
                os.remove(os.path.join(model_dir, f))
            except OSError:
                pass

    store = ModelStore.get_instance()
    store.remove_model(model_id)

    return {"status": "success"}


@router.post("/api/copy")
async def copy_model(req_body: dict):
    source = req_body.get("source", "")
    destination = req_body.get("destination", "")
    if not source or not destination:
        raise HTTPException(400, "Missing source or destination")

    src_path = registry.find_model_path(source)
    if not src_path:
        raise HTTPException(404, f"Source model {source!r} not found")

    import shutil

    safe_name = destination.replace("/", "--").replace("\\", "").replace("..", "").replace("~", "").strip()
    if not safe_name:
        raise HTTPException(400, "Invalid destination name")

    base_dir = os.path.dirname(src_path)
    dest_path = os.path.join(base_dir, safe_name, "model.litertlm")
    if not os.path.realpath(dest_path).startswith(os.path.realpath(base_dir)):
        raise HTTPException(400, "Destination path outside models directory")

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.copy2(src_path, dest_path)

    store = ModelStore.get_instance()
    store.add_model(destination, destination, dest_path, os.path.getsize(dest_path))

    return {"status": "success"}


@router.get("/v1/models")
async def openai_list_models():
    discovered = registry.discover_models()
    data = []
    for d in discovered:
        data.append({
            "id": d["id"],
            "object": "model",
            "created": int(d.get("modified", 0)),
            "owned_by": "litert-ollama",
        })
        data.append({
            "id": f"{d['id']},gpu",
            "object": "model",
            "created": int(d.get("modified", 0)),
            "owned_by": "litert-ollama",
        })
    return {"object": "list", "data": data}
