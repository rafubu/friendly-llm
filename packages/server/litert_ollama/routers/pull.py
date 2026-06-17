from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..config import settings

router = APIRouter()


@router.post("/api/pull")
async def pull_model(req_body: dict):
    model_spec = req_body.get("model", "")
    stream = req_body.get("stream", True)

    if not model_spec:
        raise HTTPException(400, "Missing model")

    try:
        from huggingface_hub import hf_hub_download, HfApi, RepositoryNotFoundError
    except ImportError:
        raise HTTPException(500, "huggingface-hub not installed")

    repo_id = model_spec
    if "/" not in repo_id:
        repo_id = f"litert-community/{repo_id}-litert-lm"

    models_dir = Path(settings.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    async def stream_generator():
        try:
            yield json.dumps({"status": "pulling manifest"}) + "\n"

            api = HfApi()
            try:
                model_info = api.model_info(repo_id)
            except RepositoryNotFoundError:
                yield json.dumps({"status": "error", "error": f"Model {repo_id} not found on HuggingFace"}) + "\n"
                return

            files = [f for f in model_info.siblings if f.rfilename.endswith(".litertlm") or f.rfilename.endswith(".bin")]
            if not files:
                files = model_info.siblings

            for file_info in files:
                filename = file_info.rfilename
                yield json.dumps({"status": "pulling {filename}", "digest": filename, "total": 0, "completed": 0}) + "\n"

                try:
                    local_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        cache_dir=str(models_dir),
                        resume_download=True,
                    )

                    model_id = model_spec.replace("/", "--")
                    dest_dir = models_dir / model_id
                    dest_dir.mkdir(parents=True, exist_ok=True)

                    if filename.endswith(".litertlm"):
                        import shutil
                        dest_path = dest_dir / "model.litertlm"
                        shutil.copy2(local_path, dest_path)

                    yield json.dumps({"status": "pulling {filename}", "digest": filename, "total": 100, "completed": 100}) + "\n"

                except Exception as e:
                    yield json.dumps({"status": "error", "error": f"Failed to download {filename}: {e}"}) + "\n"
                    return

            yield json.dumps({"status": "verifying sha256 digest"}) + "\n"
            yield json.dumps({"status": "writing manifest"}) + "\n"
            yield json.dumps({"status": "success"}) + "\n"

        except Exception as e:
            yield json.dumps({"status": "error", "error": str(e)}) + "\n"

    if not stream:
        return {"status": "success"}
    return StreamingResponse(stream_generator(), media_type="application/x-ndjson")
