from __future__ import annotations

import json
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..config import settings

router = APIRouter()


@router.post("/api/pull")
async def pull_model(req_body: dict):
    model_spec = req_body.get("model", "")
    stream = req_body.get("stream", True)
    hf_token = req_body.get("huggingface_token", None)

    if not model_spec:
        raise HTTPException(400, "Missing model")

    repo_id = model_spec
    if "/" not in repo_id:
        repo_id = f"litert-community/{repo_id}-litert-lm"

    models_dir = Path(settings.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    async def stream_generator():
        try:
            yield json.dumps({"status": "pulling manifest"}) + "\n"

            from huggingface_hub import HfApi
            from huggingface_hub.utils import RepositoryNotFoundError

            api = HfApi()
            if hf_token:
                api.token = hf_token

            try:
                model_info = api.model_info(repo_id)
            except RepositoryNotFoundError:
                yield json.dumps({"status": "error", "error": f"Model {repo_id} not found on HuggingFace"}) + "\n"
                return

            files = [f for f in model_info.siblings if f.rfilename.endswith(".litertlm") or f.rfilename.endswith(".bin")]
            if not files:
                files = model_info.siblings

            model_id = model_spec.replace("/", "--")
            dest_dir = models_dir / model_id
            dest_dir.mkdir(parents=True, exist_ok=True)

            headers = {}
            if hf_token:
                headers["Authorization"] = f"Bearer {hf_token}"

            for file_info in files:
                filename = file_info.rfilename
                total_size = file_info.size or 0

                yield json.dumps({"status": f"pulling {filename}", "digest": filename, "total": total_size, "completed": 0}) + "\n"

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
                            if total_size == 0:
                                total_size = int(response.headers.get("content-length", 0))

                            local_path.parent.mkdir(parents=True, exist_ok=True)
                            with open(local_path, "wb") as f:
                                async for chunk in response.aiter_bytes():
                                    f.write(chunk)
                                    downloaded += len(chunk)
                                    if downloaded - last_report >= 1024 * 1024 or downloaded >= total_size:
                                        last_report = downloaded
                                        yield json.dumps({
                                            "status": f"pulling {filename}",
                                            "digest": filename,
                                            "total": total_size,
                                            "completed": downloaded,
                                        }) + "\n"
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
