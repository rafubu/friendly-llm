from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..model_store import ModelStore
from ..modelfile import parse_modelfile, generate_modelfile_string, ModelfileParseError
from ..schemas import ModelfileSpec

router = APIRouter()


@router.post("/api/create")
async def create_model(req_body: dict):
    name = req_body.get("model", "")
    modelfile_text = req_body.get("modelfile", "")
    stream = req_body.get("stream", True)

    if not name:
        raise HTTPException(400, "Missing model name")
    if not modelfile_text:
        raise HTTPException(400, "Missing modelfile content")

    try:
        spec = parse_modelfile(modelfile_text)
    except ModelfileParseError as e:
        raise HTTPException(400, f"Invalid modelfile: {e}")

    if not spec.from_model:
        raise HTTPException(400, "FROM is required in modelfile")

    store = ModelStore.get_instance()

    base_model = store.get_model(spec.from_model)
    if not base_model:
        path_match = None
        from ..engine_manager import registry as reg
        path = reg.find_model_path(spec.from_model)
        if path:
            store.add_model(spec.from_model, spec.from_model, path)
            base_model = {"id": spec.from_model}
        else:
            raise HTTPException(400, f"Base model {spec.from_model!r} not found")

    spec_dict = spec.model_dump()
    mf_id = store.add_modelfile(name, spec_dict)

    async def stream_generator():
        yield json.dumps({"status": "reading model metadata"}) + "\n"
        for msg in spec.messages:
            yield json.dumps({"status": f"adding {msg.get('role', '')} message"}) + "\n"
        key, val_items = list(spec.parameters.items()), []
        for k, v in spec.parameters.items():
            yield json.dumps({"status": f"setting parameter {k}={v}"}) + "\n"
        yield json.dumps({"status": "success"}) + "\n"

    if not stream:
        return {"status": "success"}

    return StreamingResponse(
        stream_generator(),
        media_type="application/x-ndjson",
    )
