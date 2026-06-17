from __future__ import annotations

import datetime
import json
import time
from typing import Any

from fastapi import APIRouter, HTTPException
import litert_lm

from ..engine_manager import registry
from ..schemas import EmbedRequest, OpenAIEmbeddingResponse, OpenAIEmbeddingData

router = APIRouter()


@router.post("/api/embed")
async def embed(req: EmbedRequest):
    if not req.model:
        raise HTTPException(400, "Missing model")
    if not req.input:
        raise HTTPException(400, "Missing input")

    inputs = [req.input] if isinstance(req.input, str) else req.input

    try:
        entry = await registry.load_engine(req.model)
        embeddings_list = []

        for text in inputs:
            tokens = entry.engine.tokenize(text)
            session = entry.engine.create_session()
            session.__enter__()

            try:
                session.run_prefill([text])
                result = session.run_text_scoring([text])
                if result.texts:
                    import struct
                    scores = struct.unpack(f"{len(text)}f", result.texts[0].encode("utf-16-le")[: len(text) * 4])
                else:
                    scores = [0.0] * 768
            finally:
                session.__exit__(None, None, None)

            embeddings_list.append([float(s) for s in scores])

        await registry.release_engine(req.model)

        return {
            "model": req.model,
            "embeddings": embeddings_list,
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_count": sum(len(t) for t in inputs),
        }

    except Exception as e:
        await registry.release_engine(req.model)
        raise HTTPException(500, str(e))


@router.post("/api/embeddings")
async def embed_legacy(req_body: dict):
    prompt = req_body.get("prompt", "")
    model = req_body.get("model", "")
    if not model:
        raise HTTPException(400, "Missing model")
    if not prompt:
        raise HTTPException(400, "Missing prompt")

    embed_req = EmbedRequest(model=model, input=prompt)
    result = await embed(embed_req)

    if isinstance(result, dict) and result.get("embeddings"):
        return {"embedding": result["embeddings"][0]}

    raise HTTPException(500, "Embedding failed")


@router.post("/v1/embeddings")
async def openai_embed(req_body: dict):
    model = req_body.get("model", "")
    inp = req_body.get("input", "")
    if not model:
        raise HTTPException(400, "Missing model")
    if not inp:
        raise HTTPException(400, "Missing input")

    embed_req = EmbedRequest(model=model, input=inp)
    result = await embed(embed_req)

    if isinstance(result, dict) and result.get("embeddings"):
        data = []
        for i, emb in enumerate(result["embeddings"]):
            data.append(OpenAIEmbeddingData(embedding=emb, index=i))
        return OpenAIEmbeddingResponse(data=data, model=model, usage={"prompt_tokens": 0, "total_tokens": 0})

    raise HTTPException(500, "Embedding failed")
