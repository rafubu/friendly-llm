from __future__ import annotations

import datetime
import json
import time
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import litert_lm

from ..engine_manager import registry, InferenceQueueFull
from ..config import settings
from ..schemas import GenerateRequest, GenerateChunk
from ..multimodal import parse_ollama_images, needs_vision_backend, needs_audio_backend

router = APIRouter()


def _parse_sampler_options(options: dict[str, Any]) -> litert_lm.SamplerConfig | None:
    temp = options.get("temperature")
    top_p = options.get("top_p")
    top_k = options.get("top_k")
    seed = options.get("seed")
    if all(v is None for v in (temp, top_p, top_k, seed)):
        return None
    return litert_lm.SamplerConfig(temperature=temp, top_p=top_p, top_k=top_k, seed=seed)


def _build_tools(tools: list[dict[str, Any]] | None) -> list[litert_lm.Tool] | None:
    if not tools:
        return None
    result = []
    for t in tools:
        if t.get("type") == "function":
            class _ProxyTool(litert_lm.Tool):
                def get_tool_description(self) -> dict[str, Any]:
                    return t
                def execute(self, param: Any) -> Any:
                    raise NotImplementedError("Proxy tools are not executable")
            result.append(_ProxyTool())
    return result if result else None


def _estimate_prompt_tokens(prompt) -> int:
    if isinstance(prompt, str):
        return max(len(prompt) // 4, 1)
    if isinstance(prompt, dict):
        content = prompt.get("content", "")
        if isinstance(content, str):
            return max(len(content) // 4, 1)
        if isinstance(content, list):
            return sum(max(len(p.get("text", "")) // 4, 1) for p in content if isinstance(p, dict))
    return 0


@router.post("/api/generate")
async def generate_endpoint(req: GenerateRequest):
    if not req.model:
        raise HTTPException(400, "Missing model")
    if not req.prompt and not req.images:
        raise HTTPException(400, "Missing prompt or images")

    conv = None
    engine_model_id = req.model
    entry = None

    context_limit = req.options.get("num_ctx", settings.context_length)
    estimated_tokens = _estimate_prompt_tokens(req.prompt) + len(req.images or []) * 144

    if estimated_tokens > context_limit:
        overflow_response = {
            "model": req.model,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "response": "",
            "done": True,
            "done_reason": "context_overflow",
            "context_info": {
                "estimated_input_tokens": estimated_tokens,
                "context_limit": context_limit,
                "overflow_by": estimated_tokens - context_limit,
                "suggestion": "Reduce prompt length or set options.num_ctx higher.",
            },
        }
        if req.stream:
            return StreamingResponse(
                iter([json.dumps(overflow_response) + "\n"]),
                media_type="application/x-ndjson",
            )
        return JSONResponse(content=overflow_response)

    try:
        needs_vision = bool(req.images) or needs_vision_backend(
            [{"content": [{"type": "image_url"}]}] if req.images else []
        )

        entry = await registry.load_engine(
            req.model,
            vision_backend=litert_lm.Backend.CPU() if needs_vision else None,
            enable_speculative_decoding=settings.enable_speculative_decoding if settings.enable_speculative_decoding else False,
            max_num_tokens=context_limit,
        )
        engine_model_id = req.model

        sampler_config = _parse_sampler_options(req.options) if req.options else None

        messages = []
        if req.system:
            messages.append({"role": "system", "content": [{"type": "text", "text": req.system}]})

        conv = entry.engine.create_conversation(
            messages=messages or None,
            tools=None,
            automatic_tool_calling=False,
            sampler_config=sampler_config,
            enable_constrained_decoding=(req.format == "json"),
        )
        conv.__enter__()

        # Acquire queue slot (blocks if busy, raises 503 if full)
        await entry.queue.acquire()

        images = parse_ollama_images(req.images)

        if images:
            content_list = [{"type": "text", "text": req.prompt}]
            for img in images:
                if hasattr(img, "blob") and callable(getattr(img, "blob", None)):
                    blob_data = img.blob
                elif hasattr(img, "blob"):
                    blob_data = img.blob
                else:
                    blob_data = img
                content_list.append({"type": "image", "blob": blob_data})
            prompt = {"role": "user", "content": content_list}
        else:
            prompt = req.prompt

        now = datetime.datetime.now(datetime.timezone.utc)
        now_str = now.isoformat()

        async def stream_generator():
            start_time = time.perf_counter_ns()
            prompt_eval_count = 0
            slot_acquired = True
            try:
                has_context = False
                response_text = ""

                for chunk in conv.send_message_async(prompt):
                    text_out = "".join(
                        item.get("text", "")
                        for item in chunk.get("content", [])
                        if item.get("type") == "text"
                    )
                    if text_out:
                        response_text += text_out
                        yield json.dumps({
                            "model": req.model,
                            "created_at": now_str,
                            "response": text_out,
                            "done": False,
                        }) + "\n"
                        has_context = True

                if not has_context:
                    yield json.dumps({
                        "model": req.model,
                        "created_at": now_str,
                        "response": "",
                        "done": False,
                    }) + "\n"

                end_time = time.perf_counter_ns()
                eval_count = conv.token_count if conv else 0
                yield json.dumps({
                    "model": req.model,
                    "created_at": now_str,
                    "response": "",
                    "done": True,
                    "done_reason": "stop",
                    "context": [],
                    "total_duration": end_time - start_time,
                    "load_duration": 0,
                    "prompt_eval_count": eval_count or 1,
                    "prompt_eval_duration": 0,
                    "eval_count": eval_count or 1,
                    "eval_duration": end_time - start_time,
                    "context_used": eval_count,
                    "context_limit": context_limit,
                }) + "\n"

            except Exception as e:
                yield json.dumps({
                    "model": req.model,
                    "created_at": now_str,
                    "response": "",
                    "done": True,
                    "done_reason": "error",
                    "error": str(e),
                }) + "\n"
            finally:
                try:
                    conv.__exit__(None, None, None)
                except Exception:
                    pass
                try:
                    await registry.release_engine(engine_model_id)
                except Exception:
                    pass
                entry.queue.release()

        if not req.stream:
            full_response = ""
            async for chunk in stream_generator():
                data = json.loads(chunk)
                if not data.get("done"):
                    full_response += data.get("response", "")
                else:
                    return data
            return GenerateChunk(
                model=req.model,
                created_at=now_str,
                response=full_response,
                done=True,
                done_reason="stop",
            )

        return StreamingResponse(
            stream_generator(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache"},
        )

    except InferenceQueueFull:
        if conv:
            try:
                conv.__exit__(None, None, None)
            except Exception:
                pass
        try:
            await registry.release_engine(engine_model_id)
        except Exception:
            pass
        raise HTTPException(503, detail="Server at capacity", headers={"Retry-After": "5"})
    except Exception as e:
        if conv:
            try:
                conv.__exit__(None, None, None)
            except Exception:
                pass
        if entry and entry.queue.active > 0:
            entry.queue.release()
        try:
            await registry.release_engine(engine_model_id)
        except Exception:
            pass
        raise HTTPException(500, str(e))
