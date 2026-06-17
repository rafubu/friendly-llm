from __future__ import annotations

import datetime
import json
import time
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import litert_lm

from ..engine_manager import registry, InferenceQueueFull
from ..schemas import ChatRequest, ChatCompletionResponse, ChatMessage, Choice
from ..multimodal import (
    parse_openai_content,
    parse_ollama_message_images,
    needs_vision_backend,
    needs_audio_backend,
)

router = APIRouter()


def _translate_messages(messages: list[dict]) -> tuple[list[dict], dict | None]:
    if not messages:
        return [], None

    name_by_tool_call_id = {}
    for m in messages:
        if m.get("role") == "assistant" and "tool_calls" in m:
            for tc in m.get("tool_calls", []):
                tc_id = tc.get("id")
                func = tc.get("function", {})
                if tc_id and func.get("name"):
                    name_by_tool_call_id[tc_id] = func["name"]

    result = []
    last_msg = None
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")

        if role == "assistant" and "tool_calls" in m:
            translated_tc = [
                {
                    "type": "function",
                    "function": {
                        "name": tc.get("function", {}).get("name"),
                        "arguments": json.loads(tc.get("function", {}).get("arguments", "{}")),
                    },
                }
                for tc in m.get("tool_calls", [])
            ]
            result.append({"role": "assistant", "tool_calls": translated_tc, "content": content or None})
            last_msg = result[-1]
            continue

        if role == "tool":
            tc_id = m.get("tool_call_id", "")
            tool_name = name_by_tool_call_id.get(tc_id, m.get("tool_name", ""))
            result.append({
                "role": "tool",
                "content": [{"type": "tool_response", "name": tool_name, "response": content}],
            })
            last_msg = result[-1]
            continue

        if isinstance(content, list):
            translated = parse_openai_content(content)
            msg_dict = {"role": role, "content": translated}
        elif m.get("images"):
            translated = [{"type": "text", "text": content}]
            for img in parse_ollama_message_images(m):
                translated.append({"type": "image", "blob": img})
            msg_dict = {"role": role, "content": translated}
        else:
            msg_dict = {"role": role, "content": content}

        result.append(msg_dict)
        last_msg = result[-1]

    return result, last_msg


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


@router.post("/api/chat")
async def chat_ollama(req: ChatRequest):
    if not req.model:
        raise HTTPException(400, "Missing model")
    if not req.messages:
        raise HTTPException(400, "Missing messages")

    msg_dicts = [m.model_dump() for m in req.messages]
    context_messages, last_prompt = _translate_messages(msg_dicts)
    if not last_prompt:
        raise HTTPException(400, "No valid messages")

    vision = any(
        needs_vision_backend([m])
        for m in msg_dicts
    )
    audio = any(
        needs_audio_backend([m])
        for m in msg_dicts
    )

    conv = None
    entry = None
    try:
        entry = await registry.load_engine(
            req.model,
            vision_backend=litert_lm.Backend.CPU() if vision else None,
            audio_backend=litert_lm.Backend.CPU() if audio else None,
        )

        sampler_config = None
        if any(k in req.options for k in ("temperature", "top_p", "top_k", "seed")):
            sampler_config = litert_lm.SamplerConfig(
                temperature=req.options.get("temperature"),
                top_p=req.options.get("top_p"),
                top_k=req.options.get("top_k"),
                seed=req.options.get("seed"),
            )

        tool_list = _build_tools(req.tools)

        conv = entry.engine.create_conversation(
            messages=context_messages[:-1] if len(context_messages) > 1 else None,
            tools=tool_list or None,
            automatic_tool_calling=False,
            sampler_config=sampler_config,
            enable_constrained_decoding=(req.format == "json"),
        )
        conv.__enter__()

        # Acquire queue slot (blocks if busy, raises 503 if full)
        await entry.queue.acquire()

        now = datetime.datetime.now(datetime.timezone.utc)
        now_str = now.isoformat()

        async def stream_generator():
            start_time = time.perf_counter_ns()
            try:
                response_text = ""
                has_text = False

                for chunk in conv.send_message_async(last_prompt):
                    text_out = "".join(
                        item.get("text", "")
                        for item in chunk.get("content", [])
                        if item.get("type") == "text"
                    )
                    tool_calls = chunk.get("tool_calls", [])

                    if text_out:
                        response_text += text_out

                    openai_tc = None
                    if tool_calls:
                        openai_tc = [
                            {
                                "function": {
                                    "name": tc.get("function", {}).get("name"),
                                    "arguments": json.dumps(tc.get("function", {}).get("arguments", {})),
                                },
                            }
                            for tc in tool_calls
                        ]

                    chunk_data = {
                        "model": req.model,
                        "created_at": now_str,
                        "message": {
                            "role": "assistant",
                            "content": text_out,
                        },
                        "done": False,
                    }
                    if openai_tc:
                        chunk_data["message"]["tool_calls"] = openai_tc
                        has_text = True

                    if text_out or openai_tc:
                        yield json.dumps(chunk_data) + "\n"
                        has_text = True

                if not has_text:
                    yield json.dumps({
                        "model": req.model,
                        "created_at": now_str,
                        "message": {"role": "assistant", "content": ""},
                        "done": False,
                    }) + "\n"

                end_time = time.perf_counter_ns()
                yield json.dumps({
                    "model": req.model,
                    "created_at": now_str,
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "done_reason": "stop",
                    "total_duration": end_time - start_time,
                    "prompt_eval_count": 0,
                    "prompt_eval_duration": 0,
                    "eval_count": len(response_text.split()) or 1,
                    "eval_duration": end_time - start_time,
                }) + "\n"

            except Exception as e:
                yield json.dumps({
                    "model": req.model,
                    "created_at": now_str,
                    "message": {"role": "assistant", "content": ""},
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
                    await registry.release_engine(req.model)
                except Exception:
                    pass
                entry.queue.release()

        if not req.stream:
            full_text = ""
            for chunk in stream_generator():
                data = json.loads(chunk)
                if not data.get("done"):
                    full_text += data.get("message", {}).get("content", "")
            return {
                "model": req.model,
                "created_at": now_str,
                "message": {"role": "assistant", "content": full_text},
                "done": True,
                "done_reason": "stop",
            }

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
        if entry:
            try:
                await registry.release_engine(req.model)
            except Exception:
                pass
        raise HTTPException(503, detail="Server at capacity", headers={"Retry-After": "5"})
    except Exception as e:
        if conv:
            try:
                conv.__exit__(None, None, None)
            except Exception:
                pass
        if entry:
            if entry.queue.active > 0:
                entry.queue.release()
            try:
                await registry.release_engine(req.model)
            except Exception:
                pass
        raise HTTPException(500, str(e))


@router.post("/v1/chat/completions")
async def chat_openai(req: ChatRequest):
    ollama_resp = await chat_ollama(req)
    if isinstance(ollama_resp, dict):
        content = ollama_resp.get("message", {}).get("content", "")
        now = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
        return ChatCompletionResponse(
            id=f"chatcmpl_{now}",
            created=now,
            model=req.model,
            choices=[Choice(message=ChatMessage(content=content))],
        )
    if hasattr(ollama_resp, "body_iterator"):
        async def sse_stream():
            async for line in ollama_resp.body_iterator:
                try:
                    data = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if data.get("done"):
                    break
                msg = data.get("message", {})
                content = msg.get("content", "")
                now = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
                sse_data = {
                    "id": f"chatcmpl_{now}",
                    "object": "chat.completion.chunk",
                    "created": now,
                    "model": req.model,
                    "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(sse_data)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(sse_stream(), media_type="text/event-stream")
    return ollama_resp
