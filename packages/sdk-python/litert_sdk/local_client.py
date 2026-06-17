from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from .errors import ConnectionError
from .types import Chunk, Response, ModelInfo


class LitertLocalClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11434", model: str = "", timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self._client.aclose()

    async def chat_sync(
        self,
        text: str,
        *,
        images: list[str | bytes] | None = None,
        tools: list[dict] | None = None,
        format: str | None = None,
    ) -> Response:
        payload = self._build_payload(text, images, tools, format, stream=False)
        try:
            resp = await self._client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return Response(
                text=data.get("message", {}).get("content", ""),
                tool_calls=data.get("message", {}).get("tool_calls"),
            )
        except httpx.HTTPError as e:
            raise ConnectionError(f"Request failed: {e}")

    async def chat(
        self,
        text: str,
        *,
        images: list[str | bytes] | None = None,
        tools: list[dict] | None = None,
        format: str | None = None,
    ) -> AsyncIterator[Chunk]:
        payload = self._build_payload(text, images, tools, format, stream=True)
        try:
            async with self._client.stream("POST", f"{self.base_url}/api/chat", json=payload) as resp:
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    msg = data.get("message", {})
                    if data.get("done"):
                        yield Chunk(
                            text="",
                            done=True,
                            done_reason=data.get("done_reason"),
                            eval_count=data.get("eval_count"),
                            total_duration=data.get("total_duration"),
                        )
                        break
                    yield Chunk(text=msg.get("content", ""), done=False)
        except httpx.HTTPError as e:
            raise ConnectionError(f"Request failed: {e}")

    async def generate(
        self,
        prompt: str,
        *,
        images: list[str | bytes] | None = None,
        options: dict | None = None,
        format: str | None = None,
    ) -> AsyncIterator[Chunk]:
        payload = {"model": self.model or "", "prompt": prompt, "stream": True}
        if images:
            import base64
            payload["images"] = []
            for img in images:
                if isinstance(img, bytes):
                    payload["images"].append(base64.b64encode(img).decode("utf-8"))
                elif img.startswith("/") or img.startswith("."):
                    with open(img, "rb") as f:
                        payload["images"].append(base64.b64encode(f.read()).decode("utf-8"))
                else:
                    with open(img, "rb") as f:
                        payload["images"].append(base64.b64encode(f.read()).decode("utf-8"))
        if options:
            payload["options"] = options
        if format:
            payload["format"] = format

        try:
            async with self._client.stream("POST", f"{self.base_url}/api/generate", json=payload) as resp:
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if data.get("done"):
                        yield Chunk(
                            text="",
                            done=True,
                            done_reason=data.get("done_reason"),
                            eval_count=data.get("eval_count"),
                        )
                        break
                    yield Chunk(text=data.get("response", ""), done=False)
        except httpx.HTTPError as e:
            raise ConnectionError(f"Request failed: {e}")

    async def embed(self, text: str) -> list[float]:
        payload = {"model": self.model or "", "input": text}
        try:
            resp = await self._client.post(f"{self.base_url}/api/embed", json=payload)
            resp.raise_for_status()
            data = resp.json()
            if data.get("embeddings"):
                return data["embeddings"][0]
            return []
        except httpx.HTTPError as e:
            raise ConnectionError(f"Request failed: {e}")

    async def list_models(self) -> list[ModelInfo]:
        try:
            resp = await self._client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            return [ModelInfo(id=m["name"]) for m in data.get("models", [])]
        except httpx.HTTPError as e:
            raise ConnectionError(f"Request failed: {e}")

    def _build_payload(
        self,
        text: str,
        images: list[str | bytes] | None,
        tools: list[dict] | None,
        format: str | None,
        stream: bool,
    ) -> dict[str, Any]:
        msg = {"role": "user", "content": text}
        if images:
            import base64
            b64_images = []
            for img in images:
                if isinstance(img, bytes):
                    b64_images.append(base64.b64encode(img).decode("utf-8"))
                else:
                    with open(img, "rb") as f:
                        b64_images.append(base64.b64encode(f.read()).decode("utf-8"))
            msg["images"] = b64_images

        payload: dict[str, Any] = {
            "model": self.model or "",
            "messages": [msg],
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools
        if format:
            payload["format"] = format
        return payload
