from __future__ import annotations

import base64
import mimetypes
import os
import re
from typing import Any

import litert_lm
from litert_lm import Content


def parse_openai_content(
    content: str | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]

    result = []
    for part in content:
        t = part.get("type", "")
        if t == "text":
            result.append(part)
        elif t == "image_url":
            url = part.get("image_url", {}).get("url", "")
            converted = _convert_image_url(url)
            if converted:
                result.append(converted)
        elif t == "input_audio":
            data = part.get("input_audio", {}).get("data", "")
            result.append({"type": "audio", "blob": data})
        elif t == "image":
            result.append(part)
        elif t == "audio":
            result.append(part)
        else:
            result.append(part)
    return result


def _convert_image_url(url: str) -> dict[str, Any] | None:
    if not url:
        return None
    if url.startswith("data:"):
        try:
            header, data = url.split(",", 1)
            if "base64" in header:
                return {"type": "image", "blob": data}
        except ValueError:
            pass
        return None
    if url.startswith(("http://", "https://")):
        import urllib.request
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                raw = resp.read()
                b64 = base64.b64encode(raw).decode("utf-8")
                return {"type": "image", "blob": b64}
        except Exception:
            return None
    if url.startswith("file://"):
        path = url[7:]
        if os.path.exists(path):
            return {"type": "image", "path": os.path.abspath(path)}
        return None
    if os.path.exists(url):
        return {"type": "image", "path": os.path.abspath(url)}
    return None


def parse_ollama_images(
    images: list[str],
) -> list[Content]:
    if not images:
        return []
    result = []
    for img in images:
        try:
            raw = base64.b64decode(img)
            result.append(Content.ImageBytes(raw))
        except Exception:
            pass
    return result


def parse_ollama_message_images(
    msg: dict[str, Any],
) -> list[Content]:
    images = msg.get("images", [])
    if not images:
        return []
    return parse_ollama_images(images)


def detect_media_type(path: str) -> str | None:
    mime, _ = mimetypes.guess_type(path)
    if mime:
        if mime.startswith("audio/"):
            return "audio"
        if mime.startswith("image/"):
            return "image"
    ext = os.path.splitext(path)[1].lower()
    image_exts = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
    audio_exts = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac"}
    if ext in image_exts:
        return "image"
    if ext in audio_exts:
        return "audio"
    return None


def needs_vision_backend(messages: list[dict[str, Any]]) -> bool:
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if part.get("type") in ("image", "image_url"):
                    return True
        if msg.get("images"):
            return True
    return False


def needs_audio_backend(messages: list[dict[str, Any]]) -> bool:
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if part.get("type") in ("audio", "input_audio"):
                    return True
    return False
