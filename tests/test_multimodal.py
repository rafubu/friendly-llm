from __future__ import annotations

from litert_ollama.multimodal import (
    parse_openai_content,
    parse_ollama_images,
    needs_vision_backend,
    needs_audio_backend,
)


def test_parse_openai_text():
    result = parse_openai_content("Hello")
    assert len(result) == 1
    assert result[0]["type"] == "text"
    assert result[0]["text"] == "Hello"


def test_parse_openai_content_list():
    content = [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAECAwQFBg=="}},
    ]
    result = parse_openai_content(content)
    assert len(result) == 2
    assert result[0]["type"] == "text"
    assert result[1]["type"] == "image"


def test_parse_openai_audio():
    content = [
        {"type": "text", "text": "Transcribe this"},
        {"type": "input_audio", "input_audio": {"data": "base64audio"}},
    ]
    result = parse_openai_content(content)
    assert result[1]["type"] == "audio"
    assert result[1]["blob"] == "base64audio"


def test_needs_vision():
    assert needs_vision_backend([{"content": [{"type": "image_url"}]}])
    assert needs_vision_backend([{"images": ["base64image"]}])
    assert not needs_vision_backend([{"content": "text only"}])


def test_needs_audio():
    assert needs_audio_backend([{"content": [{"type": "audio"}]}])
    assert not needs_audio_backend([{"content": [{"type": "text", "text": "hi"}]}])


def test_parse_ollama_images_empty():
    assert parse_ollama_images([]) == []
