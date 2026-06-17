from __future__ import annotations

from litert_ollama.schemas import (
    GenerateRequest,
    ChatRequest,
    ChatCompletionResponse,
    Choice,
    ChatMessage,
    ModelTag,
    TagResponse,
    ShowRequest,
    EmbedRequest,
    CreateRequest,
    ModelfileSpec,
)


def test_generate_request_defaults():
    req = GenerateRequest(model="test", prompt="Hello")
    assert req.model == "test"
    assert req.prompt == "Hello"
    assert req.stream is True
    assert req.format is None
    assert req.images == []


def test_generate_request_with_options():
    req = GenerateRequest(
        model="test",
        prompt="Hi",
        format="json",
        options={"temperature": 0.7, "top_p": 0.9},
    )
    assert req.format == "json"
    assert req.options["temperature"] == 0.7


def test_chat_request():
    req = ChatRequest(
        model="test",
        messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ],
    )
    assert len(req.messages) == 2


def test_chat_completion_response():
    resp = ChatCompletionResponse(
        id="test_1",
        created=1234567890,
        model="gemma4-12b",
        choices=[Choice(message=ChatMessage(content="Hello world"))],
        usage={"prompt_tokens": 10, "completion_tokens": 3},
    )
    assert resp.choices[0].message.content == "Hello world"
    assert resp.usage["completion_tokens"] == 3


def test_tag_response():
    tag = ModelTag(name="gemma4-12b", modified_at="2026-01-01", size=2_000_000_000)
    resp = TagResponse(models=[tag])
    assert len(resp.models) == 1
    assert resp.models[0].name == "gemma4-12b"


def test_show_request():
    req = ShowRequest(model="gemma4-12b", verbose=True)
    assert req.model == "gemma4-12b"
    assert req.verbose is True


def test_embed_request():
    req = EmbedRequest(model="test", input="Hello world")
    assert req.input == "Hello world"


def test_embed_request_list():
    req = EmbedRequest(model="test", input=["Hello", "World"])
    assert isinstance(req.input, list)


def test_create_request():
    req = CreateRequest(model="my-model", modelfile="FROM gemma4-12b")
    assert req.model == "my-model"
    assert "FROM" in req.modelfile


def test_modelfile_spec():
    spec = ModelfileSpec(
        from_model="gemma4-12b",
        parameters={"temperature": 0.7},
        system="You are helpful",
        messages=[{"role": "user", "content": "Hi"}],
    )
    assert spec.from_model == "gemma4-12b"
    assert spec.system == "You are helpful"
    assert len(spec.messages) == 1
