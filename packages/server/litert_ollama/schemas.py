from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    model: str
    prompt: str = ""
    images: list[str] = Field(default_factory=list, description="Base64-encoded images")
    format: Literal["json", None] = None
    options: dict[str, Any] = Field(default_factory=dict)
    system: str | None = None
    template: str | None = None
    context: list[int] | None = None
    stream: bool = True
    raw: bool = False
    keep_alive: str | None = None


class Message(BaseModel):
    role: str
    content: str | list[dict[str, Any]] | None = ""
    images: list[str] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] | None = None
    tool_name: str | None = None
    tool_call_id: str | None = None


class ChatRequest(BaseModel):
    model: str
    messages: list[Message] = Field(default_factory=list)
    tools: list[dict[str, Any]] | None = None
    format: Literal["json", None] = None
    options: dict[str, Any] = Field(default_factory=dict)
    stream: bool = True
    keep_alive: str | None = None


class ToolCall(BaseModel):
    function: dict[str, Any]


class ChatMessage(BaseModel):
    role: str = "assistant"
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: dict[str, int] | None = None


class GenerateChunk(BaseModel):
    model: str
    created_at: str
    response: str = ""
    done: bool = False
    done_reason: str | None = None
    context: list[int] | None = None
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None
    context_used: int | None = None
    context_limit: int | None = None
    context_info: dict[str, Any] | None = None


class ChatChunk(BaseModel):
    model: str
    created_at: str
    message: Message | None = None
    done: bool = False
    done_reason: str | None = None
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None
    context_used: int | None = None
    context_limit: int | None = None
    context_info: dict[str, Any] | None = None


class ModelTag(BaseModel):
    name: str
    model: str | None = None
    modified_at: str
    size: int
    digest: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class TagResponse(BaseModel):
    models: list[ModelTag]


class ShowRequest(BaseModel):
    model: str
    verbose: bool = False


class ShowResponse(BaseModel):
    modelfile: str = ""
    parameters: str = ""
    template: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    model_info: dict[str, Any] = Field(default_factory=dict)
    capabilities: list[str] = Field(default_factory=list)


class CopyRequest(BaseModel):
    source: str
    destination: str


class DeleteRequest(BaseModel):
    model: str


class PullRequest(BaseModel):
    model: str
    insecure: bool = False
    stream: bool = True


class EmbedRequest(BaseModel):
    model: str
    input: str | list[str]
    truncate: bool = True
    options: dict[str, Any] = Field(default_factory=dict)
    keep_alive: str | None = None


class EmbedResponse(BaseModel):
    model: str
    embeddings: list[list[float]]
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None


class CreateRequest(BaseModel):
    model: str
    modelfile: str = ""
    stream: bool = True
    path: str | None = None


class ModelfileSpec(BaseModel):
    from_model: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    system: str | None = None
    template: str | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    license: str | None = None
    adapter: str | None = None


class OpenAIEmbeddingInput(BaseModel):
    model: str
    input: str | list[str]


class OpenAIEmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: list[float]
    index: int = 0


class OpenAIEmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[OpenAIEmbeddingData]
    model: str
    usage: dict[str, int]


class OpenAIModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "litert-ollama"


class OpenAIModelList(BaseModel):
    object: str = "list"
    data: list[OpenAIModelInfo]
