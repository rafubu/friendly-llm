from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    text: str = ""
    done: bool = False
    done_reason: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    eval_count: int | None = None
    total_duration: int | None = None


@dataclass
class Response:
    text: str = ""
    tool_calls: list[dict[str, Any]] | None = None
    usage: dict[str, int] = field(default_factory=dict)


@dataclass
class ModelInfo:
    id: str
    node: str = ""
    name: str = ""
    load: int = 0
    max_load: int = 5
    visibility: str = "public"


@dataclass
class NodeInfo:
    node_id: str
    name: str
    models: list[ModelInfo] = field(default_factory=list)
    load: int = 0
    max_load: int = 5
    vram_free_mb: int = 0
