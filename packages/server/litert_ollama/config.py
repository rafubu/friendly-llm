from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LITERT_", env_file=".env")

    host: str = Field(default="127.0.0.1", description="Server host")
    port: int = Field(default=11434, description="Server port")
    models_dir: str = Field(
        default=str(Path.home() / ".litert-lm" / "models"),
        description="Directory for .litertlm model files",
    )
    backend: Literal["cpu", "gpu", "auto"] = Field(
        default="auto", description="Inference backend"
    )
    keep_alive: str = Field(
        default="5m", description="How long to keep idle models loaded"
    )
    context_length: int = Field(
        default=32768, description="Maximum context length in tokens"
    )
    max_loaded_models: int = Field(
        default=3, description="Maximum models loaded simultaneously"
    )
    max_concurrent_per_model: int = Field(
        default=2, description="Maximum concurrent inference requests per model"
    )
    max_queue_size: int = Field(
        default=20, description="Maximum queued requests before rejecting"
    )
    inference_timeout: int = Field(
        default=300, description="Maximum inference time per request in seconds"
    )
    max_output_tokens: int = Field(
        default=8192, description="Maximum output tokens per request"
    )
    rate_limit_per_minute: int = Field(
        default=30, description="Max requests per minute per client"
    )


settings = Settings()
