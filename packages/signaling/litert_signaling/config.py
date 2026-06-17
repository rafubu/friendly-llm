from __future__ import annotations

import secrets
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LITERT_SIGNALING_", env_file=".env")

    host: str = Field(default="0.0.0.0", description="Signaling server host")
    port: int = Field(default=9876, description="Signaling server port")
    db_path: str = Field(
        default=str(Path.home() / ".litert-signaling" / "signaling.db"),
        description="Database path",
    )
    jwt_secret: str = Field(
        default="",
        description="JWT signing secret. Auto-generated if empty.",
    )
    rate_limit_per_minute: int = Field(default=60, description="Max WebSocket messages per minute")

    @field_validator("jwt_secret", mode="before")
    @classmethod
    def _ensure_secret(cls, v: str) -> str:
        if not v or v == "change-me-in-production":
            gen = secrets.token_hex(32)
            print(f"[WARNING] JWT secret not set. Generated: {gen}")
            print("[WARNING] Set LITERT_SIGNALING_JWT_SECRET env var to persist it.")
            return gen
        return v


settings = Settings()
