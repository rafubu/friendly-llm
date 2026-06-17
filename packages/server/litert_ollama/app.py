from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .engine_manager import registry
from .model_store import ModelStore
from .ratelimit import RateLimitMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    ModelStore.get_instance()
    await registry.start_gc()
    yield
    await registry.shutdown()


app = FastAPI(
    title="LiteRT-Ollama",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    RateLimitMiddleware,
    max_requests=settings.rate_limit_per_minute,
    window_seconds=60,
)


from .routers import generate, chat, models, create, embeddings, pull, misc, metrics, admin  # noqa: E402, F401


app.include_router(generate.router, tags=["Ollama"])
app.include_router(chat.router, tags=["Ollama"])
app.include_router(models.router, tags=["Ollama"])
app.include_router(create.router, tags=["Ollama"])
app.include_router(embeddings.router, tags=["Ollama"])
app.include_router(pull.router, tags=["Ollama"])
app.include_router(misc.router, tags=["Ollama"])
app.include_router(metrics.router, tags=["Ollama"])
app.include_router(admin.router, tags=["Admin"])


@app.get("/")
async def root():
    return {
        "name": "LiteRT-Ollama",
        "version": "0.1.0",
        "engine": "litert-lm",
        "endpoints": {
            "ollama": ["/api/generate", "/api/chat", "/api/tags", "/api/show", "/api/create", "/api/embed", "/api/pull", "/api/ps", "/api/version", "/api/copy", "/api/delete"],
            "openai": ["/v1/chat/completions", "/v1/models", "/v1/embeddings"],
        },
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
