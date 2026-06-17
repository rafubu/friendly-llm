from __future__ import annotations

import asyncio
import time
from collections import defaultdict

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from .config import settings


class SlidingWindowRateLimiter:
    """In-memory sliding window rate limiter.
    Tracks requests per IP within a time window.
    """

    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        self._max = max_requests
        self._window = window_seconds
        self._clients: dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def check(self, client_ip: str) -> bool:
        """Returns True if request is allowed, False if rate limited."""
        now = time.time()
        cutoff = now - self._window

        async with self._lock:
            timestamps = self._clients[client_ip]
            # Prune old entries
            while timestamps and timestamps[0] < cutoff:
                timestamps.pop(0)

            if len(timestamps) >= self._max:
                return False

            timestamps.append(now)
            return True


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that applies rate limiting."""

    def __init__(self, app, max_requests: int = 30, window_seconds: int = 60):
        super().__init__(app)
        self._limiter = SlidingWindowRateLimiter(
            max_requests=max_requests,
            window_seconds=window_seconds,
        )

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health, static files, and admin
        path = request.url.path
        if path in ("/health", "/", "/api/version") or path.startswith(
            ("/admin", "/static", "/v1/models")
        ):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        allowed = await self._limiter.check(client_ip)

        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please wait before retrying.",
                headers={"Retry-After": "30"},
            )

        return await call_next(request)
