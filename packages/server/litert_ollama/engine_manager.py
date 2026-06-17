from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

import litert_lm

from .config import settings

logger = logging.getLogger(__name__)


class InferenceQueueFull(Exception):
    pass


class InferenceTimeout(Exception):
    pass


class InferenceQueue:
    """Controls concurrent inference per model via semaphore.
    Provides backpressure with a bounded queue.
    If queue is full, raises InferenceQueueFull (caller returns 503).
    """

    def __init__(self, max_concurrent: int, max_queue: int, timeout: float):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._waiters: asyncio.Queue[asyncio.Future] = asyncio.Queue(max_queue)
        self._timeout = timeout
        self._max_concurrent = max_concurrent
        self._active = 0

    async def acquire(self) -> None:
        """Block until a slot is available.
        If a slot is free, returns immediately.
        If all slots busy, queues the request and waits.
        If queue is full, raises InferenceQueueFull (caller returns 503).
        """
        if not self._semaphore.locked():
            await self._semaphore.acquire()
            self._active += 1
            return

        try:
            waiter: asyncio.Future = asyncio.get_event_loop().create_future()
            self._waiters.put_nowait(waiter)
            await waiter
            self._active += 1
        except asyncio.QueueFull:
            raise InferenceQueueFull("Server at capacity. Try again later.")

    def release(self) -> None:
        """Release slot and wake the next waiter if any."""
        if not self._waiters.empty():
            try:
                waiter = self._waiters.get_nowait()
                if not waiter.done():
                    waiter.set_result(True)
            except asyncio.QueueEmpty:
                pass
        self._semaphore.release()
        self._active -= 1

    def try_acquire(self) -> bool:
        """Non-blocking attempt. Returns True if slot acquired immediately."""
        acquired = self._semaphore.acquire_nowait()
        if acquired:
            self._active += 1
            return True
        return False

    def cancel_waiting(self):
        """Cancel all pending waiters (e.g. on shutdown)."""
        while not self._waiters.empty():
            try:
                w = self._waiters.get_nowait()
                if not w.done():
                    w.cancel()
            except asyncio.QueueEmpty:
                break

    @property
    def active(self) -> int:
        return self._active

    @property
    def waiting(self) -> int:
        return self._waiters.qsize()

    @property
    def max_concurrent(self) -> int:
        return self._max_concurrent

    @property
    def available_slots(self) -> int:
        return self._max_concurrent - self._active


@dataclass
class EngineEntry:
    engine: litert_lm.Engine
    model_id: str
    model_path: str
    backend: litert_lm.Backend
    last_used: float = field(default_factory=time.time)
    ref_count: int = 0
    queue: InferenceQueue = field(init=False)

    def __post_init__(self):
        self.queue = InferenceQueue(
            max_concurrent=settings.max_concurrent_per_model,
            max_queue=settings.max_queue_size,
            timeout=settings.inference_timeout,
        )


class ModelRegistry:
    def __init__(self):
        self._engines: dict[str, EngineEntry] = {}
        self._global_lock = asyncio.Lock()
        self._keep_alive_secs = self._parse_keep_alive(settings.keep_alive)
        self._gc_task: asyncio.Task | None = None

    def _parse_keep_alive(self, val: str) -> int:
        val = val.strip().lower()
        if val.endswith("s"):
            return int(val[:-1])
        if val.endswith("m"):
            return int(val[:-1]) * 60
        if val.endswith("h"):
            return int(val[:-1]) * 3600
        if val == "0" or val == "-1m":
            return 0
        return 300

    async def start_gc(self):
        if self._gc_task is None or self._gc_task.done():
            self._gc_task = asyncio.create_task(self._gc_loop())

    async def _gc_loop(self):
        while True:
            await asyncio.sleep(60)
            now = time.time()
            async with self._global_lock:
                to_delete = [
                    mid
                    for mid, entry in self._engines.items()
                    if entry.ref_count == 0
                    and (now - entry.last_used) > self._keep_alive_secs
                ]
                for mid in to_delete:
                    entry = self._engines.pop(mid)
                    entry.queue.cancel_waiting()
                    try:
                        entry.engine.__exit__(None, None, None)
                    except Exception:
                        pass

    def _select_backend(self, model_path: str) -> litert_lm.Backend:
        if settings.backend == "gpu":
            return litert_lm.Backend.GPU()
        if settings.backend == "cpu":
            return litert_lm.Backend.CPU()
        try:
            from litert_lm_builder import litertlm_peek
            import io

            with io.StringIO() as buf:
                meta = litertlm_peek.read_litertlm_header(model_path, buf)
            section = meta.SectionMetadata()
            if section:
                for i in range(section.ObjectsLength()):
                    obj = section.Objects(i)
                    if obj and litertlm_peek.get_model_type(obj) == "tf_lite_artisan_text_decoder":
                        return litert_lm.Backend.GPU()
        except Exception:
            pass
        return litert_lm.Backend.CPU()

    def discover_models(self) -> list[dict[str, Any]]:
        models_dir = Path(settings.models_dir)
        if not models_dir.exists():
            return []
        results = []
        for model_dir in models_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model_file = model_dir / "model.litertlm"
            if model_file.exists():
                size = model_file.stat().st_size
                modified = model_file.stat().st_mtime
                results.append({
                    "id": model_dir.name.replace("--", "/"),
                    "path": str(model_file),
                    "size": size,
                    "modified": modified,
                })
        return results

    def find_model_path(self, model_id: str) -> str | None:
        for entry in self.discover_models():
            if entry["id"] == model_id:
                return entry["path"]
        return None

    async def load_engine(
        self,
        model_id: str,
        *,
        vision_backend: litert_lm.Backend | None = None,
        audio_backend: litert_lm.Backend | None = None,
        enable_speculative_decoding: bool | None = None,
    ) -> EngineEntry:
        async with self._global_lock:
            if model_id in self._engines:
                entry = self._engines[model_id]
                entry.ref_count += 1
                entry.last_used = time.time()
                return entry

            # Enforce max_loaded_models
            loaded_count = len(self._engines)
            if loaded_count >= settings.max_loaded_models:
                # Try to GC idle models first
                now = time.time()
                idle = [
                    mid
                    for mid, e in self._engines.items()
                    if e.ref_count == 0
                ]
                if idle:
                    mid = idle[0]
                    old = self._engines.pop(mid)
                    old.queue.cancel_waiting()
                    try:
                        old.engine.__exit__(None, None, None)
                    except Exception:
                        pass
                    logger.info(f"Unloaded idle model {mid} to make room")
                else:
                    raise RuntimeError(
                        f"Max loaded models reached ({settings.max_loaded_models}). "
                        f"All models have active references."
                    )

            model_path = self.find_model_path(model_id)
            if not model_path:
                raise FileNotFoundError(f"Model {model_id!r} not found")

            backend = self._select_backend(model_path)
            engine = litert_lm.Engine(
                model_path,
                backend=backend,
                max_num_tokens=settings.context_length,
                vision_backend=vision_backend,
                audio_backend=audio_backend,
                enable_speculative_decoding=enable_speculative_decoding,
            )
            engine.__enter__()
            entry = EngineEntry(
                engine=engine,
                model_id=model_id,
                model_path=model_path,
                backend=backend,
            )
            entry.ref_count = 1
            self._engines[model_id] = entry
            return entry

    async def release_engine(self, model_id: str):
        async with self._global_lock:
            entry = self._engines.get(model_id)
            if entry:
                entry.ref_count = max(0, entry.ref_count - 1)
                entry.last_used = time.time()

    async def unload_engine(self, model_id: str):
        async with self._global_lock:
            entry = self._engines.pop(model_id, None)
            if entry:
                entry.queue.cancel_waiting()
                try:
                    entry.engine.__exit__(None, None, None)
                except Exception:
                    pass

    def get_loaded_models(self) -> list[dict[str, Any]]:
        now = time.time()
        result = []
        for mid, entry in list(self._engines.items()):
            result.append({
                "id": entry.model_id,
                "ref_count": entry.ref_count,
                "idle_seconds": int(now - entry.last_used),
                "queue_active": entry.queue.active,
                "queue_waiting": entry.queue.waiting,
                "queue_available": entry.queue.available_slots,
            })
        return result

    async def shutdown(self):
        if self._gc_task and not self._gc_task.done():
            self._gc_task.cancel()
            try:
                await self._gc_task
            except asyncio.CancelledError:
                pass
        async with self._global_lock:
            for mid, entry in list(self._engines.items()):
                entry.queue.cancel_waiting()
                try:
                    entry.engine.__exit__(None, None, None)
                except Exception:
                    pass
            self._engines.clear()


registry = ModelRegistry()
