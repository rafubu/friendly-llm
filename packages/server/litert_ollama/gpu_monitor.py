from __future__ import annotations

import asyncio
import logging
import subprocess
from typing import Any

logger = logging.getLogger(__name__)


class GPUMonitor:
    """Monitors GPU state via nvidia-smi. Falls back gracefully if not found."""

    def __init__(self):
        self._available: bool | None = None
        self._cache: dict[str, Any] | None = None
        self._cache_time: float = 0
        self._cache_ttl = 2.0  # Cache GPU info for 2 seconds

    async def _check_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            self._available = proc.returncode == 0
            return self._available
        except FileNotFoundError:
            self._available = False
            return False

    async def get_info(self) -> dict[str, Any] | None:
        """Returns GPU info dict or None if unavailable."""
        import time
        now = time.time()
        if self._cache and (now - self._cache_time) < self._cache_ttl:
            return self._cache

        if not await self._check_available():
            return None

        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            if proc.returncode != 0:
                return None

            lines = stdout.decode().strip().split("\n")
            gpus = []
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 7:
                    gpu = {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_used_mb": int(float(parts[2])),
                        "memory_total_mb": int(float(parts[3])),
                        "memory_free_mb": int(float(parts[3])) - int(float(parts[2])),
                        "gpu_util_percent": float(parts[4]),
                        "memory_util_percent": float(parts[5]),
                        "temperature_celsius": int(float(parts[6])),
                    }
                    gpus.append(gpu)

            result = {
                "available": True,
                "count": len(gpus),
                "gpus": gpus,
                "total_vram_mb": sum(g["memory_total_mb"] for g in gpus),
                "used_vram_mb": sum(g["memory_used_mb"] for g in gpus),
                "free_vram_mb": sum(g["memory_free_mb"] for g in gpus),
            }
            self._cache = result
            self._cache_time = now
            return result

        except Exception as e:
            logger.debug(f"nvidia-smi error: {e}")
            return None


gpu_monitor = GPUMonitor()
