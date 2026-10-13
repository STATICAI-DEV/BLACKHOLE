"""Read-only system and durable-storage status helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

from .config import BlockedStorageError, LabSettings


@dataclass(frozen=True, slots=True)
class SystemSnapshot:
    observed_at: str
    cpu_percent: float
    ram_used_bytes: int
    ram_total_bytes: int
    storage_ready: bool
    storage_code: str
    storage_detail: str
    checkpoints_found: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_system(settings: LabSettings) -> SystemSnapshot:
    try:
        health = settings.ensure_storage()
    except BlockedStorageError as exc:
        storage_ready = False
        storage_code = "BLOCKED_STORAGE"
        storage_detail = str(exc)
    else:
        storage_ready = health.ready
        storage_code = health.code
        storage_detail = health.detail
    memory = psutil.virtual_memory()
    checkpoint_root = Path(settings.lab_root) / "checkpoints"
    checkpoints = len(list(checkpoint_root.glob("*/CHECKPOINT_MANIFEST.json"))) if checkpoint_root.is_dir() else 0
    return SystemSnapshot(
        observed_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        cpu_percent=float(psutil.cpu_percent(interval=None)),
        ram_used_bytes=int(memory.used),
        ram_total_bytes=int(memory.total),
        storage_ready=storage_ready,
        storage_code=storage_code,
        storage_detail=storage_detail,
        checkpoints_found=checkpoints,
    )

