"""Persisted JSONL metric records; no synthetic defaults."""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping
from .config import require_storage_root

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class MetricsStore:
    def __init__(self, lab_root: Path | str) -> None:
        self.lab_root = Path(lab_root)
        self._lock = threading.RLock()

    def _path(self, run_id: str) -> Path:
        if not _RUN_ID.fullmatch(run_id):
            raise ValueError("run_id must contain only safe identifier characters")
        return self.lab_root / "runs" / run_id / "metrics.jsonl"

    def append(self, run_id: str, metric: Mapping[str, Any]) -> dict[str, Any]:
        require_storage_root(self.lab_root, "lab")
        record = dict(metric)
        step = record.get("step")
        if not isinstance(step, int) or step < 0:
            raise ValueError("metric step must be a non-negative integer")
        with self._lock:
            prior = self.load(run_id)
            if prior and step <= prior[-1]["step"]:
                raise ValueError("metric steps must strictly increase")
            record.setdefault("timestamp_utc", datetime.now(UTC).isoformat().replace("+00:00", "Z"))
            path = self._path(run_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            encoded = (json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
            descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
            try:
                os.write(descriptor, encoded)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return record

    def load(self, run_id: str) -> list[dict[str, Any]]:
        path = self._path(run_id)
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid metrics JSON on line {line_number}") from exc
            if not isinstance(record.get("step"), int):
                raise ValueError(f"invalid metric step on line {line_number}")
            records.append(record)
        return records

    def series(self, run_id: str, field: str) -> dict[str, list[Any]]:
        points = [record for record in self.load(run_id) if record.get(field) is not None]
        return {
            "step": [record["step"] for record in points],
            "value": [record[field] for record in points],
        }
