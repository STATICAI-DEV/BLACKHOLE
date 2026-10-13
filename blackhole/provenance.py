"""Append-by-convention, one-file-per-event provenance records."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from .atomic import atomic_write_json
from .config import require_storage_root
from .redaction import redact_text, redact_value

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
def _validate_run_id(run_id: str) -> None:
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("run_id must contain only safe identifier characters")


@dataclass(frozen=True, slots=True)
class EventRecord:
    path: Path
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AuditResult:
    valid: bool
    event_count: int
    errors: tuple[str, ...]


class EventStore:
    def __init__(self, lab_root: Path | str) -> None:
        self.lab_root = Path(lab_root)
        self.root = Path(lab_root) / "governance" / "provenance"

    def emit(
        self,
        run_id: str,
        event_type: str,
        result: str,
        *,
        details: Mapping[str, Any] | None = None,
        file_hashes: Mapping[str, str] | None = None,
        source_references: Sequence[str] | None = None,
        code_revision: str | None = None,
    ) -> EventRecord:
        _validate_run_id(run_id)
        require_storage_root(self.lab_root, "lab")
        if not event_type or not result:
            raise ValueError("event_type and result are required")
        timestamp = datetime.now(UTC)
        event_id = uuid.uuid4().hex
        timestamp_text = timestamp.isoformat(timespec="microseconds").replace("+00:00", "Z")
        payload: dict[str, Any] = {
            "schema_version": "blackhole.provenance.v1",
            "event_id": event_id,
            "timestamp_utc": timestamp_text,
            "run_id": run_id,
            "event_type": event_type,
            "result": result,
            "details": redact_value(dict(details or {})),
            "file_hashes": redact_value(dict(file_hashes or {})),
            "software_revision": redact_value(code_revision),
            "source_references": redact_value(list(source_references or [])),
        }
        filename = f"{timestamp.strftime('%Y%m%dT%H%M%S.%fZ')}_{event_id}.json"
        path = self.root / run_id / "EVENTS" / filename
        atomic_write_json(path, payload)
        return EventRecord(path=path, payload=payload)

    def audit(self, run_id: str) -> AuditResult:
        _validate_run_id(run_id)
        errors: list[str] = []
        event_ids: set[str] = set()
        paths = sorted((self.root / run_id / "EVENTS").glob("*.json"))
        for path in paths:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"{path.name}: unreadable event: {exc.__class__.__name__}")
                continue
            for key in ("event_id", "timestamp_utc", "run_id", "event_type", "result"):
                if not payload.get(key):
                    errors.append(f"{path.name}: missing {key}")
            if payload.get("run_id") != run_id:
                errors.append(f"{path.name}: run_id mismatch")
            event_id = payload.get("event_id")
            if event_id in event_ids:
                errors.append(f"{path.name}: duplicate event_id")
            if isinstance(event_id, str):
                event_ids.add(event_id)
            serialized = path.read_text(encoding="utf-8")
            if redact_text(serialized) != serialized:
                errors.append(f"{path.name}: possible credential leak")
        return AuditResult(not errors, len(paths), tuple(errors))
