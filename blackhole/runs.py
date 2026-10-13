"""Durable run configuration and truthful finalization records."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from .atomic import atomic_write_json, atomic_write_text
from .config import require_storage_root

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class RunStore:
    def __init__(self, lab_root: Path | str) -> None:
        self.lab_root = Path(lab_root)
        self.root = Path(lab_root) / "runs"

    def run_root(self, run_id: str) -> Path:
        if not _RUN_ID.fullmatch(run_id):
            raise ValueError("run_id must contain only safe identifier characters")
        return self.root / run_id

    def create(
        self,
        run_id: str,
        config: Mapping[str, Any],
        dataset_manifest: Mapping[str, Any],
    ) -> dict[str, Any]:
        require_storage_root(self.lab_root, "lab")
        root = self.run_root(run_id)
        try:
            root.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise FileExistsError(f"run already exists: {run_id}") from exc
        created_at = _utc_now()
        run_config = {
            "schema_version": "blackhole.run-config.v1",
            "run_id": run_id,
            "created_at": created_at,
            **dict(config),
        }
        data_manifest = {
            "schema_version": "blackhole.data-manifest.v1",
            "run_id": run_id,
            **dict(dataset_manifest),
        }
        atomic_write_json(root / "RUN_CONFIG.json", run_config)
        atomic_write_json(root / "DATA_MANIFEST.json", data_manifest)
        return {"run_id": run_id, "status": "created", "created_at": created_at}

    def finalize(
        self,
        run_id: str,
        *,
        result: str,
        actual_metrics: Mapping[str, Any],
        failures: Sequence[str],
    ) -> Path:
        root = self.run_root(run_id)
        if not (root / "RUN_CONFIG.json").is_file():
            raise FileNotFoundError(f"run does not exist: {run_id}")
        manifest = {
            "schema_version": "blackhole.run-manifest.v1",
            "run_id": run_id,
            "finalized_at": _utc_now(),
            "result": result,
            "actual_metrics": dict(actual_metrics),
            "failures": list(failures),
            "artifacts": sorted(path.name for path in root.iterdir() if path.is_file()),
            "claims": {
                "training_completed": result == "completed",
                "conversational_quality_proven": False,
            },
        }
        atomic_write_json(root / "RUN_MANIFEST.json", manifest)
        summary = (
            f"# BLACKHOLE Run {run_id}\n\n"
            f"**Result:** `{result}`  \n"
            f"**Finalized:** {manifest['finalized_at']}\n\n"
            "## Actual metrics\n\n"
            + ("\n".join(f"- `{key}`: {value}" for key, value in actual_metrics.items()) or "- None recorded")
            + "\n\n## Failures\n\n"
            + ("\n".join(f"- {failure}" for failure in failures) or "- None recorded")
            + "\n\n## What was not proven\n\n"
            "- Conversational quality was not established by run finalization alone.\n"
            "- No metric absent from the persisted run artifacts is implied.\n"
        )
        summary_path = root / "RUN_SUMMARY.md"
        atomic_write_text(summary_path, summary)
        return summary_path
