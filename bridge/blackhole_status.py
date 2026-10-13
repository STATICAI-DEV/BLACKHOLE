"""Read-only Hugging Face to GitHub BLACKHOLE status normalizer."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from blackhole.atomic import atomic_write_json, atomic_write_text
from blackhole.redaction import redact_text

SPACE_ID = "BluStatic/BLACKHOLE"
DATA_BUCKET = "BluStatic/BLACKHOLE/DATA"
LAB_BUCKET = "BluStatic/BLACKHOLE/LAB"


def _configured_volume_records(info: object, runtime: object) -> list[dict[str, Any]]:
    info_runtime = getattr(info, "runtime", None)
    configured = getattr(info_runtime, "volumes", None)
    if configured is None:
        configured = getattr(runtime, "volumes", None)
    return [
        {
            "type": getattr(volume, "type", None),
            "source": getattr(volume, "source", None),
            "path": getattr(volume, "path", None),
            "mount_path": getattr(volume, "mount_path", None),
            "read_only": getattr(volume, "read_only", None),
        }
        for volume in (configured or [])
    ]


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class StatusClient(Protocol):
    def get_space(self) -> dict[str, Any] | None: ...

    def get_jobs(self) -> list[dict[str, Any]]: ...

    def get_dataset_catalog(self) -> dict[str, Any]: ...

    def get_latest_run(self) -> dict[str, Any] | None: ...


class HuggingFaceReadClient:
    """Narrow adapter around authenticated, read-only Hub APIs."""

    def __init__(self, token: str) -> None:
        if not token:
            raise ValueError("BLOCKED_SECRET: HF_READ_TOKEN is required")
        from huggingface_hub import HfApi, HfFileSystem

        self.api = HfApi(token=token)
        self.fs = HfFileSystem(token=token)

    def get_space(self) -> dict[str, Any] | None:
        from huggingface_hub.errors import RepositoryNotFoundError

        try:
            info = self.api.space_info(SPACE_ID)
            runtime = self.api.get_space_runtime(SPACE_ID)
        except RepositoryNotFoundError:
            return None
        volumes = _configured_volume_records(info, runtime)
        return {
            "repo_id": SPACE_ID,
            "private": bool(getattr(info, "private", False)),
            "stage": str(getattr(runtime, "stage", "UNKNOWN")),
            "hardware": getattr(runtime, "hardware", None),
            "requested_hardware": getattr(runtime, "requested_hardware", None),
            "volumes": volumes,
            "sha": getattr(info, "sha", None),
        }

    def get_jobs(self) -> list[dict[str, Any]]:
        list_jobs = getattr(self.api, "list_jobs", None)
        if list_jobs is None:
            return []
        jobs: list[dict[str, Any]] = []
        for job in list_jobs():
            rendered = str(job)
            if "BLACKHOLE" not in rendered.upper():
                continue
            jobs.append(
                {
                    "id": getattr(job, "id", None),
                    "status": str(getattr(job, "status", "UNKNOWN")),
                    "created_at": str(getattr(job, "created_at", "")) or None,
                }
            )
        return jobs

    def _read_json(self, uri: str) -> dict[str, Any]:
        with self.fs.open(uri, "r") as handle:
            return json.load(handle)

    def get_dataset_catalog(self) -> dict[str, Any]:
        latest_uri = f"hf://buckets/{DATA_BUCKET}/manifests/dataset_catalog_latest.json"
        latest = self._read_json(latest_uri)
        pointer = latest.get("catalog") or latest.get("latest") or latest.get("path")
        catalog = latest
        catalog_path = "manifests/dataset_catalog_latest.json"
        if isinstance(pointer, str):
            catalog_path = pointer.removeprefix(f"hf://buckets/{DATA_BUCKET}/")
            catalog = self._read_json(f"hf://buckets/{DATA_BUCKET}/{catalog_path}")
        result = {
            "bucket": DATA_BUCKET,
            "catalog": catalog_path,
            "primary_language": catalog.get("primary_language") or "en",
            "english_train_pairs": catalog.get("english_train_pairs"),
            "english_validation_pairs": catalog.get("english_validation_pairs"),
            "audit_completed": bool(catalog.get("audit_completed", False)),
        }
        return result

    def get_latest_run(self) -> dict[str, Any] | None:
        matches = sorted(
            self.fs.glob(f"hf://buckets/{LAB_BUCKET}/runs/*/RUN_MANIFEST.json")
        )
        if not matches:
            return None
        uri = matches[-1]
        if not uri.startswith("hf://"):
            uri = "hf://" + uri
        return self._read_json(uri)


def collect_status(
    client: StatusClient,
    *,
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    observed = observed_at or datetime.now(UTC)
    errors: list[dict[str, str]] = []

    failed_sources: set[str] = set()

    def read(source: str, call, default):
        try:
            value = call()
            if value is not None and not isinstance(value, (dict, list)):
                raise TypeError("source returned an unexpected value type")
            return value
        except Exception as exc:  # source boundaries must become explicit status
            failed_sources.add(source)
            errors.append({"source": source, "code": "BLOCKED", "detail": redact_text(exc, limit=240)})
            return default

    space = read("space", client.get_space, None)
    jobs = read("jobs", client.get_jobs, [])
    dataset = read("dataset_catalog", client.get_dataset_catalog, None)
    latest_run = read("latest_run", client.get_latest_run, None)

    if "space" in failed_sources:
        space_status = {
            "repo_id": SPACE_ID, "state": "UNKNOWN", "private": None,
            "hardware": None, "revision": None, "volumes": [],
        }
    elif space is None:
        space_status = {
            "repo_id": SPACE_ID,
            "state": "NOT_DEPLOYED",
            "private": None,
            "hardware": None,
            "revision": None,
            "volumes": [],
        }
    else:
        space_status = {
            "repo_id": SPACE_ID,
            "state": space.get("stage") or "UNKNOWN",
            "private": space.get("private"),
            "hardware": space.get("hardware"),
            "revision": space.get("sha"),
            "volumes": space.get("volumes") or [],
        }

    training = {
        "state": "UNKNOWN" if "latest_run" in failed_sources else "NOT_STARTED",
        "run_id": None,
        "latest_step": None,
        "max_steps": None,
        "train_loss": None,
        "validation_loss": None,
        "checkpoint": None,
        "checkpoint_sha256": None,
        "last_metric_at": None,
    }
    model = {"adapter": None, "architecture": None, "version": None}
    if latest_run:
        metrics = latest_run.get("actual_metrics") or {}
        training.update(
            {
                "state": str(latest_run.get("result") or "UNKNOWN").upper(),
                "run_id": latest_run.get("run_id"),
                "latest_step": metrics.get("last_step"),
                "max_steps": metrics.get("max_steps"),
                "train_loss": metrics.get("train_loss"),
                "validation_loss": metrics.get("validation_loss") or metrics.get("val_loss"),
                "checkpoint": latest_run.get("checkpoint"),
                "checkpoint_sha256": latest_run.get("checkpoint_sha256"),
                "last_metric_at": metrics.get("last_metric_at"),
            }
        )
        model.update(
            {
                "adapter": latest_run.get("model_adapter"),
                "architecture": latest_run.get("model_architecture"),
                "version": latest_run.get("model_version"),
            }
        )

    dataset_status = (
        {"state": "UNKNOWN", "bucket": DATA_BUCKET, "catalog": None}
        if dataset is None
        else {"state": "AVAILABLE", **dataset}
    )
    return {
        "schema_version": "blackhole.status.v1",
        "observed_at": _utc_text(observed),
        "snapshot_kind": "cached GitHub snapshot",
        "source_state": "BLOCKED" if errors else "CURRENT",
        "sources": {
            "space": SPACE_ID,
            "dataset_bucket": DATA_BUCKET,
            "lab_bucket": LAB_BUCKET,
        },
        "space": space_status,
        "latest_job": jobs[0] if jobs else None,
        "model": model,
        "training": training,
        "datasets": dataset_status,
        "blockers": errors,
        "next_action": "Restore source access and rerun manually" if errors else "No automatic action scheduled",
    }


def blocked_secret_status(observed_at: datetime | None = None) -> dict[str, Any]:
    observed = observed_at or datetime.now(UTC)
    return {
        "schema_version": "blackhole.status.v1",
        "observed_at": _utc_text(observed),
        "snapshot_kind": "cached GitHub snapshot",
        "source_state": "BLOCKED",
        "sources": {"space": SPACE_ID, "dataset_bucket": DATA_BUCKET, "lab_bucket": LAB_BUCKET},
        "space": {"repo_id": SPACE_ID, "state": "UNKNOWN", "private": None, "hardware": None, "revision": None, "volumes": []},
        "latest_job": None,
        "model": {"adapter": None, "architecture": None, "version": None},
        "training": {"state": "UNKNOWN", "run_id": None, "latest_step": None, "max_steps": None, "train_loss": None, "validation_loss": None, "checkpoint": None, "checkpoint_sha256": None, "last_metric_at": None},
        "datasets": {"state": "UNKNOWN", "bucket": DATA_BUCKET, "catalog": None},
        "blockers": [{"source": "authentication", "code": "BLOCKED_SECRET", "detail": "HF_READ_TOKEN is not configured"}],
        "next_action": "Configure a private read token as HF_READ_TOKEN and rerun manually",
    }


def write_outputs(status: dict[str, Any], output: Path | str, reports_dir: Path | str) -> Path:
    output_path = Path(output)
    report_root = Path(reports_dir)
    observed = datetime.fromisoformat(status["observed_at"].replace("Z", "+00:00"))
    report_path = report_root / observed.astimezone(UTC).strftime("%Y-%m-%d_%H%M.md")
    atomic_write_json(output_path, status)
    training = status["training"]
    report = (
        "# BLACKHOLE Status Snapshot\n\n"
        f"**Observed:** {status['observed_at']}  \n"
        f"**Source:** {status['snapshot_kind']}  \n"
        f"**Source state:** `{status['source_state']}`\n\n"
        f"- Space: `{status['space']['state']}`\n"
        f"- Latest job: `{status['latest_job'] or 'UNKNOWN'}`\n"
        f"- Training: `{training['state']}`\n"
        f"- Latest step: `{training['latest_step']}`\n"
        f"- Train / validation loss: `{training['train_loss']}` / `{training['validation_loss']}`\n"
        f"- Checkpoint SHA-256: `{training['checkpoint_sha256']}`\n"
        f"- Dataset state: `{status['datasets']['state']}`\n"
        f"- Next action: {status['next_action']}\n\n"
        "This is a cached snapshot, not real-time telemetry. Null/UNKNOWN values are not evidence of zero activity.\n"
    )
    atomic_write_text(report_path, report)
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="bridge/status.json")
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args(argv)
    token = os.environ.get("HF_READ_TOKEN") or os.environ.get("HF_TOKEN") or ""
    if not token:
        status = blocked_secret_status()
        write_outputs(status, args.output, args.reports_dir)
        print("BLOCKED_SECRET: HF_READ_TOKEN is not configured")
        return 2
    status = collect_status(HuggingFaceReadClient(token))
    report = write_outputs(status, args.output, args.reports_dir)
    print(f"Wrote {args.output} and {report}")
    return 0 if status["source_state"] == "CURRENT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
