"""Fail-closed single-writer training lifecycle control."""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from .adapters.registry import AdapterRegistry
from .checkpoints import CheckpointStore
from .config import LabSettings
from .datasets import DatasetRegistry
from .provenance import EventStore
from .runs import RunStore


class TrainingController:
    def __init__(
        self,
        settings: LabSettings,
        adapters: AdapterRegistry,
        checkpoints: CheckpointStore | None = None,
    ) -> None:
        self.settings = settings
        self.adapters = adapters
        self.checkpoints = checkpoints or CheckpointStore()
        self.runs = RunStore(settings.lab_root)
        self.events = EventStore(settings.lab_root)
        self.datasets = DatasetRegistry(settings.data_root)
        self.lock_path = settings.lab_root / "runs" / ".training.lock"
        self._active: dict[str, Any] | None = None
        self._lock_nonce: str | None = None

    def start(
        self,
        config: Mapping[str, Any],
        dataset_manifest: Mapping[str, Any],
        *,
        owner_authorized: bool,
    ) -> dict[str, Any]:
        if not owner_authorized:
            raise PermissionError("TRAINING_NOT_AUTHORIZED: explicit owner authorization is required")
        adapter_name = str(config.get("model_adapter") or "")
        if adapter_name not in self.adapters.names():
            raise RuntimeError("NO_MODEL: requested model adapter is not registered")
        dataset_manifest = self.datasets.verify_persisted(dataset_manifest)
        train_steps = config.get("train_steps")
        if not isinstance(train_steps, int) or train_steps <= 0:
            raise ValueError("train_steps must be a positive integer")
        self.settings.ensure_storage()
        run_id = str(config.get("run_id") or "")
        self._acquire_lock(run_id)
        try:
            self.runs.create(run_id, config, dataset_manifest)
            self.events.emit(
                run_id,
                "training_started",
                "success",
                details={"model_adapter": adapter_name, "train_steps": train_steps},
                source_references=[str(dataset_manifest.get("dataset_id") or "")],
            )
        except Exception:
            self._release_lock()
            raise
        self._active = {
            "run_id": run_id,
            "status": "running",
            "model_adapter": adapter_name,
            "started_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }
        return dict(self._active)

    def request_stop(self) -> dict[str, Any]:
        if self._active is None:
            raise RuntimeError("NO_ACTIVE_TRAINING")
        self._assert_lock_owned()
        run_id = self._active["run_id"]
        self.events.emit(run_id, "stop", "success", details={"reason": "operator_requested"})
        self.runs.finalize(run_id, result="stopped", actual_metrics={"last_step": 0}, failures=[])
        stopped = {**self._active, "status": "stopped"}
        self._release_lock()
        self._active = None
        return stopped

    def complete(self, *, actual_metrics: Mapping[str, Any]) -> dict[str, Any]:
        return self._finalize(result="completed", actual_metrics=actual_metrics, failures=[])

    def stop(self, *, actual_metrics: Mapping[str, Any]) -> dict[str, Any]:
        """Finalize an in-process LOCAL training run as safely stopped."""
        return self._finalize(result="stopped", actual_metrics=actual_metrics, failures=[])

    def fail(self, *, actual_metrics: Mapping[str, Any], failures: list[str]) -> dict[str, Any]:
        return self._finalize(result="failed", actual_metrics=actual_metrics, failures=failures)

    def _finalize(
        self,
        *,
        result: str,
        actual_metrics: Mapping[str, Any],
        failures: list[str],
    ) -> dict[str, Any]:
        if self._active is None:
            raise RuntimeError("NO_ACTIVE_TRAINING")
        self._assert_lock_owned()
        run_id = self._active["run_id"]
        self.runs.finalize(run_id, result=result, actual_metrics=actual_metrics, failures=failures)
        self.events.emit(
            run_id,
            "training_finished",
            "success" if result == "completed" else "failed",
            details={"result": result, "actual_metrics": dict(actual_metrics), "failures": failures},
        )
        self._release_lock()
        self._active = None
        return {"run_id": run_id, "status": result}

    def resume(
        self,
        checkpoint_root: Path | str,
        dataset_manifest: Mapping[str, Any],
        *,
        owner_authorized: bool,
    ) -> dict[str, Any]:
        if not owner_authorized:
            raise PermissionError("TRAINING_NOT_AUTHORIZED: explicit owner authorization is required")
        inspection = self.checkpoints.inspect(checkpoint_root)
        if not inspection.resume_ready or inspection.manifest is None:
            raise RuntimeError("RESUME_UNVERIFIED: checkpoint lacks exact resume evidence")
        manifest = inspection.manifest
        adapter_name = str(manifest["model_adapter"])
        config = {
            "run_id": f"{manifest['run_id']}-resume-{manifest['step']}",
            "model_adapter": adapter_name,
            "seed": manifest.get("seed"),
            "train_steps": 1,
            "resumed_from": str(Path(checkpoint_root)),
        }
        state = self.start(config, dataset_manifest, owner_authorized=True)
        self.events.emit(state["run_id"], "resume", "success", details={"checkpoint": str(checkpoint_root)})
        return state

    def status(self) -> dict[str, Any]:
        if self._active is not None:
            return dict(self._active)
        if self.lock_path.is_file():
            try:
                owner = json.loads(self.lock_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                owner = {"run_id": None}
            return {"status": "blocked", "code": "TRAINING_ALREADY_ACTIVE", "owner": owner}
        return {"status": "idle", "run_id": None}

    def _acquire_lock(self, run_id: str) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise RuntimeError("TRAINING_ALREADY_ACTIVE: single-writer lock exists") from exc
        nonce = uuid.uuid4().hex
        payload = json.dumps({"run_id": run_id, "pid": os.getpid(), "nonce": nonce}, sort_keys=True).encode("utf-8")
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._lock_nonce = nonce

    def _release_lock(self) -> None:
        self._assert_lock_owned()
        self.lock_path.unlink()
        self._lock_nonce = None

    def _assert_lock_owned(self) -> None:
        if self._lock_nonce is None:
            raise RuntimeError("TRAINING_LOCK_OWNERSHIP_LOST: no owned lock nonce")
        try:
            owner = json.loads(self.lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("TRAINING_LOCK_OWNERSHIP_LOST: lock is missing or unreadable") from exc
        if owner.get("nonce") != self._lock_nonce:
            raise RuntimeError("TRAINING_LOCK_OWNERSHIP_LOST: lock belongs to another process")
