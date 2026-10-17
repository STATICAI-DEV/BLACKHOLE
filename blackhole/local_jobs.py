"""Durable, fail-closed controller queue for outbound local GPU workers."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import secrets
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from .atomic import atomic_write_json
from .checkpoints import CheckpointStore
from .config import LabSettings
from .datasets import DatasetRegistry
from .redaction import redact_value

JOB_SCHEMA = "blackhole.local-job.v1"
WORKER_SCHEMA = "blackhole.local-worker.v1"
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_WINDOWS_PATH = re.compile(r"(?i)\b[A-Z]:\\[^\s;,]+")
_UNIX_LOCAL_PATH = re.compile(r"(?<!:)\/(?:home|Users|tmp|private|var)\/[^\s;,]+")
_ALLOWED_CONFIG = {
    "execution_target",
    "run_id",
    "model_adapter",
    "seed",
    "train_steps",
    "batch_size",
    "gradient_accumulation_steps",
    "context_length",
    "learning_rate",
    "checkpoint_interval",
    "evaluation_interval",
    "job_type",
    "resume_checkpoint_uri",
    "resume_step",
    "dataset_position",
    "dataset_epoch",
    "shuffle_seed",
    "training_recipe_version",
    "validation_max_rows",
    "weight_decay",
    "warmup_steps",
    "min_lr_ratio",
    "reset_optimizer_on_resume",
    "dpo_beta",
}
_COHORT_CONFIG = {
    "execution_target",
    "job_type",
    "cohort_id",
    "members",
    "train_steps",
    "batch_size",
    "gradient_accumulation_steps",
    "context_length",
    "learning_rate",
    "checkpoint_interval",
    "evaluation_interval",
    "shuffle_seed",
    "dataset_position",
    "dataset_epoch",
    "training_recipe_version",
    "validation_max_rows",
    "weight_decay",
    "warmup_steps",
    "min_lr_ratio",
    "reset_optimizer_on_resume",
    "dpo_beta",
}
_EVALUATION_CONFIG = {
    "execution_target",
    "job_type",
    "evaluation_id",
    "model_adapter",
    "checkpoint_run_id",
    "prompt_set_version",
    "seed",
    "temperature",
    "top_k",
    "max_new_tokens",
}
_TRAINING_RECIPES = {
    "causal-pretrain-v1",
    "reasoning-mid-v1",
    "software-engineering-sft-v1",
    "instruction-sft-v1",
    "business-sft-v1",
    "relationship-sft-v1",
    "browser-agent-sft-v1",
    "executive-persona-sft-v1",
    "judgment-sft-v1",
    "preference-dpo-v1",
}
_TERMINAL = {"BLOCKED", "FAILED", "COMPLETED", "CANCELLED"}
_TRANSITIONS = {
    "CLAIMED": {"RUNNING", "BLOCKED", "FAILED", "COMPLETED", "CANCELLED"},
    "RUNNING": {"RUNNING", "BLOCKED", "FAILED", "COMPLETED", "CANCELLED"},
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sanitize_remote_value(value: Any, key: str = "") -> Any:
    """Redact credentials and local filesystem paths before remote persistence."""
    redacted = redact_value(value, key)
    if isinstance(redacted, Mapping):
        return {str(item_key): sanitize_remote_value(item_value, str(item_key)) for item_key, item_value in redacted.items()}
    if isinstance(redacted, list):
        return [sanitize_remote_value(item) for item in redacted]
    if isinstance(redacted, str):
        if ("path" in key.casefold() or "directory" in key.casefold()) and not redacted.startswith("hf://"):
            return "[LOCAL_PATH]"
        return _UNIX_LOCAL_PATH.sub("[LOCAL_PATH]", _WINDOWS_PATH.sub("[LOCAL_PATH]", redacted))
    if redacted is None or isinstance(redacted, (bool, int, float)):
        return redacted
    return str(redacted)


class LocalJobQueue:
    """Single-controller queue persisted on the Space's durable LAB mount."""

    def __init__(self, settings: LabSettings) -> None:
        self.settings = settings
        self.root = settings.lab_root / "local-worker"
        self.jobs_root = self.root / "jobs"
        self.workers_root = self.root / "workers"
        self.datasets = DatasetRegistry(settings.data_root)
        self._lock = threading.RLock()

    def enqueue(
        self,
        config: Mapping[str, Any],
        dataset_manifest: Mapping[str, Any],
        *,
        owner_authorized: bool,
    ) -> dict[str, Any]:
        if not owner_authorized:
            raise PermissionError("TRAINING_NOT_AUTHORIZED: explicit owner authorization is required")
        self.settings.ensure_storage()
        record = dict(config)
        record.setdefault("job_type", "TRAIN")
        if record.get("job_type") != "TRAIN":
            raise ValueError("training enqueue requires job_type TRAIN")
        if record.get("execution_target") != "LOCAL":
            raise ValueError("execution_target must be LOCAL")
        unexpected = sorted(set(record) - _ALLOWED_CONFIG)
        if unexpected:
            raise ValueError(f"unsupported job configuration fields: {', '.join(unexpected)}")
        run_id = record.get("run_id")
        adapter = record.get("model_adapter")
        if not isinstance(run_id, str) or not _ID.fullmatch(run_id):
            raise ValueError("run_id must contain only safe identifier characters")
        if not isinstance(adapter, str) or not _ID.fullmatch(adapter):
            raise ValueError("model_adapter must contain only safe identifier characters")
        steps = record.get("train_steps")
        if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= 10_000_000:
            raise ValueError("train_steps must be an integer between 1 and 10000000")
        batch_size = record.get("batch_size", 1)
        accumulation = record.get("gradient_accumulation_steps", 1)
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or not 1 <= batch_size <= 65_536:
            raise ValueError("batch_size must be an integer between 1 and 65536")
        if not isinstance(accumulation, int) or isinstance(accumulation, bool) or not 1 <= accumulation <= 65_536:
            raise ValueError("gradient_accumulation_steps must be an integer between 1 and 65536")
        dataset = self.datasets.verify_persisted(dataset_manifest)
        rows = int(dataset.get("rows") or 0)
        dataset_position = record.get("dataset_position", 0)
        dataset_epoch = record.get("dataset_epoch", 0)
        shuffle_seed = record.get("shuffle_seed", record.get("seed", 1234))
        validation_max_rows = record.get("validation_max_rows", 256)
        recipe = record.get("training_recipe_version", "causal-pretrain-v1")
        if not isinstance(dataset_position, int) or isinstance(dataset_position, bool) or dataset_position < 0:
            raise ValueError("dataset_position must be a non-negative integer")
        if not isinstance(dataset_epoch, int) or isinstance(dataset_epoch, bool) or dataset_epoch < 0:
            raise ValueError("dataset_epoch must be a non-negative integer")
        if not isinstance(shuffle_seed, int) or isinstance(shuffle_seed, bool) or shuffle_seed < 0:
            raise ValueError("shuffle_seed must be a non-negative integer")
        if (
            not isinstance(validation_max_rows, int)
            or isinstance(validation_max_rows, bool)
            or not 1 <= validation_max_rows <= 100_000
        ):
            raise ValueError("validation_max_rows must be an integer between 1 and 100000")
        if recipe not in _TRAINING_RECIPES:
            raise ValueError("unsupported training_recipe_version")
        required = steps * batch_size * accumulation
        if rows and dataset_position + required > rows:
            raise ValueError(
                f"training requires {required} examples from position {dataset_position}, "
                f"but approved dataset has only {rows} rows"
            )
        record["dataset_position"] = dataset_position
        record["dataset_epoch"] = dataset_epoch
        record["shuffle_seed"] = shuffle_seed
        record["gradient_accumulation_steps"] = accumulation
        weight_decay = record.get("weight_decay", 0.1)
        warmup_steps = record.get("warmup_steps", 100)
        min_lr_ratio = record.get("min_lr_ratio", 0.1)
        reset_optimizer = record.get("reset_optimizer_on_resume", False)
        dpo_beta = record.get("dpo_beta", 0.1)
        if (
            not isinstance(weight_decay, (int, float))
            or isinstance(weight_decay, bool)
            or not 0 <= float(weight_decay) <= 1
        ):
            raise ValueError("weight_decay must be between 0 and 1")
        if not isinstance(warmup_steps, int) or isinstance(warmup_steps, bool) or warmup_steps < 0:
            raise ValueError("warmup_steps must be a non-negative integer")
        if (
            not isinstance(min_lr_ratio, (int, float))
            or isinstance(min_lr_ratio, bool)
            or not 0 <= float(min_lr_ratio) <= 1
        ):
            raise ValueError("min_lr_ratio must be between 0 and 1")
        if not isinstance(reset_optimizer, bool):
            raise ValueError("reset_optimizer_on_resume must be boolean")
        if (
            not isinstance(dpo_beta, (int, float))
            or isinstance(dpo_beta, bool)
            or not math.isfinite(float(dpo_beta))
            or not 0 < float(dpo_beta) <= 2
        ):
            raise ValueError("dpo_beta must be a positive finite number no greater than 2")
        record["validation_max_rows"] = validation_max_rows
        record["training_recipe_version"] = recipe
        record["weight_decay"] = float(weight_decay)
        record["warmup_steps"] = warmup_steps
        record["min_lr_ratio"] = float(min_lr_ratio)
        record["reset_optimizer_on_resume"] = reset_optimizer
        if recipe == "preference-dpo-v1" or "dpo_beta" in record:
            record["dpo_beta"] = float(dpo_beta)

        job_id = "local-" + uuid.uuid4().hex
        now = _utc_now()
        config_sha256 = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        job = {
            "schema_version": JOB_SCHEMA,
            "job_id": job_id,
            "execution_target": "LOCAL",
            "state": "QUEUED",
            "created_at": now,
            "updated_at": now,
            "config": record,
            "dataset_manifest": dataset,
            "authorization": {
                "owner_authorized": True,
                "authorized_at": now,
                "scope": "LOCAL_TRAIN",
                "config_sha256": config_sha256,
                "command_block_approved": True,
            },
            "claim": None,
            "result": None,
            "history": [{"at": now, "state": "QUEUED"}],
        }
        job_root = self.jobs_root / job_id
        with self._lock:
            job_root.mkdir(parents=True, exist_ok=False)
            atomic_write_json(job_root / "JOB.json", job)
        return self._public_job(job)

    def enqueue_cohort(
        self,
        config: Mapping[str, Any],
        dataset_manifest: Mapping[str, Any],
        *,
        owner_authorized: bool,
    ) -> dict[str, Any]:
        """Queue synchronized multi-model LOCAL training in one worker/CUDA process."""
        if not owner_authorized:
            raise PermissionError("TRAINING_NOT_AUTHORIZED: explicit owner authorization is required")
        self.settings.ensure_storage()
        record = dict(config)
        record.setdefault("job_type", "COHORT_TRAIN")
        if record.get("execution_target") != "LOCAL":
            raise ValueError("cohort execution_target must be LOCAL")
        if record.get("job_type") != "COHORT_TRAIN":
            raise ValueError("cohort enqueue requires job_type COHORT_TRAIN")
        unexpected = sorted(set(record) - _COHORT_CONFIG)
        if unexpected:
            raise ValueError(f"unsupported cohort configuration fields: {', '.join(unexpected)}")
        cohort_id = record.get("cohort_id")
        if not isinstance(cohort_id, str) or not _ID.fullmatch(cohort_id):
            raise ValueError("cohort_id must contain only safe identifier characters")
        members = record.get("members")
        if not isinstance(members, list) or not 2 <= len(members) <= 8:
            raise ValueError("cohort members must contain between 2 and 8 models")
        normalized_members: list[dict[str, Any]] = []
        member_ids: set[str] = set()
        run_ids: set[str] = set()
        for raw in members:
            if not isinstance(raw, Mapping):
                raise ValueError("each cohort member must be an object")
            member_id = raw.get("member_id")
            run_id = raw.get("run_id")
            adapter = raw.get("model_adapter")
            seed = raw.get("seed")
            if not isinstance(member_id, str) or not _ID.fullmatch(member_id):
                raise ValueError("member_id must contain only safe identifier characters")
            if not isinstance(run_id, str) or not _ID.fullmatch(run_id):
                raise ValueError("member run_id must contain only safe identifier characters")
            if not isinstance(adapter, str) or not _ID.fullmatch(adapter):
                raise ValueError("member model_adapter must contain only safe identifier characters")
            if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
                raise ValueError("member seed must be a non-negative integer")
            if member_id in member_ids or run_id in run_ids:
                raise ValueError("cohort member_id and run_id values must be unique")
            member_ids.add(member_id)
            run_ids.add(run_id)
            normalized_members.append(
                {
                    "member_id": member_id,
                    "run_id": run_id,
                    "model_adapter": adapter,
                    "seed": seed,
                }
            )
        record["members"] = normalized_members
        if any(member["model_adapter"] in {"STATIC-X", "STATIC-T", "STATIC-K"} for member in normalized_members):
            raise ValueError(
                "foundation-model cohort training is disabled on the single local GPU; "
                "train STATIC-T, STATIC-X and STATIC-K sequentially"
            )
        steps = record.get("train_steps")
        micro_batch = record.get("batch_size", 1)
        accumulation = record.get("gradient_accumulation_steps", 1)
        if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= 10_000_000:
            raise ValueError("train_steps must be an integer between 1 and 10000000")
        if not isinstance(micro_batch, int) or isinstance(micro_batch, bool) or not 1 <= micro_batch <= 65_536:
            raise ValueError("batch_size must be an integer between 1 and 65536")
        if not isinstance(accumulation, int) or isinstance(accumulation, bool) or not 1 <= accumulation <= 65_536:
            raise ValueError("gradient_accumulation_steps must be an integer between 1 and 65536")
        effective_batch = micro_batch * accumulation
        dataset = self.datasets.verify_persisted(dataset_manifest)
        rows = int(dataset.get("rows") or 0)
        start = int(record.get("dataset_position", 0))
        required = steps * effective_batch
        if start < 0:
            raise ValueError("dataset_position must be non-negative")
        if rows and start + required > rows:
            raise ValueError(
                f"cohort requires {required} examples from position {start}, "
                f"but approved dataset has only {rows} rows; reduce steps/effective batch or use an explicit epoch plan"
            )
        shuffle_seed = record.get("shuffle_seed", 1234)
        if not isinstance(shuffle_seed, int) or isinstance(shuffle_seed, bool) or shuffle_seed < 0:
            raise ValueError("shuffle_seed must be a non-negative integer")
        dataset_epoch = record.get("dataset_epoch", 0)
        validation_max_rows = record.get("validation_max_rows", 256)
        recipe = record.get("training_recipe_version", "causal-pretrain-v1")
        if not isinstance(dataset_epoch, int) or isinstance(dataset_epoch, bool) or dataset_epoch < 0:
            raise ValueError("dataset_epoch must be a non-negative integer")
        if (
            not isinstance(validation_max_rows, int)
            or isinstance(validation_max_rows, bool)
            or not 1 <= validation_max_rows <= 100_000
        ):
            raise ValueError("validation_max_rows must be an integer between 1 and 100000")
        if recipe not in _TRAINING_RECIPES:
            raise ValueError("unsupported training_recipe_version")
        record["shuffle_seed"] = shuffle_seed
        record["dataset_position"] = start
        record["dataset_epoch"] = dataset_epoch
        weight_decay = record.get("weight_decay", 0.1)
        warmup_steps = record.get("warmup_steps", 100)
        min_lr_ratio = record.get("min_lr_ratio", 0.1)
        reset_optimizer = record.get("reset_optimizer_on_resume", False)
        dpo_beta = record.get("dpo_beta", 0.1)
        if (
            not isinstance(weight_decay, (int, float))
            or isinstance(weight_decay, bool)
            or not 0 <= float(weight_decay) <= 1
        ):
            raise ValueError("weight_decay must be between 0 and 1")
        if not isinstance(warmup_steps, int) or isinstance(warmup_steps, bool) or warmup_steps < 0:
            raise ValueError("warmup_steps must be a non-negative integer")
        if (
            not isinstance(min_lr_ratio, (int, float))
            or isinstance(min_lr_ratio, bool)
            or not 0 <= float(min_lr_ratio) <= 1
        ):
            raise ValueError("min_lr_ratio must be between 0 and 1")
        if not isinstance(reset_optimizer, bool):
            raise ValueError("reset_optimizer_on_resume must be boolean")
        if (
            not isinstance(dpo_beta, (int, float))
            or isinstance(dpo_beta, bool)
            or not math.isfinite(float(dpo_beta))
            or not 0 < float(dpo_beta) <= 2
        ):
            raise ValueError("dpo_beta must be a positive finite number no greater than 2")
        record["validation_max_rows"] = validation_max_rows
        record["training_recipe_version"] = recipe
        record["weight_decay"] = float(weight_decay)
        record["warmup_steps"] = warmup_steps
        record["min_lr_ratio"] = float(min_lr_ratio)
        record["reset_optimizer_on_resume"] = reset_optimizer
        if recipe == "preference-dpo-v1" or "dpo_beta" in record:
            record["dpo_beta"] = float(dpo_beta)
        record["gradient_accumulation_steps"] = accumulation
        job_id = "local-" + uuid.uuid4().hex
        now = _utc_now()
        config_sha256 = hashlib.sha256(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        job = {
            "schema_version": JOB_SCHEMA,
            "job_id": job_id,
            "execution_target": "LOCAL",
            "state": "QUEUED",
            "created_at": now,
            "updated_at": now,
            "config": record,
            "dataset_manifest": dataset,
            "authorization": {
                "owner_authorized": True,
                "authorized_at": now,
                "scope": "LOCAL_COHORT_TRAIN",
                "config_sha256": config_sha256,
                "command_block_approved": True,
            },
            "claim": None,
            "result": None,
            "history": [{"at": now, "state": "QUEUED"}],
        }
        job_root = self.jobs_root / job_id
        with self._lock:
            job_root.mkdir(parents=True, exist_ok=False)
            atomic_write_json(job_root / "JOB.json", job)
        return self._public_job(job)

    def enqueue_evaluation(
        self,
        config: Mapping[str, Any],
        *,
        owner_authorized: bool,
    ) -> dict[str, Any]:
        """Queue a local CUDA evaluation against an existing durable checkpoint."""
        if not owner_authorized:
            raise PermissionError("EVALUATION_NOT_AUTHORIZED: explicit owner authorization is required")
        self.settings.ensure_storage()
        record = dict(config)
        record.setdefault("job_type", "EVALUATION")
        if record.get("execution_target") != "LOCAL":
            raise ValueError("evaluation execution_target must be LOCAL")
        if record.get("job_type") != "EVALUATION":
            raise ValueError("evaluation job_type must be EVALUATION")
        unexpected = sorted(set(record) - _EVALUATION_CONFIG)
        if unexpected:
            raise ValueError(f"unsupported evaluation configuration fields: {', '.join(unexpected)}")
        for key in ("evaluation_id", "model_adapter", "checkpoint_run_id", "prompt_set_version"):
            value = record.get(key)
            if not isinstance(value, str) or not _ID.fullmatch(value):
                raise ValueError(f"{key} must contain only safe identifier characters")
        if record["prompt_set_version"] not in {"english-heldout-v1", "english-heldout-v2"}:
            raise ValueError("unsupported prompt_set_version")
        seed = record.get("seed", 1234)
        top_k = record.get("top_k", 20)
        max_new_tokens = record.get("max_new_tokens", 128)
        temperature = record.get("temperature", 0.2)
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 256:
            raise ValueError("top_k must be an integer between 1 and 256")
        if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or not 1 <= max_new_tokens <= 4096:
            raise ValueError("max_new_tokens must be an integer between 1 and 4096")
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or not 0 <= float(temperature) <= 5:
            raise ValueError("temperature must be between 0 and 5")
        record.update(
            seed=seed,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            temperature=float(temperature),
        )
        job_id = "local-" + uuid.uuid4().hex
        now = _utc_now()
        config_sha256 = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        job = {
            "schema_version": JOB_SCHEMA,
            "job_id": job_id,
            "execution_target": "LOCAL",
            "state": "QUEUED",
            "created_at": now,
            "updated_at": now,
            "config": record,
            "dataset_manifest": {},
            "authorization": {
                "owner_authorized": True,
                "authorized_at": now,
                "scope": "LOCAL_EVALUATION",
                "config_sha256": config_sha256,
                "command_block_approved": True,
            },
            "claim": None,
            "result": None,
            "history": [{"at": now, "state": "QUEUED"}],
        }
        job_root = self.jobs_root / job_id
        with self._lock:
            job_root.mkdir(parents=True, exist_ok=False)
            atomic_write_json(job_root / "JOB.json", job)
        return self._public_job(job)

    def enqueue_resume(
        self,
        checkpoint_root: Path | str,
        dataset_manifest: Mapping[str, Any],
        *,
        owner_authorized: bool,
    ) -> dict[str, Any]:
        """Queue a real LOCAL continuation from a verified resume checkpoint."""
        if not owner_authorized:
            raise PermissionError("TRAINING_NOT_AUTHORIZED: explicit owner authorization is required")
        self.settings.ensure_storage()
        checkpoint = Path(checkpoint_root)
        inspection = CheckpointStore().inspect(checkpoint)
        if not inspection.resume_ready or inspection.manifest is None:
            raise RuntimeError("RESUME_UNVERIFIED: checkpoint lacks exact resume evidence")
        manifest = inspection.manifest
        source_run_id = str(manifest["run_id"])
        source_config_path = self.settings.lab_root / "runs" / source_run_id / "RUN_CONFIG.json"
        try:
            source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("RESUME_UNVERIFIED: source RUN_CONFIG.json is unavailable") from exc
        original_steps = source_config.get("train_steps")
        completed_step = manifest.get("step")
        source_step_offset = source_config.get("resume_step", 0)
        if (
            not isinstance(original_steps, int)
            or isinstance(original_steps, bool)
            or not isinstance(completed_step, int)
            or isinstance(completed_step, bool)
            or not isinstance(source_step_offset, int)
            or isinstance(source_step_offset, bool)
        ):
            raise RuntimeError("RESUME_UNVERIFIED: training step evidence is invalid")
        completed_local_steps = completed_step - source_step_offset
        remaining_steps = original_steps - completed_local_steps
        if remaining_steps <= 0:
            raise RuntimeError("RESUME_NOT_NEEDED: checkpoint already reached the configured training steps")
        try:
            relative = checkpoint.resolve().relative_to(self.settings.lab_root.resolve()).as_posix()
        except ValueError as exc:
            raise RuntimeError("RESUME_UNVERIFIED: checkpoint is outside the LAB root") from exc
        run_id = f"{source_run_id}-resume-{completed_step}-{uuid.uuid4().hex[:8]}"
        record = {
            "execution_target": "LOCAL",
            "job_type": "TRAIN",
            "run_id": run_id,
            "model_adapter": str(manifest["model_adapter"]),
            "seed": source_config.get("seed"),
            "train_steps": remaining_steps,
            "batch_size": int(source_config.get("batch_size", 1)),
            "context_length": int(source_config.get("context_length", 256)),
            "learning_rate": float(source_config.get("learning_rate", 0.0003)),
            "checkpoint_interval": int(source_config.get("checkpoint_interval", 50)),
            "evaluation_interval": int(source_config.get("evaluation_interval", 10)),
            "resume_checkpoint_uri": f"hf://buckets/STATICAI/BLACKHOLE/LAB/{relative}",
            "resume_step": completed_step,
            "dataset_position": int(manifest.get("dataset_position", completed_local_steps * int(source_config.get("batch_size", 1)))),
            "dataset_epoch": int(source_config.get("dataset_epoch", 0)),
            "shuffle_seed": int(source_config.get("shuffle_seed", source_config.get("seed", 1234))),
            "validation_max_rows": int(source_config.get("validation_max_rows", 256)),
            "training_recipe_version": str(source_config.get("training_recipe_version", "causal-pretrain-v1")),
            "weight_decay": float(source_config.get("weight_decay", 0.1)),
            "warmup_steps": int(source_config.get("warmup_steps", 100)),
            "min_lr_ratio": float(source_config.get("min_lr_ratio", 0.1)),
            "reset_optimizer_on_resume": False,
        }
        return self.enqueue(record, dataset_manifest, owner_authorized=True)

    def request_cancel(self, job_id: str | None = None) -> dict[str, Any]:
        """Request safe cancellation of a queued/claimed/running LOCAL job."""
        self.settings.ensure_storage()
        with self._lock:
            if job_id:
                if not _ID.fullmatch(job_id):
                    raise ValueError("job_id must contain only safe identifier characters")
                candidates = [self.jobs_root / job_id / "JOB.json"]
            else:
                candidates = sorted(
                    self.jobs_root.glob("*/JOB.json"),
                    key=lambda path: (
                        {"RUNNING": 0, "CLAIMED": 1, "QUEUED": 2}.get(
                            str(self._read_job_path(path).get("state") or ""), 9
                        ),
                        -path.stat().st_mtime,
                    ),
                )
            for path in candidates:
                if not path.is_file():
                    continue
                job = self._read_job_path(path)
                state = str(job.get("state") or "")
                if state not in {"QUEUED", "CLAIMED", "RUNNING"}:
                    continue
                now = _utc_now()
                if state == "QUEUED":
                    job["state"] = "CANCELLED"
                    job["updated_at"] = now
                    job["result"] = {"code": "CANCELLED", "reason": "operator_requested_before_claim"}
                    job.setdefault("history", []).append(
                        {"at": now, "state": "CANCELLED", "details": job["result"]}
                    )
                else:
                    job["cancel_requested"] = True
                    job["cancel_requested_at"] = now
                    job["updated_at"] = now
                    job.setdefault("history", []).append(
                        {"at": now, "state": "CANCEL_REQUESTED", "details": {"reason": "operator_requested"}}
                    )
                atomic_write_json(path, job)
                return self._public_job(job)
        raise RuntimeError("NO_ACTIVE_LOCAL_JOB")

    def status(self, job_id: str) -> dict[str, Any]:
        if not isinstance(job_id, str) or not _ID.fullmatch(job_id):
            raise ValueError("job_id must contain only safe identifier characters")
        return self._public_job(self._read_job_path(self.jobs_root / job_id / "JOB.json"))

    def announce(self, report: Mapping[str, Any]) -> dict[str, Any]:
        self.settings.ensure_storage()
        worker_id = report.get("worker_id")
        if not isinstance(worker_id, str) or not _ID.fullmatch(worker_id):
            raise ValueError("worker_id must contain only safe identifier characters")
        cuda_available = report.get("cuda_available") is True
        now = _utc_now()
        persisted = {
            "schema_version": WORKER_SCHEMA,
            "worker_id": worker_id,
            "status": "READY" if cuda_available else "BLOCKED_CUDA",
            "observed_at": now,
            "capabilities": sanitize_remote_value(dict(report)),
        }
        with self._lock:
            atomic_write_json(self.workers_root / f"{worker_id}.json", persisted)
        return persisted

    def claim(self, worker_id: str, job_id: str | None = None) -> dict[str, Any]:
        if not isinstance(worker_id, str) or not _ID.fullmatch(worker_id):
            raise ValueError("worker_id must contain only safe identifier characters")
        worker_path = self.workers_root / f"{worker_id}.json"
        try:
            worker = json.loads(worker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("WORKER_NOT_ANNOUNCED") from exc
        if worker.get("status") != "READY":
            raise RuntimeError("BLOCKED_CUDA: worker capability is not ready")

        with self._lock:
            active = []
            for active_path in self.jobs_root.glob("*/JOB.json"):
                if not active_path.is_file():
                    continue
                active_job = self._read_job_path(active_path)
                if active_job.get("state") in {"CLAIMED", "RUNNING"}:
                    active.append(active_job)
            if active:
                owned = [
                    item for item in active
                    if isinstance(item.get("claim"), Mapping)
                    and item["claim"].get("worker_id") == worker_id
                    and (job_id is None or item.get("job_id") == job_id)
                ]
                if not owned:
                    return {
                        "status": "BUSY",
                        "job": None,
                        "active_job_id": str(active[0].get("job_id") or ""),
                    }
            if job_id is not None:
                if not isinstance(job_id, str) or not _ID.fullmatch(job_id):
                    raise ValueError("job_id must contain only safe identifier characters")
                paths = [self.jobs_root / job_id / "JOB.json"]
            else:
                paths = sorted(self.jobs_root.glob("*/JOB.json"))
            for path in paths:
                if not path.is_file():
                    continue
                job = self._read_job_path(path)
                if job.get("schema_version") != JOB_SCHEMA or job.get("execution_target") != "LOCAL":
                    continue
                if job.get("state") != "QUEUED":
                    continue
                authorization = job.get("authorization") if isinstance(job.get("authorization"), Mapping) else {}
                config = job.get("config") if isinstance(job.get("config"), Mapping) else {}
                config_sha256 = hashlib.sha256(
                    json.dumps(dict(config), sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                if (
                    authorization.get("owner_authorized") is not True
                    or authorization.get("command_block_approved") is not True
                    or authorization.get("config_sha256") != config_sha256
                ):
                    now = _utc_now()
                    job["state"] = "BLOCKED"
                    job["updated_at"] = now
                    job["result"] = {
                        "code": "BLOCKED_AUTHORIZATION",
                        "error": "job authorization is missing or no longer matches its configuration",
                    }
                    job.setdefault("history", []).append(
                        {"at": now, "state": "BLOCKED", "details": job["result"]}
                    )
                    atomic_write_json(path, job)
                    continue
                claim_token = secrets.token_urlsafe(32)
                now = _utc_now()
                job["state"] = "CLAIMED"
                job["updated_at"] = now
                job["claim"] = {
                    "worker_id": worker_id,
                    "claim_token": claim_token,
                    "claimed_at": now,
                }
                job.setdefault("history", []).append({"at": now, "state": "CLAIMED", "worker_id": worker_id})
                atomic_write_json(path, job)
                return {"status": "CLAIMED", "job": self._public_job(job), "claim_token": claim_token}
        return {"status": "IDLE" if job_id is None else "NOT_CLAIMABLE", "job": None}

    def update(
        self,
        job_id: str,
        worker_id: str,
        claim_token: str,
        state: str,
        details: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(job_id, str) or not _ID.fullmatch(job_id):
            raise ValueError("job_id must contain only safe identifier characters")
        if not isinstance(worker_id, str) or not _ID.fullmatch(worker_id):
            raise ValueError("worker_id must contain only safe identifier characters")
        if state not in {"RUNNING", *_TERMINAL}:
            raise ValueError("unsupported local job state")
        path = self.jobs_root / job_id / "JOB.json"
        with self._lock:
            job = self._read_job_path(path)
            current = str(job.get("state") or "")
            if current in _TERMINAL:
                raise RuntimeError("TERMINAL_JOB: terminal local job state is immutable")
            claim = job.get("claim") if isinstance(job.get("claim"), Mapping) else {}
            stored_token = str(claim.get("claim_token") or "")
            if claim.get("worker_id") != worker_id or not stored_token or not hmac.compare_digest(stored_token, claim_token):
                raise PermissionError("CLAIM_OWNERSHIP_MISMATCH")
            if state not in _TRANSITIONS.get(current, set()):
                raise RuntimeError(f"INVALID_JOB_TRANSITION: {current} -> {state}")
            sanitized = sanitize_remote_value(dict(details))
            if state == "COMPLETED":
                references = sanitized.get("artifact_refs") if isinstance(sanitized, Mapping) else None
                if not isinstance(references, list) or not references or not all(
                    isinstance(item, str) and item.startswith("hf://buckets/STATICAI/BLACKHOLE/LAB/")
                    for item in references
                ):
                    raise ValueError("COMPLETED requires LAB bucket artifact_refs")
            now = _utc_now()
            job["state"] = state
            job["updated_at"] = now
            job["result"] = sanitized
            job.setdefault("history", []).append({"at": now, "state": state, "details": sanitized})
            atomic_write_json(path, job)
            return self._public_job(job)

    def list_jobs(self) -> list[dict[str, Any]]:
        if not self.jobs_root.exists():
            return []
        return [self._public_job(self._read_job_path(path)) for path in sorted(self.jobs_root.glob("*/JOB.json"))]

    @staticmethod
    def _read_job_path(path: Path) -> dict[str, Any]:
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError("local job does not exist") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("local job record is unreadable") from exc
        if not isinstance(job, dict):
            raise RuntimeError("local job record is invalid")
        return job

    @staticmethod
    def _public_job(job: Mapping[str, Any]) -> dict[str, Any]:
        rendered = dict(job)
        claim = rendered.get("claim")
        if isinstance(claim, Mapping):
            rendered["claim"] = {key: value for key, value in claim.items() if key != "claim_token"}
        authorization = rendered.pop("authorization", None)
        public = sanitize_remote_value(rendered)
        if isinstance(authorization, Mapping):
            public["authorization"] = {
                key: authorization.get(key)
                for key in (
                    "owner_authorized",
                    "authorized_at",
                    "scope",
                    "config_sha256",
                    "command_block_approved",
                )
            }
        return public
