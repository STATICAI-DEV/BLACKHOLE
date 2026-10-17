"""Authenticated outbound worker for explicitly local CUDA training jobs."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

from .adapters.registry import AdapterRegistry
from .atomic import atomic_write_json, sha256_file
from .checkpoints import CheckpointStore
from .config import LabSettings
from .datasets import DatasetRegistry
from .evaluation import EvaluationStore
from .local_jobs import JOB_SCHEMA, sanitize_remote_value

FOUNDATION_TRAINING_RECIPES = {
    "causal-pretrain-v1",
    "reasoning-mid-v1",
    "software-engineering-sft-v1",
    "instruction-sft-v1",
    "business-sft-v1",
    "finance-sft-v1",
    "relationship-sft-v1",
    "browser-agent-sft-v1",
    "executive-persona-sft-v1",
    "ethics-sft-v1",
    "judgment-sft-v1",
    "preference-dpo-v1",
}
from .metrics import MetricsStore
from .runs import RunStore
from .training import TrainingController

DATA_PREFIX = "hf://buckets/STATICAI/BLACKHOLE/DATA/"
LAB_PREFIX = "hf://buckets/STATICAI/BLACKHOLE/LAB/"


class BlockedJob(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class CancelledJob(RuntimeError):
    """Raised after a safe operator-requested LOCAL cancellation checkpoint."""


def inspect_cuda(torch_module: Any | None = None) -> dict[str, Any]:
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError:
            return {
                "cuda_available": False,
                "status": "BLOCKED_CUDA",
                "error": "PyTorch CUDA runtime is not installed",
            }
    try:
        if not torch_module.cuda.is_available() or torch_module.cuda.device_count() < 1:
            return {
                "cuda_available": False,
                "status": "BLOCKED_CUDA",
                "error": "PyTorch did not expose an NVIDIA CUDA device",
                "torch_version": str(getattr(torch_module, "__version__", "UNKNOWN")),
            }
        free_bytes, total_bytes = torch_module.cuda.mem_get_info(0)
        major, minor = torch_module.cuda.get_device_capability(0)
        return {
            "cuda_available": True,
            "device": "cuda:0",
            "gpu_name": str(torch_module.cuda.get_device_name(0)),
            "vram_free_bytes": int(free_bytes),
            "vram_total_bytes": int(total_bytes),
            "compute_capability": f"{major}.{minor}",
            "torch_version": str(getattr(torch_module, "__version__", "UNKNOWN")),
        }
    except Exception as exc:
        return {
            "cuda_available": False,
            "status": "BLOCKED_CUDA",
            "error": sanitize_remote_value(str(exc)),
            "torch_version": str(getattr(torch_module, "__version__", "UNKNOWN")),
        }


def default_worker_id() -> str:
    identity = f"{platform.node()}|{uuid.getnode()}".encode("utf-8")
    return "worker-" + hashlib.sha256(identity).hexdigest()[:16]


class LocalWorker:
    def __init__(
        self,
        client: Any,
        adapters: AdapterRegistry,
        local_root: Path | str,
        *,
        torch_module: Any | None = None,
        filesystem: Any | None = None,
        worker_id: str | None = None,
    ) -> None:
        self.client = client
        self.adapters = adapters
        self.local_root = Path(local_root)
        self.torch_module = torch_module
        self.filesystem = filesystem
        self.worker_id = worker_id or default_worker_id()

    def capability_report(self) -> dict[str, Any]:
        return {"worker_id": self.worker_id, **inspect_cuda(self.torch_module)}

    def check_only(self) -> dict[str, Any]:
        capability = self.capability_report()
        response = self._call("local_worker_announce", capability)
        if response.get("status") == "ERROR":
            return {"status": "BLOCKED_CONNECTION", "connection": "FAILED", "error": response.get("error")}
        return {
            "status": "PASS" if capability.get("cuda_available") else "BLOCKED_CUDA",
            "connection": "AUTHENTICATED",
            "worker_id": self.worker_id,
            "capabilities": sanitize_remote_value(capability),
            "controller": sanitize_remote_value(response),
            "training_started": False,
        }

    def poll_once(self, *, execute_training: bool, job_id: str | None = None) -> dict[str, Any]:
        capability = self.capability_report()
        announced = self._call("local_worker_announce", capability)
        if announced.get("status") == "ERROR":
            return {"status": "BLOCKED_CONNECTION", "error": announced.get("error")}
        if not capability.get("cuda_available"):
            return {"status": "BLOCKED_CUDA", "capabilities": sanitize_remote_value(capability)}
        if not execute_training:
            return {"status": "TRAINING_DISABLED", "training_started": False}
        claim_payload = {"worker_id": self.worker_id}
        if job_id:
            claim_payload["job_id"] = job_id
        claim = self._call("local_worker_claim", claim_payload)
        if claim.get("status") != "CLAIMED":
            return sanitize_remote_value(claim)
        job = claim.get("job")
        claim_token = claim.get("claim_token")
        if not isinstance(job, Mapping) or not isinstance(claim_token, str) or not claim_token:
            return {"status": "BLOCKED_JOB", "error": "controller returned an invalid claim"}
        return self._execute_claim(dict(job), claim_token)

    def _execute_claim(self, job: dict[str, Any], claim_token: str) -> dict[str, Any]:
        authorization = job.get("authorization")
        if not isinstance(authorization, Mapping) or authorization.get("owner_authorized") is not True:
            return self._update(
                job,
                claim_token,
                "BLOCKED",
                {"code": "BLOCKED_AUTHORIZATION", "error": "job lacks durable owner authorization"},
            )
        config = job.get("config")
        if not isinstance(config, Mapping):
            return self._update(
                job,
                claim_token,
                "BLOCKED",
                {"code": "BLOCKED_JOB", "error": "job configuration is missing"},
            )
        job_type = str(config.get("job_type") or "TRAIN")
        if job_type == "EVALUATION":
            return self._execute_evaluation_claim(job, claim_token, dict(config))
        if job_type == "COHORT_TRAIN":
            return self._execute_cohort_claim(job, claim_token, dict(config))
        if job_type != "TRAIN":
            return self._update(
                job,
                claim_token,
                "BLOCKED",
                {"code": "BLOCKED_JOB", "error": "unsupported local job type"},
            )

        try:
            adapter, config, dataset = self._validate_claim(job)
        except BlockedJob as exc:
            return self._update(job, claim_token, "BLOCKED", {"code": exc.code, "error": str(exc)})

        try:
            dataset_path = self._download_artifact(
                str(dataset["artifact_uri"]),
                str(dataset["content_sha256"]),
                f"{dataset['record_id']}-train.jsonl",
            )
            validation_path = None
            if dataset.get("validation_artifact_uri"):
                validation_path = self._download_artifact(
                    str(dataset["validation_artifact_uri"]),
                    str(dataset["validation_content_sha256"]),
                    f"{dataset['record_id']}-validation.jsonl",
                )
            settings = self._settings()
            settings.ensure_storage()
            persisted_dataset = self._persist_dataset(settings, dataset)
            resume_uri = config.get("resume_checkpoint_uri")
            if resume_uri:
                resume_root = self._download_checkpoint(str(resume_uri), str(config["model_adapter"]))
                if config.get("reset_optimizer_on_resume") is True:
                    try:
                        adapter.load(resume_root, weights_only=True)
                    except TypeError as exc:
                        raise BlockedJob("BLOCKED_CHECKPOINT", "adapter does not support weights-only stage transition") from exc
                else:
                    adapter.load(resume_root)

            def should_cancel() -> bool:
                try:
                    status = self._call(
                        "local_worker_status",
                        {"job_id": str(job.get("job_id") or "")},
                    )
                    return status.get("cancel_requested") is True or status.get("state") == "CANCELLED"
                except Exception:
                    return False

            try:
                self._update(
                    job,
                    claim_token,
                    "RUNNING",
                    {"device": "cuda:0", "job_type": "TRAIN", "step": 0},
                )
            except Exception:
                # The claimed job is already authorized. A telemetry outage must not kill valid local CUDA work.
                pass

            def sync_metric(metric: Mapping[str, Any]) -> Mapping[str, Any]:
                try:
                    return self._update(
                        job,
                        claim_token,
                        "RUNNING",
                        {"device": "cuda:0", "job_type": "TRAIN", "metrics": metric},
                    )
                except Exception:
                    return {"status": "DEFERRED"}

            result = self._run_training(
                settings,
                adapter,
                config,
                persisted_dataset,
                dataset_path,
                validation_path,
                on_metric=sync_metric,
                should_cancel=should_cancel,
            )
            refs, hashes = self._upload_run(settings, str(config["run_id"]))
            return self._update(
                job,
                claim_token,
                "COMPLETED",
                {
                    "code": "COMPLETED",
                    "job_type": "TRAIN",
                    "metrics": result,
                    "artifact_refs": refs,
                    "sha256": hashes,
                },
            )
        except CancelledJob as exc:
            try:
                refs, hashes = self._upload_run(settings, str(config["run_id"]))
            except Exception:
                refs, hashes = [], {}
            return self._update(
                job,
                claim_token,
                "CANCELLED",
                {
                    "code": "CANCELLED",
                    "reason": str(exc),
                    "artifact_refs": refs,
                    "sha256": hashes,
                },
            )
        except BlockedJob as exc:
            return self._update(job, claim_token, "BLOCKED", {"code": exc.code, "error": str(exc)})
        except Exception as exc:
            try:
                return self._update(
                    job,
                    claim_token,
                    "FAILED",
                    {"code": "LOCAL_EXECUTION_FAILED", "error": sanitize_remote_value(str(exc))},
                )
            except Exception:
                return {
                    "status": "FAILED_LOCAL_WITH_CONTROLLER_UNREACHABLE",
                    "error": sanitize_remote_value(str(exc)),
                }

    def _execute_cohort_claim(
        self,
        job: Mapping[str, Any],
        claim_token: str,
        config: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Train a synchronized cohort in one CUDA process against one shared row order."""
        if job.get("schema_version") != JOB_SCHEMA or job.get("execution_target") != "LOCAL":
            return self._update(
                job, claim_token, "BLOCKED",
                {"code": "BLOCKED_JOB", "error": "cohort job is not explicitly LOCAL"},
            )
        members = config.get("members")
        dataset = job.get("dataset_manifest")
        if not isinstance(members, list) or not isinstance(dataset, Mapping):
            return self._update(
                job, claim_token, "BLOCKED",
                {"code": "BLOCKED_JOB", "error": "cohort members or dataset manifest is missing"},
            )
        capability = inspect_cuda(self.torch_module)
        if not capability.get("cuda_available"):
            return self._update(
                job, claim_token, "BLOCKED",
                {"code": "BLOCKED_CUDA", "error": "CUDA unavailable for cohort training"},
            )
        artifact_uri = dataset.get("artifact_uri")
        digest = dataset.get("content_sha256")
        if (
            not isinstance(artifact_uri, str)
            or not artifact_uri.startswith(DATA_PREFIX)
            or not artifact_uri.endswith(".jsonl")
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            return self._update(
                job, claim_token, "BLOCKED",
                {"code": "BLOCKED_INPUTS", "error": "approved cohort DATA artifact is unavailable"},
            )
        for member in members:
            adapter_name = str(member.get("model_adapter") or "") if isinstance(member, Mapping) else ""
            if adapter_name not in self.adapters.names():
                return self._update(
                    job, claim_token, "BLOCKED",
                    {"code": "BLOCKED_INPUTS", "error": f"cohort adapter is unavailable: {adapter_name}"},
                )

        try:
            dataset_path = self._download_artifact(
                artifact_uri,
                digest,
                f"{dataset['record_id']}-cohort-train.jsonl",
            )
            validation_path = None
            if dataset.get("validation_artifact_uri"):
                validation_path = self._download_artifact(
                    str(dataset["validation_artifact_uri"]),
                    str(dataset["validation_content_sha256"]),
                    f"{dataset['record_id']}-cohort-validation.jsonl",
                )
            settings = self._settings()
            settings.ensure_storage()
            persisted_dataset = self._persist_dataset(settings, dataset)
            try:
                self._update(
                    job,
                    claim_token,
                    "RUNNING",
                    {
                        "device": "cuda:0",
                        "job_type": "COHORT_TRAIN",
                        "cohort_id": config.get("cohort_id"),
                        "members": [item.get("member_id") for item in members],
                        "step": 0,
                    },
                )
            except Exception:
                pass

            def sync_metric(payload: Mapping[str, Any]) -> None:
                try:
                    self._update(
                        job,
                        claim_token,
                        "RUNNING",
                        {
                            "device": "cuda:0",
                            "job_type": "COHORT_TRAIN",
                            "cohort_id": config.get("cohort_id"),
                            "metrics": payload,
                        },
                    )
                except Exception:
                    pass

            result = self._run_cohort_training(
                settings,
                config,
                persisted_dataset,
                dataset_path,
                validation_path,
                job_id=str(job.get("job_id") or ""),
                on_metric=sync_metric,
            )
            refs: list[str] = []
            hashes: dict[str, str] = {}
            for member in members:
                member_refs, member_hashes = self._upload_run(settings, str(member["run_id"]))
                refs.extend(member_refs)
                hashes.update(member_hashes)
            return self._update(
                job,
                claim_token,
                "COMPLETED",
                {
                    "code": "COMPLETED",
                    "job_type": "COHORT_TRAIN",
                    "cohort_id": config.get("cohort_id"),
                    "metrics": result,
                    "artifact_refs": refs,
                    "sha256": hashes,
                },
            )
        except CancelledJob as exc:
            refs: list[str] = []
            hashes: dict[str, str] = {}
            for member in members:
                try:
                    member_refs, member_hashes = self._upload_run(settings, str(member["run_id"]))
                    refs.extend(member_refs)
                    hashes.update(member_hashes)
                except Exception:
                    pass
            return self._update(
                job,
                claim_token,
                "CANCELLED",
                {
                    "code": "CANCELLED",
                    "job_type": "COHORT_TRAIN",
                    "cohort_id": config.get("cohort_id"),
                    "reason": str(exc),
                    "artifact_refs": refs,
                    "sha256": hashes,
                },
            )
        except BlockedJob as exc:
            return self._update(job, claim_token, "BLOCKED", {"code": exc.code, "error": str(exc)})
        except Exception as exc:
            try:
                return self._update(
                    job,
                    claim_token,
                    "FAILED",
                    {
                        "code": "LOCAL_COHORT_FAILED",
                        "cohort_id": config.get("cohort_id"),
                        "error": sanitize_remote_value(str(exc)),
                    },
                )
            except Exception:
                return {
                    "status": "FAILED_LOCAL_WITH_CONTROLLER_UNREACHABLE",
                    "error": sanitize_remote_value(str(exc)),
                }

    def _execute_evaluation_claim(
        self,
        job: Mapping[str, Any],
        claim_token: str,
        config: Mapping[str, Any],
    ) -> dict[str, Any]:
        if job.get("schema_version") != JOB_SCHEMA or job.get("execution_target") != "LOCAL":
            return self._update(
                job,
                claim_token,
                "BLOCKED",
                {"code": "BLOCKED_JOB", "error": "evaluation job is not explicitly LOCAL"},
            )
        adapter_name = str(config.get("model_adapter") or "")
        if adapter_name not in self.adapters.names():
            return self._update(
                job,
                claim_token,
                "BLOCKED",
                {"code": "BLOCKED_INPUTS", "error": "requested model adapter is unavailable"},
            )
        prompt_set_version = str(config.get("prompt_set_version") or "")
        if prompt_set_version not in {"english-heldout-v1", "english-heldout-v2"}:
            return self._update(
                job,
                claim_token,
                "BLOCKED",
                {"code": "BLOCKED_INPUTS", "error": "unsupported prompt set"},
            )
        capability = inspect_cuda(self.torch_module)
        if not capability.get("cuda_available"):
            return self._update(
                job,
                claim_token,
                "BLOCKED",
                {"code": "BLOCKED_CUDA", "error": "CUDA unavailable for local Arena"},
            )
        if self.filesystem is None:
            from huggingface_hub import HfFileSystem
            self.filesystem = HfFileSystem(token=True)

        run_id = str(config.get("checkpoint_run_id") or "")
        remote_root = f"{LAB_PREFIX}runs/{run_id}/checkpoint-final"
        try:
            with self.filesystem.open(f"{remote_root}/CHECKPOINT_MANIFEST.json", "r") as source:
                manifest = json.load(source)
        except Exception as exc:
            return self._update(
                job,
                claim_token,
                "BLOCKED",
                {"code": "BLOCKED_CHECKPOINT", "error": sanitize_remote_value(str(exc))},
            )
        if (
            manifest.get("run_id") != run_id
            or manifest.get("model_adapter") != adapter_name
            or manifest.get("trained") is not True
        ):
            return self._update(
                job,
                claim_token,
                "BLOCKED",
                {"code": "BLOCKED_CHECKPOINT", "error": "checkpoint manifest identity mismatch"},
            )
        files = manifest.get("files") if isinstance(manifest.get("files"), Mapping) else {}
        hashes = manifest.get("sha256") if isinstance(manifest.get("sha256"), Mapping) else {}
        model_name = files.get("model")
        model_hash = hashes.get("model")
        if not isinstance(model_name, str) or not isinstance(model_hash, str) or len(model_hash) != 64:
            return self._update(
                job,
                claim_token,
                "BLOCKED",
                {"code": "BLOCKED_CHECKPOINT", "error": "checkpoint model evidence is missing"},
            )
        checkpoint_model = self._download_artifact(
            f"{remote_root}/{model_name}",
            model_hash,
            f"arena/{run_id}/model.pt",
        )
        checkpoint_root = checkpoint_model.parent
        adapter = self.adapters.get(adapter_name)
        inspection = adapter.inspect()
        if not isinstance(inspection, Mapping) or inspection.get("device") != "cuda:0":
            return self._update(
                job,
                claim_token,
                "BLOCKED",
                {"code": "BLOCKED_CUDA", "error": "adapter is not configured for cuda:0"},
            )

        try:
            self._update(
                job,
                claim_token,
                "RUNNING",
                {"device": "cuda:0", "job_type": "EVALUATION", "checkpoint_run_id": run_id},
            )
        except Exception:
            pass

        try:
            adapter.load(checkpoint_root)
            torch_module = self.torch_module
            if torch_module is None:
                import torch as torch_module
            seed = int(config.get("seed", 1234))
            temperature = float(config.get("temperature", 0.2))
            top_k = int(config.get("top_k", 20))
            max_new_tokens = int(config.get("max_new_tokens", 128))
            if prompt_set_version == "english-heldout-v1":
                prompts = (
                    ("vocab-001", "basic_vocabulary", "Name a common fruit.", ("apple", "banana", "orange", "pear", "grape"), "contains"),
                    ("format-001", "format_compliance", "Return one word.", (), "one_word"),
                )
            else:
                prompts = (
                    ("vocab-001", "vocabulary", "Name a common fruit.", ("apple", "banana", "orange", "pear", "grape"), "contains"),
                    ("vocab-002", "vocabulary", "What animal barks?", ("dog",), "contains"),
                    ("vocab-003", "vocabulary", "What color is grass?", ("green",), "contains"),
                    ("vocab-004", "vocabulary", "What color is snow?", ("white",), "contains"),
                    ("vocab-005", "vocabulary", "What is the opposite of hot?", ("cold",), "contains"),
                    ("vocab-006", "vocabulary", "What is the plural of book?", ("books",), "contains"),
                    ("vocab-007", "vocabulary", "What month comes after January?", ("february",), "contains"),
                    ("vocab-008", "vocabulary", "What day comes after Monday?", ("tuesday",), "contains"),
                    ("vocab-009", "vocabulary", "What do bees make?", ("honey",), "contains"),
                    ("vocab-010", "vocabulary", "What do humans breathe?", ("air", "oxygen"), "contains"),
                    ("qa-001", "question_answering", "What is 2 + 2?", ("4",), "exact"),
                    ("qa-002", "question_answering", "What is 7 - 3?", ("4",), "exact"),
                    ("qa-003", "question_answering", "What is 3 times 3?", ("9",), "exact"),
                    ("qa-004", "question_answering", "How many days are in a week?", ("7", "seven"), "contains"),
                    ("qa-005", "question_answering", "How many hours are in a day?", ("24", "twenty four", "twenty-four"), "contains"),
                    ("qa-006", "question_answering", "Is water wet? Answer yes or no.", ("yes",), "exact"),
                    ("qa-007", "question_answering", "Is fire cold? Answer yes or no.", ("no",), "exact"),
                    ("qa-008", "question_answering", "Which is larger, 10 or 2?", ("10",), "exact"),
                    ("qa-009", "question_answering", "What number comes after 5?", ("6", "six"), "exact"),
                    ("qa-010", "question_answering", "How many legs does a typical dog have?", ("4", "four"), "contains"),
                    ("inst-001", "instruction_following", "Reply only with YES.", ("yes",), "exact"),
                    ("inst-002", "instruction_following", "Reply only with NO.", ("no",), "exact"),
                    ("inst-003", "instruction_following", "Return exactly one word: ready", ("ready",), "exact"),
                    ("inst-004", "instruction_following", "Write the number 7 and nothing else.", ("7",), "exact"),
                    ("inst-005", "instruction_following", "Reply with the word blue.", ("blue",), "exact"),
                    ("inst-006", "instruction_following", "Say hello in one word.", ("hello", "hi"), "exact"),
                    ("inst-007", "instruction_following", "Reply with two words: thank you", ("thank you",), "exact"),
                    ("inst-008", "instruction_following", "Return one word meaning happy.", ("happy", "glad", "joyful"), "one_word_contains"),
                    ("inst-009", "instruction_following", "Answer with a single digit: one plus one.", ("2",), "exact"),
                    ("inst-010", "instruction_following", "Reply only with OK.", ("ok",), "exact"),
                    ("reason-001", "basic_reasoning", "Which is larger: 9 or 4?", ("9",), "exact"),
                    ("reason-002", "basic_reasoning", "Which is smaller: 3 or 8?", ("3",), "exact"),
                    ("reason-003", "basic_reasoning", "Sam has 2 apples and gets 1 more. How many apples?", ("3", "three"), "contains"),
                    ("reason-004", "basic_reasoning", "A box has 5 balls. Remove 2. How many remain?", ("3", "three"), "contains"),
                    ("reason-005", "basic_reasoning", "If today is Monday, what day is tomorrow?", ("tuesday",), "contains"),
                    ("reason-006", "basic_reasoning", "If something is not true, it is what?", ("false",), "contains"),
                    ("reason-007", "basic_reasoning", "Which can fly: a bird or a rock?", ("bird",), "contains"),
                    ("reason-008", "basic_reasoning", "Which is heavier: a car or a feather?", ("car",), "contains"),
                    ("reason-009", "basic_reasoning", "If all cats are animals, is a cat an animal?", ("yes",), "contains"),
                    ("reason-010", "basic_reasoning", "Complete: 1, 2, 3, what comes next?", ("4", "four"), "contains"),
                    ("chat-001", "conversation", "Hello!", ("hello", "hi", "hey"), "contains"),
                    ("chat-002", "conversation", "Good morning.", ("morning", "hello", "hi"), "contains"),
                    ("chat-003", "conversation", "Thank you.", ("welcome", "pleasure"), "contains"),
                    ("chat-004", "conversation", "How are you?", ("well", "good", "fine", "okay", "ok"), "contains"),
                    ("chat-005", "conversation", "Nice to meet you.", ("meet", "nice", "pleasure"), "contains"),
                    ("chat-006", "conversation", "Goodbye.", ("goodbye", "bye"), "contains"),
                    ("chat-007", "conversation", "Can you help me?", ("yes", "help", "sure"), "contains"),
                    ("chat-008", "conversation", "Please answer my question.", ("yes", "sure", "okay", "ok"), "contains"),
                    ("chat-009", "conversation", "I am happy today.", ("glad", "happy", "good"), "contains"),
                    ("chat-010", "conversation", "I made a mistake.", ("okay", "ok", "happens", "help", "fix"), "contains"),
                )
            results: list[dict[str, Any]] = []
            synchronize = getattr(torch_module.cuda, "synchronize", None)
            for prompt_id, category, prompt, accepted, scoring in prompts:
                torch_module.manual_seed(seed)
                manual_seed_all = getattr(torch_module.cuda, "manual_seed_all", None)
                if callable(manual_seed_all):
                    manual_seed_all(seed)
                if callable(synchronize):
                    synchronize()
                started = time.perf_counter()
                output = adapter.generate(
                    prompt,
                    temperature=temperature,
                    top_k=top_k,
                    max_new_tokens=max_new_tokens,
                )
                if callable(synchronize):
                    synchronize()
                normalized = " ".join(output.strip().casefold().split())
                if scoring == "exact":
                    passed = normalized in accepted
                elif scoring == "one_word":
                    passed = bool(normalized) and len(normalized.split()) == 1
                elif scoring == "one_word_contains":
                    passed = len(normalized.split()) == 1 and any(item in normalized for item in accepted)
                else:
                    passed = any(item in normalized for item in accepted)
                results.append(
                    {
                        "prompt_id": prompt_id,
                        "category": category,
                        "prompt": prompt,
                        "output": output,
                        "passed": bool(passed),
                        "duration_seconds": round(time.perf_counter() - started, 6),
                    }
                )

            pass_count = sum(1 for item in results if item["passed"])
            category_scores: dict[str, dict[str, int]] = {}
            for item in results:
                score = category_scores.setdefault(item["category"], {"passed": 0, "total": 0})
                score["total"] += 1
                score["passed"] += int(item["passed"])

            settings = self._settings()
            settings.ensure_storage()
            evaluation_id = str(config["evaluation_id"])
            payload = {
                "prompt_set_version": prompt_set_version,
                "checkpoint": remote_root,
                "eval_hardware": "local-cuda",
                "gpu_name": capability.get("gpu_name"),
                "generation_parameters": {
                    "temperature": temperature,
                    "top_k": top_k,
                    "max_new_tokens": max_new_tokens,
                },
                "model_adapter": adapter_name,
                "train_run_id": run_id,
                "seed": seed,
                "score": {
                    "passed": pass_count,
                    "total": len(results),
                    "accuracy": pass_count / len(results),
                    "categories": category_scores,
                },
                "results": results,
            }
            saved = EvaluationStore(settings.lab_root).save(evaluation_id, payload)
            local_result = settings.lab_root / "evaluations" / evaluation_id / "EVALUATION_RESULTS.json"
            remote_result = f"{LAB_PREFIX}evaluations/{evaluation_id}/EVALUATION_RESULTS.json"
            with local_result.open("rb") as source, self.filesystem.open(remote_result, "wb") as destination:
                while chunk := source.read(1024 * 1024):
                    destination.write(chunk)
            local_hash = sha256_file(local_result)
            with self.filesystem.open(remote_result, "rb") as source:
                remote_hash = hashlib.sha256(source.read()).hexdigest()
            if remote_hash != local_hash:
                raise RuntimeError("evaluation upload SHA-256 mismatch")
            return self._update(
                job,
                claim_token,
                "COMPLETED",
                {
                    "code": "COMPLETED",
                    "job_type": "EVALUATION",
                    "evaluation_id": evaluation_id,
                    "results": saved["results"],
                    "artifact_refs": [remote_result],
                    "sha256": {remote_result: local_hash},
                },
            )
        except Exception as exc:
            try:
                return self._update(
                    job,
                    claim_token,
                    "FAILED",
                    {"code": "LOCAL_EVALUATION_FAILED", "error": sanitize_remote_value(str(exc))},
                )
            except Exception:
                return {
                    "status": "FAILED_LOCAL_WITH_CONTROLLER_UNREACHABLE",
                    "error": sanitize_remote_value(str(exc)),
                }

    def _validate_claim(self, job: Mapping[str, Any]) -> tuple[Any, dict[str, Any], dict[str, Any]]:
        if job.get("schema_version") != JOB_SCHEMA or job.get("execution_target") != "LOCAL":
            raise BlockedJob("BLOCKED_JOB", "job schema or execution target is not supported")
        config = job.get("config")
        dataset = job.get("dataset_manifest")
        if not isinstance(config, Mapping) or not isinstance(dataset, Mapping):
            raise BlockedJob("BLOCKED_JOB", "job configuration or dataset manifest is missing")
        if config.get("execution_target") != "LOCAL":
            raise BlockedJob("BLOCKED_JOB", "job configuration is not explicitly LOCAL")
        adapter_name = str(config.get("model_adapter") or "")
        if adapter_name not in self.adapters.names():
            raise BlockedJob("BLOCKED_INPUTS", "requested in-project model adapter is unavailable")
        artifact_uri = dataset.get("artifact_uri")
        if not isinstance(artifact_uri, str) or not artifact_uri.startswith(DATA_PREFIX) or not artifact_uri.endswith(".jsonl"):
            raise BlockedJob("BLOCKED_INPUTS", "approved DATA bucket JSONL artifact is unavailable")
        digest = dataset.get("content_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise BlockedJob("BLOCKED_INPUTS", "dataset content hash is unavailable")
        validation_uri = dataset.get("validation_artifact_uri")
        validation_digest = dataset.get("validation_content_sha256")
        if validation_uri is not None or validation_digest is not None:
            if (
                not isinstance(validation_uri, str)
                or not validation_uri.startswith(DATA_PREFIX)
                or not validation_uri.endswith(".jsonl")
            ):
                raise BlockedJob("BLOCKED_INPUTS", "approved validation DATA bucket JSONL artifact is unavailable")
            if not isinstance(validation_digest, str) or len(validation_digest) != 64:
                raise BlockedJob("BLOCKED_INPUTS", "validation dataset content hash is unavailable")
        adapter = self.adapters.get(adapter_name)
        configure = getattr(adapter, "configure_for_training", None)
        if callable(configure):
            try:
                configure(config)
            except (TypeError, ValueError) as exc:
                raise BlockedJob(
                    "BLOCKED_INPUTS",
                    f"adapter training configuration is invalid: {sanitize_remote_value(str(exc))}",
                ) from exc
        inspection = adapter.inspect()
        if not isinstance(inspection, Mapping) or inspection.get("device") != "cuda:0":
            raise BlockedJob("BLOCKED_CUDA", "adapter is not explicitly configured for cuda:0")
        return adapter, dict(config), dict(dataset)

    def _download_artifact(self, uri: str, expected_sha256: str, filename: str) -> Path:
        if self.filesystem is None:
            from huggingface_hub import HfFileSystem

            self.filesystem = HfFileSystem(token=True)
        target = self.local_root / "cache" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        with self.filesystem.open(uri, "rb") as source, target.open("wb") as destination:
            while chunk := source.read(1024 * 1024):
                destination.write(chunk)
        if sha256_file(target) != expected_sha256:
            target.unlink(missing_ok=True)
            raise BlockedJob("BLOCKED_INPUTS", "dataset artifact SHA-256 mismatch")
        return target

    def _download_checkpoint(self, uri: str, expected_adapter: str) -> Path:
        if not uri.startswith(LAB_PREFIX) or "/checkpoint-" not in uri:
            raise BlockedJob("BLOCKED_CHECKPOINT", "resume checkpoint URI is outside the BLACKHOLE LAB")
        if self.filesystem is None:
            from huggingface_hub import HfFileSystem
            self.filesystem = HfFileSystem(token=True)
        try:
            with self.filesystem.open(f"{uri}/CHECKPOINT_MANIFEST.json", "r") as source:
                manifest = json.load(source)
        except Exception as exc:
            raise BlockedJob("BLOCKED_CHECKPOINT", "resume checkpoint manifest is unavailable") from exc
        checkpoint_id = str(manifest.get("checkpoint_id") or "")
        if not checkpoint_id or manifest.get("model_adapter") != expected_adapter:
            raise BlockedJob("BLOCKED_CHECKPOINT", "resume checkpoint identity mismatch")
        files = manifest.get("files") if isinstance(manifest.get("files"), Mapping) else {}
        hashes = manifest.get("sha256") if isinstance(manifest.get("sha256"), Mapping) else {}
        target = self.local_root / "cache" / "resume" / checkpoint_id
        target.mkdir(parents=True, exist_ok=True)
        for role, filename in files.items():
            if not isinstance(filename, str):
                raise BlockedJob("BLOCKED_CHECKPOINT", f"invalid checkpoint file for {role}")
            relative = Path(filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise BlockedJob("BLOCKED_CHECKPOINT", f"unsafe checkpoint file for {role}")
            digest = hashes.get(role)
            if not isinstance(digest, str) or len(digest) != 64:
                raise BlockedJob("BLOCKED_CHECKPOINT", f"missing checkpoint hash for {role}")
            downloaded = self._download_artifact(
                f"{uri}/{relative.as_posix()}",
                digest,
                f"resume/{checkpoint_id}/{relative.as_posix()}",
            )
            if downloaded != target / relative:
                raise BlockedJob("BLOCKED_CHECKPOINT", "checkpoint cache path mismatch")
        atomic_write_json(target / "CHECKPOINT_MANIFEST.json", manifest)
        inspection = CheckpointStore().inspect(target, expected_adapter=expected_adapter)
        if not inspection.resume_ready:
            raise BlockedJob("BLOCKED_CHECKPOINT", "downloaded checkpoint failed resume verification")
        return target

    def _cancel_requested(self, job_id: str) -> bool:
        try:
            status = self._call("local_worker_status", {"job_id": job_id})
            return status.get("cancel_requested") is True or status.get("state") == "CANCELLED"
        except Exception:
            return False

    def _settings(self) -> LabSettings:
        return LabSettings.from_env(
            {
                "BLACKHOLE_ENV": "development",
                "BLACKHOLE_DATA_ROOT": str(self.local_root / "data"),
                "BLACKHOLE_LAB_ROOT": str(self.local_root / "lab"),
            }
        )

    @staticmethod
    def _persist_dataset(settings: LabSettings, dataset: Mapping[str, Any]) -> dict[str, Any]:
        record_id = str(dataset.get("record_id") or "")
        if len(record_id) != 20:
            raise BlockedJob("BLOCKED_INPUTS", "dataset registry identity is invalid")
        content = {key: value for key, value in dataset.items() if key != "registry_digest"}
        actual = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        if dataset.get("registry_digest") != actual:
            raise BlockedJob("BLOCKED_INPUTS", "dataset registry digest mismatch")
        target = settings.data_root / "manifests" / "registered" / f"{record_id}.json"
        atomic_write_json(target, dict(dataset))
        return DatasetRegistry(settings.data_root).verify_persisted(dataset)

    def _run_cohort_training(
        self,
        settings: LabSettings,
        config: Mapping[str, Any],
        dataset: Mapping[str, Any],
        dataset_path: Path,
        validation_path: Path | None,
        *,
        job_id: str,
        on_metric: Any,
    ) -> dict[str, Any]:
        capability = inspect_cuda(self.torch_module)
        if not capability.get("cuda_available"):
            raise BlockedJob("BLOCKED_CUDA", "CUDA became unavailable before cohort execution")
        torch_module = self.torch_module
        if torch_module is None:
            import torch as torch_module
        torch_module.cuda.set_device(0)

        members = list(config["members"])
        micro_batch = int(config.get("batch_size", 1))
        accumulation = int(config.get("gradient_accumulation_steps", 1))
        effective_batch = micro_batch * accumulation
        total_steps = int(config["train_steps"])
        start_position = int(config.get("dataset_position", 0))
        shuffle_seed = int(config.get("shuffle_seed", 1234))
        checkpoint_interval = int(config.get("checkpoint_interval", 50))
        evaluation_interval = int(config.get("evaluation_interval", total_steps))
        learning_rate = float(config.get("learning_rate", 0.0003))
        context_length = int(config.get("context_length", 2048))
        training_recipe = str(config.get("training_recipe_version", "causal-pretrain-v1"))
        weight_decay = float(config.get("weight_decay", 0.1))
        warmup_steps = int(config.get("warmup_steps", 100))
        min_lr_ratio = float(config.get("min_lr_ratio", 0.1))

        rows = [
            json.loads(line)
            for line in dataset_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        order = list(range(len(rows)))
        random.Random(shuffle_seed).shuffle(order)
        required = total_steps * effective_batch
        if start_position + required > len(order):
            raise BlockedJob(
                "BLOCKED_INPUTS",
                "cohort dataset does not contain enough unused shuffled examples",
            )
        validation_rows = []
        if validation_path is not None:
            validation_rows = [
                json.loads(line)
                for line in validation_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not validation_rows:
                raise BlockedJob("BLOCKED_INPUTS", "validation dataset is empty")

        adapters: dict[str, Any] = {}
        member_configs: dict[str, dict[str, Any]] = {}
        runs = RunStore(settings.lab_root)
        for member in members:
            run_id = str(member["run_id"])
            if runs.run_root(run_id).exists():
                raise BlockedJob("BLOCKED_INPUTS", f"cohort run already exists: {run_id}")
        for member in members:
            member_id = str(member["member_id"])
            adapter_name = str(member["model_adapter"])
            seed = int(member["seed"])
            spawned = self.adapters.spawn(
                adapter_name,
                seed=seed,
                learning_rate=learning_rate,
            )
            configure = getattr(spawned, "configure_for_training", None)
            if callable(configure):
                configure(
                    {
                        "seed": seed,
                        "learning_rate": learning_rate,
                        "context_length": context_length,
                        "train_steps": total_steps,
                        "training_recipe_version": training_recipe,
                        "weight_decay": weight_decay,
                        "warmup_steps": warmup_steps,
                        "min_lr_ratio": min_lr_ratio,
                    }
                )
            for required_method in ("begin_accumulation", "accumulate_step", "finish_accumulation"):
                if not callable(getattr(spawned, required_method, None)):
                    raise BlockedJob(
                        "BLOCKED_INPUTS",
                        f"adapter {adapter_name} does not support synchronized gradient accumulation",
                    )
            inspection = spawned.inspect()
            if not isinstance(inspection, Mapping) or inspection.get("device") != "cuda:0":
                raise BlockedJob("BLOCKED_CUDA", f"cohort member {member_id} is not on cuda:0")
            member_config = {
                "execution_target": "LOCAL",
                "job_type": "COHORT_MEMBER",
                "cohort_id": str(config["cohort_id"]),
                "member_id": member_id,
                "run_id": str(member["run_id"]),
                "model_adapter": adapter_name,
                "seed": seed,
                "train_steps": total_steps,
                "micro_batch_size": micro_batch,
                "gradient_accumulation_steps": accumulation,
                "effective_batch_size": effective_batch,
                "context_length": context_length,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "warmup_steps": warmup_steps,
                "min_lr_ratio": min_lr_ratio,
                "training_recipe_version": training_recipe,
                "checkpoint_interval": checkpoint_interval,
                "evaluation_interval": evaluation_interval,
                "shuffle_seed": shuffle_seed,
                "dataset_position_start": start_position,
            }
            runs.create(str(member["run_id"]), member_config, dataset)
            adapters[member_id] = spawned
            member_configs[member_id] = member_config

        metrics = MetricsStore(settings.lab_root)
        last_by_member: dict[str, dict[str, Any]] = {
            str(member["member_id"]): {"last_step": 0, "examples_seen": start_position}
            for member in members
        }
        completed_step = 0
        try:
            for step in range(1, total_steps + 1):
                if self._cancel_requested(job_id):
                    if completed_step:
                        for member in members:
                            member_id = str(member["member_id"])
                            run_id = str(member["run_id"])
                            checkpoint = self._save_checkpoint(
                                settings,
                                adapters[member_id],
                                {
                                    **member_configs[member_id],
                                    "batch_size": micro_batch,
                                },
                                step=completed_step,
                                final=False,
                                dataset_position=start_position + completed_step * effective_batch,
                            )
                            self._sync_checkpoint_best_effort(settings, run_id, checkpoint)
                    raise CancelledJob(
                        f"operator requested cohort stop at completed step {completed_step}"
                    )

                for adapter in adapters.values():
                    adapter.begin_accumulation()
                losses: dict[str, list[float]] = {member_id: [] for member_id in adapters}
                step_started = time.perf_counter()

                for accumulation_index in range(accumulation):
                    offset = (
                        start_position
                        + (step - 1) * effective_batch
                        + accumulation_index * micro_batch
                    )
                    indices = order[offset : offset + micro_batch]
                    batch = [rows[index] for index in indices]
                    for member_id, adapter in adapters.items():
                        observed = adapter.accumulate_step(batch, divisor=accumulation)
                        if not isinstance(observed, Mapping):
                            raise RuntimeError("adapter accumulate_step did not return metrics")
                        losses[member_id].append(float(observed["train_loss"]))

                for adapter in adapters.values():
                    adapter.finish_accumulation()

                examples_seen = start_position + step * effective_batch
                evaluated_now = bool(validation_rows) and (
                    step % evaluation_interval == 0 or step == total_steps
                )
                validation_batch = (
                    validation_rows[: min(micro_batch, len(validation_rows))]
                    if evaluated_now
                    else []
                )
                cohort_seconds = time.perf_counter() - step_started

                for member in members:
                    member_id = str(member["member_id"])
                    run_id = str(member["run_id"])
                    record = {
                        "step": step,
                        "examples_seen": examples_seen,
                        "micro_batch_size": micro_batch,
                        "gradient_accumulation_steps": accumulation,
                        "effective_batch_size": effective_batch,
                        "train_loss": sum(losses[member_id]) / len(losses[member_id]),
                        "cohort_seconds_per_step": cohort_seconds,
                    }
                    if evaluated_now:
                        evaluated = adapters[member_id].evaluate(validation_batch)
                        if not isinstance(evaluated, Mapping):
                            raise RuntimeError("adapter evaluate did not return metrics")
                        record.update({str(k): float(v) for k, v in evaluated.items()})
                    metrics.append(run_id, record)
                    last_by_member[member_id] = {"last_step": step, **record}

                completed_step = step
                checkpoint_now = step % checkpoint_interval == 0 and step < total_steps
                if checkpoint_now:
                    for member in members:
                        member_id = str(member["member_id"])
                        run_id = str(member["run_id"])
                        checkpoint = self._save_checkpoint(
                            settings,
                            adapters[member_id],
                            {
                                **member_configs[member_id],
                                "batch_size": micro_batch,
                            },
                            step=step,
                            final=False,
                            dataset_position=examples_seen,
                        )
                        self._sync_checkpoint_best_effort(settings, run_id, checkpoint)

                if step % 25 == 0 or evaluated_now or checkpoint_now or step == total_steps:
                    on_metric(
                        {
                            "step": step,
                            "examples_seen": examples_seen,
                            "effective_batch_size": effective_batch,
                            "members": {
                                member_id: last_by_member[member_id]
                                for member_id in sorted(last_by_member)
                            },
                        }
                    )

            final_position = start_position + total_steps * effective_batch
            for member in members:
                member_id = str(member["member_id"])
                run_id = str(member["run_id"])
                checkpoint = self._save_checkpoint(
                    settings,
                    adapters[member_id],
                    {
                        **member_configs[member_id],
                        "batch_size": micro_batch,
                    },
                    step=total_steps,
                    final=True,
                    dataset_position=final_position,
                )
                self._sync_checkpoint_best_effort(settings, run_id, checkpoint)
                self._flush_pending_sync(settings, run_id)
                runs.finalize(
                    run_id,
                    result="completed",
                    actual_metrics=last_by_member[member_id],
                    failures=[],
                )

            cohort_root = settings.lab_root / "cohorts" / str(config["cohort_id"])
            cohort_root.mkdir(parents=True, exist_ok=True)
            atomic_write_json(
                cohort_root / "COHORT_MANIFEST.json",
                {
                    "schema_version": "blackhole.cohort.v1",
                    "cohort_id": str(config["cohort_id"]),
                    "result": "completed",
                    "members": members,
                    "train_steps": total_steps,
                    "micro_batch_size": micro_batch,
                    "gradient_accumulation_steps": accumulation,
                    "effective_batch_size": effective_batch,
                    "shuffle_seed": shuffle_seed,
                    "dataset_position_start": start_position,
                    "dataset_position_end": final_position,
                    "same_examples_for_all_members": True,
                    "same_optimizer_update_boundaries": True,
                },
            )
            try:
                self._upload_directory(
                    cohort_root,
                    f"{LAB_PREFIX}cohorts/{config['cohort_id']}",
                )
            except Exception:
                pass
            return {
                "cohort_id": str(config["cohort_id"]),
                "last_step": total_steps,
                "examples_seen": final_position,
                "effective_batch_size": effective_batch,
                "members": last_by_member,
            }
        except CancelledJob:
            for member in members:
                member_id = str(member["member_id"])
                run_id = str(member["run_id"])
                try:
                    runs.finalize(
                        run_id,
                        result="stopped",
                        actual_metrics=last_by_member[member_id],
                        failures=[],
                    )
                except Exception:
                    pass
            raise
        except Exception as exc:
            for member in members:
                member_id = str(member["member_id"])
                run_id = str(member["run_id"])
                try:
                    runs.finalize(
                        run_id,
                        result="failed",
                        actual_metrics=last_by_member[member_id],
                        failures=[str(sanitize_remote_value(exc))],
                    )
                except Exception:
                    pass
            raise

    def _run_training(
        self,
        settings: LabSettings,
        adapter: Any,
        config: Mapping[str, Any],
        dataset: Mapping[str, Any],
        dataset_path: Path,
        validation_path: Path | None,
        *,
        on_metric: Any,
        should_cancel: Any = None,
    ) -> dict[str, Any]:
        """Run one deterministic, explicitly bounded dataset epoch segment on local CUDA."""
        capability = inspect_cuda(self.torch_module)
        if not capability.get("cuda_available"):
            raise BlockedJob("BLOCKED_CUDA", "CUDA became unavailable before execution")
        torch_module = self.torch_module
        if torch_module is None:
            import torch as torch_module
        torch_module.cuda.set_device(0)

        controller = TrainingController(settings, self.adapters)
        run_id = str(config["run_id"])
        controller.start(config, dataset, owner_authorized=True)
        metrics = MetricsStore(settings.lab_root)
        step_offset = int(config.get("resume_step", 0))
        total_steps = int(config["train_steps"])
        batch_size = int(config.get("batch_size", 1))
        accumulation = int(config.get("gradient_accumulation_steps", 1))
        effective_batch = batch_size * accumulation
        checkpoint_interval = int(config.get("checkpoint_interval", 50))
        evaluation_interval = int(config.get("evaluation_interval", total_steps))
        dataset_position = int(config.get("dataset_position", 0))
        dataset_epoch = int(config.get("dataset_epoch", 0))
        shuffle_seed = int(config.get("shuffle_seed", config.get("seed", 1234)))
        validation_max_rows = int(config.get("validation_max_rows", 256))
        recipe = str(config.get("training_recipe_version", "causal-pretrain-v1"))
        if checkpoint_interval < 1:
            raise BlockedJob("BLOCKED_INPUTS", "checkpoint_interval must be at least 1")
        if evaluation_interval < 1:
            raise BlockedJob("BLOCKED_INPUTS", "evaluation_interval must be at least 1")
        if recipe not in FOUNDATION_TRAINING_RECIPES:
            raise BlockedJob("BLOCKED_INPUTS", "unsupported training recipe")

        rows = [
            json.loads(line)
            for line in dataset_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows:
            raise BlockedJob("BLOCKED_INPUTS", "approved training dataset is empty")
        order = list(range(len(rows)))
        random.Random(shuffle_seed + dataset_epoch).shuffle(order)
        required = total_steps * effective_batch
        if dataset_position < 0 or dataset_position + required > len(order):
            raise BlockedJob(
                "BLOCKED_INPUTS",
                f"dataset epoch requires {required} examples from position {dataset_position}, "
                f"but only {len(order)} rows are available",
            )

        validation_rows: list[dict[str, Any]] = []
        if validation_path is not None:
            validation_rows = [
                json.loads(line)
                for line in validation_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not validation_rows:
                raise BlockedJob("BLOCKED_INPUTS", "validation dataset is empty")
            validation_rows = validation_rows[: min(validation_max_rows, len(validation_rows))]

        def aggregate_validation() -> dict[str, float]:
            if not validation_rows:
                return {}
            weighted_loss = 0.0
            examples = 0
            for offset in range(0, len(validation_rows), batch_size):
                batch = validation_rows[offset : offset + batch_size]
                observed = adapter.evaluate(batch)
                if not isinstance(observed, Mapping) or "val_loss" not in observed:
                    raise RuntimeError("adapter evaluate did not return val_loss")
                weighted_loss += float(observed["val_loss"]) * len(batch)
                examples += len(batch)
            return {
                "val_loss": weighted_loss / examples,
                "validation_examples": float(examples),
            }

        last: dict[str, Any] = {
            "last_step": step_offset,
            "dataset_epoch": dataset_epoch,
            "dataset_position": dataset_position,
        }
        try:
            for local_step in range(1, total_steps + 1):
                completed_position = dataset_position + (local_step - 1) * effective_batch
                if callable(should_cancel) and (local_step == 1 or local_step % 10 == 0) and should_cancel():
                    completed_step = step_offset + local_step - 1
                    if completed_step > step_offset:
                        checkpoint = self._save_checkpoint(
                            settings,
                            adapter,
                            config,
                            step=completed_step,
                            final=False,
                            dataset_position=completed_position,
                        )
                        self._sync_checkpoint_best_effort(settings, run_id, checkpoint)
                    raise CancelledJob(f"operator requested stop at completed step {completed_step}")

                step = step_offset + local_step
                started = time.perf_counter()
                begin = getattr(adapter, "begin_accumulation", None)
                accumulate = getattr(adapter, "accumulate_step", None)
                finish = getattr(adapter, "finish_accumulation", None)
                observed_totals: dict[str, float] = {}
                if accumulation > 1 and not (callable(begin) and callable(accumulate) and callable(finish)):
                    raise RuntimeError("adapter does not support gradient accumulation")
                if callable(begin) and callable(accumulate) and callable(finish):
                    begin()
                    for micro_step in range(accumulation):
                        offset = (
                            dataset_position
                            + (local_step - 1) * effective_batch
                            + micro_step * batch_size
                        )
                        indices = order[offset : offset + batch_size]
                        batch = [rows[index] for index in indices]
                        micro = accumulate(batch, divisor=accumulation)
                        if not isinstance(micro, Mapping):
                            raise RuntimeError("adapter accumulate_step did not return metrics")
                        for key, value in micro.items():
                            observed_totals[str(key)] = observed_totals.get(str(key), 0.0) + float(value)
                    finish()
                    observed = {
                        key: value / accumulation
                        for key, value in observed_totals.items()
                    }
                else:
                    offset = dataset_position + (local_step - 1) * effective_batch
                    indices = order[offset : offset + batch_size]
                    batch = [rows[index] for index in indices]
                    observed = adapter.train_step(batch)
                    if not isinstance(observed, Mapping):
                        raise RuntimeError("adapter train_step did not return metrics")
                current_position = dataset_position + local_step * effective_batch
                record: dict[str, Any] = {
                    "step": step,
                    **{str(key): float(value) for key, value in observed.items()},
                    "dataset_epoch": dataset_epoch,
                    "dataset_position": current_position,
                    "examples_seen_this_epoch": current_position,
                    "gradient_accumulation_steps": accumulation,
                    "effective_batch_size": effective_batch,
                }
                evaluated_now = bool(validation_rows) and (
                    local_step % evaluation_interval == 0 or local_step == total_steps
                )
                if evaluated_now:
                    record.update(aggregate_validation())
                record["seconds_per_step"] = time.perf_counter() - started
                metrics.append(run_id, record)
                last = {"last_step": step, **record}

                final_step = step_offset + total_steps
                checkpoint_now = local_step % checkpoint_interval == 0 and step < final_step
                if checkpoint_now:
                    checkpoint = self._save_checkpoint(
                        settings,
                        adapter,
                        config,
                        step=step,
                        final=False,
                        dataset_position=current_position,
                    )
                    self._sync_checkpoint_best_effort(settings, run_id, checkpoint)

                if local_step % 25 == 0 or evaluated_now or checkpoint_now or local_step == total_steps:
                    try:
                        on_metric(record)
                    except Exception:
                        pass

            final_step = step_offset + total_steps
            final_position = dataset_position + total_steps * effective_batch
            checkpoint = self._save_checkpoint(
                settings,
                adapter,
                config,
                step=final_step,
                final=True,
                dataset_position=final_position,
            )
            self._sync_checkpoint_best_effort(settings, run_id, checkpoint)
            self._flush_pending_sync(settings, run_id)
            controller.complete(actual_metrics=last)
            return last
        except CancelledJob:
            if controller.status().get("status") == "running":
                controller.stop(actual_metrics=last)
            raise
        except Exception as exc:
            if controller.status().get("status") == "running":
                controller.fail(actual_metrics=last, failures=[str(sanitize_remote_value(exc))])
            raise

    def _save_checkpoint(
        self,
        settings: LabSettings,
        adapter: Any,
        config: Mapping[str, Any],
        *,
        step: int,
        final: bool,
        dataset_position: int | None = None,
    ) -> Path:
        run_id = str(config["run_id"])
        run_root = settings.lab_root / "runs" / run_id
        checkpoint = run_root / ("checkpoint-final" if final else f"checkpoint-step-{step:08d}")
        save_result = adapter.save(checkpoint)
        files = save_result.get("files") if isinstance(save_result, Mapping) else None
        if not isinstance(files, Mapping) or "model" not in files:
            raise RuntimeError("adapter save did not return a model artifact")
        manifest_files: dict[str, str] = {}
        hashes: dict[str, str] = {}
        for role, filename in files.items():
            relative = Path(str(filename))
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError("adapter returned an unsafe artifact path")
            artifact = checkpoint / relative
            if not artifact.is_file():
                raise RuntimeError(f"adapter artifact is missing: {role}")
            manifest_files[str(role)] = relative.as_posix()
            hashes[str(role)] = sha256_file(artifact)
        atomic_write_json(
            checkpoint / "CHECKPOINT_MANIFEST.json",
            {
                "schema_version": "blackhole.checkpoint.v1",
                "checkpoint_id": f"{run_id}-{'final' if final else f'step-{step}'}",
                "run_id": run_id,
                "step": step,
                "dataset_position": (
                    int(dataset_position)
                    if dataset_position is not None
                    else step * int(config.get("batch_size", 1))
                ),
                "seed": config.get("seed"),
                "model_adapter": str(config["model_adapter"]),
                "trained": True,
                "checkpoint_kind": "final" if final else "recovery",
                "files": manifest_files,
                "sha256": hashes,
            },
        )
        if not final:
            recovery = sorted(
                path for path in run_root.glob("checkpoint-step-*")
                if path.is_dir()
            )
            for old in recovery[:-3]:
                shutil.rmtree(old)
        return checkpoint

    def _upload_directory(self, local_root: Path, remote_root: str) -> None:
        if self.filesystem is None:
            from huggingface_hub import HfFileSystem
            self.filesystem = HfFileSystem(token=True)
        for local_path in sorted(path for path in local_root.rglob("*") if path.is_file()):
            relative = local_path.relative_to(local_root).as_posix()
            with local_path.open("rb") as source, self.filesystem.open(
                f"{remote_root}/{relative}", "wb"
            ) as destination:
                while chunk := source.read(1024 * 1024):
                    destination.write(chunk)

    def _sync_checkpoint_best_effort(
        self,
        settings: LabSettings,
        run_id: str,
        checkpoint: Path,
    ) -> None:
        run_root = settings.lab_root / "runs" / run_id
        pending_path = run_root / "PENDING_SYNC.json"
        pending: list[str] = []
        if pending_path.is_file():
            try:
                payload = json.loads(pending_path.read_text(encoding="utf-8"))
                pending = [str(item) for item in payload.get("checkpoints", [])]
            except (OSError, json.JSONDecodeError):
                pending = []
        if checkpoint.name not in pending:
            pending.append(checkpoint.name)
        atomic_write_json(
            pending_path,
            {"schema_version": "blackhole.pending-sync.v1", "checkpoints": pending},
        )
        self._flush_pending_sync(settings, run_id)

    def _flush_pending_sync(self, settings: LabSettings, run_id: str) -> None:
        run_root = settings.lab_root / "runs" / run_id
        pending_path = run_root / "PENDING_SYNC.json"
        if not pending_path.is_file():
            return
        try:
            payload = json.loads(pending_path.read_text(encoding="utf-8"))
            pending = [str(item) for item in payload.get("checkpoints", [])]
        except (OSError, json.JSONDecodeError):
            return
        remaining: list[str] = []
        for name in pending:
            checkpoint = run_root / name
            if not checkpoint.is_dir():
                continue
            try:
                self._upload_directory(
                    checkpoint,
                    f"{LAB_PREFIX}runs/{run_id}/{name}",
                )
            except Exception:
                remaining.append(name)
        if remaining:
            atomic_write_json(
                pending_path,
                {"schema_version": "blackhole.pending-sync.v1", "checkpoints": remaining},
            )
        else:
            pending_path.unlink(missing_ok=True)

    def _upload_run(self, settings: LabSettings, run_id: str) -> tuple[list[str], dict[str, str]]:
        if self.filesystem is None:
            from huggingface_hub import HfFileSystem

            self.filesystem = HfFileSystem(token=True)
        run_root = settings.lab_root / "runs" / run_id
        references: list[str] = []
        hashes: dict[str, str] = {}
        for local_path in sorted(path for path in run_root.rglob("*") if path.is_file()):
            relative = local_path.relative_to(run_root).as_posix()
            remote = f"{LAB_PREFIX}runs/{run_id}/{relative}"
            with local_path.open("rb") as source, self.filesystem.open(remote, "wb") as destination:
                while chunk := source.read(1024 * 1024):
                    destination.write(chunk)
            references.append(remote)
            hashes[remote] = sha256_file(local_path)
        if not references:
            raise RuntimeError("no durable run artifacts were produced")
        return references, hashes

    def _update(
        self,
        job: Mapping[str, Any],
        claim_token: str,
        state: str,
        details: Mapping[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "job_id": str(job.get("job_id") or ""),
            "worker_id": self.worker_id,
            "claim_token": claim_token,
            "state": state,
            "details": sanitize_remote_value(dict(details)),
        }
        return self._call("local_worker_update", payload)

    def _call(self, name: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        response = self.client.predict(dict(payload), api_name=f"/{name}")
        if not isinstance(response, Mapping):
            raise RuntimeError(f"controller API {name} returned a non-object response")
        return dict(response)
