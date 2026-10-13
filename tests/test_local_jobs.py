import json
from pathlib import Path

import pytest


def _settings(tmp_path: Path):
    from blackhole.config import LabSettings

    settings = LabSettings.from_env(
        {
            "BLACKHOLE_ENV": "test",
            "BLACKHOLE_DATA_ROOT": str(tmp_path / "data"),
            "BLACKHOLE_LAB_ROOT": str(tmp_path / "lab"),
        }
    )
    settings.ensure_storage()
    return settings


def _dataset(settings) -> dict[str, object]:
    from blackhole.datasets import DatasetRegistry

    return DatasetRegistry(settings.data_root).register(
        {
            "dataset_id": "owner/training-data",
            "revision": "abc123",
            "license": "apache-2.0",
            "license_review": "approved",
            "quality_review": "approved",
            "content_sha256": "a" * 64,
            "approval_status": "Approved",
            "role": "chat",
            "split": "train",
            "language": "en",
            "rows": 1,
        }
    )


def _config(run_id: str = "local-run-001") -> dict[str, object]:
    return {
        "execution_target": "LOCAL",
        "run_id": run_id,
        "model_adapter": "static-0t",
        "seed": 7,
        "train_steps": 3,
        "batch_size": 1,
    }


def _worker(worker_id: str) -> dict[str, object]:
    return {
        "worker_id": worker_id,
        "cuda_available": True,
        "device": "cuda:0",
        "gpu_name": "Test GPU",
        "vram_free_bytes": 7_000_000_000,
        "vram_total_bytes": 8_000_000_000,
    }


def test_enqueue_requires_authorization_local_target_and_persisted_dataset(tmp_path: Path) -> None:
    from blackhole.local_jobs import LocalJobQueue

    settings = _settings(tmp_path)
    queue = LocalJobQueue(settings)
    dataset = _dataset(settings)

    with pytest.raises(PermissionError, match="TRAINING_NOT_AUTHORIZED"):
        queue.enqueue(_config(), dataset, owner_authorized=False)
    with pytest.raises(ValueError, match="execution_target must be LOCAL"):
        queue.enqueue({**_config(), "execution_target": "CLOUD"}, dataset, owner_authorized=True)
    with pytest.raises(ValueError, match="unsupported job configuration"):
        queue.enqueue({**_config(), "command": "whoami"}, dataset, owner_authorized=True)
    with pytest.raises(RuntimeError, match="DATASET_NOT_APPROVED"):
        queue.enqueue(_config(), {**dataset, "rows": 2}, owner_authorized=True)
    assert queue.list_jobs() == []


def test_unique_jobs_are_durable_and_contain_no_execution_command(tmp_path: Path) -> None:
    from blackhole.local_jobs import LocalJobQueue

    settings = _settings(tmp_path)
    queue = LocalJobQueue(settings)
    first = queue.enqueue(_config("run-a"), _dataset(settings), owner_authorized=True)
    second = queue.enqueue(_config("run-b"), _dataset(settings), owner_authorized=True)

    assert first["job_id"] != second["job_id"]
    assert first["state"] == "QUEUED"
    persisted = json.loads((settings.lab_root / "local-worker/jobs" / first["job_id"] / "JOB.json").read_text())
    assert persisted["schema_version"] == "blackhole.local-job.v1"
    assert persisted["execution_target"] == "LOCAL"
    assert "\"command\":" not in json.dumps(persisted).lower()
    assert persisted["authorization"]["command_block_approved"] is True
    assert len(persisted["authorization"]["config_sha256"]) == 64


def test_claim_is_exclusive_and_updates_require_matching_owner_and_token(tmp_path: Path) -> None:
    from blackhole.local_jobs import LocalJobQueue

    settings = _settings(tmp_path)
    queue = LocalJobQueue(settings)
    queued = queue.enqueue(_config(), _dataset(settings), owner_authorized=True)
    queue.announce(_worker("worker-a"))
    queue.announce(_worker("worker-b"))

    claim = queue.claim("worker-a")
    assert claim["status"] == "CLAIMED"
    assert claim["job"]["job_id"] == queued["job_id"]
    assert claim["claim_token"]
    busy = queue.claim("worker-b")
    assert busy["status"] == "BUSY"
    assert busy["active_job_id"] == queued["job_id"]

    with pytest.raises(PermissionError, match="CLAIM_OWNERSHIP_MISMATCH"):
        queue.update(queued["job_id"], "worker-b", claim["claim_token"], "RUNNING", {})
    with pytest.raises(PermissionError, match="CLAIM_OWNERSHIP_MISMATCH"):
        queue.update(queued["job_id"], "worker-a", "wrong-token", "RUNNING", {})

    running = queue.update(queued["job_id"], "worker-a", claim["claim_token"], "RUNNING", {"step": 0})
    assert running["state"] == "RUNNING"


def test_terminal_state_is_immutable_and_remote_status_is_sanitized(tmp_path: Path) -> None:
    from blackhole.local_jobs import LocalJobQueue

    settings = _settings(tmp_path)
    queue = LocalJobQueue(settings)
    queued = queue.enqueue(_config(), _dataset(settings), owner_authorized=True)
    queue.announce(_worker("worker-a"))
    claim = queue.claim("worker-a")

    result = queue.update(
        queued["job_id"],
        "worker-a",
        claim["claim_token"],
        "BLOCKED",
        {
            "error": "failed at C:\\Users\\operator\\private with hf_supersecret123456",
            "authorization": "Bearer should-not-survive",
        },
    )
    rendered = json.dumps(result)
    assert result["state"] == "BLOCKED"
    assert "C:\\Users\\operator" not in rendered
    assert "hf_supersecret123456" not in rendered
    assert "should-not-survive" not in rendered
    assert "[LOCAL_PATH]" in rendered
    assert "[REDACTED]" in rendered

    with pytest.raises(RuntimeError, match="TERMINAL_JOB"):
        queue.update(queued["job_id"], "worker-a", claim["claim_token"], "RUNNING", {})


def test_announce_rejects_cpu_worker_and_redacts_local_paths(tmp_path: Path) -> None:
    from blackhole.local_jobs import LocalJobQueue

    queue = LocalJobQueue(_settings(tmp_path))
    report = _worker("worker-a") | {
        "cuda_available": False,
        "error": "torch missing at C:\\private\\venv; token=hf_abcdefghijk",
    }
    result = queue.announce(report)
    rendered = json.dumps(result)

    assert result["status"] == "BLOCKED_CUDA"
    assert "C:\\private" not in rendered
    assert "hf_abcdefghijk" not in rendered



def test_local_evaluation_job_is_authorized_and_does_not_require_dataset(tmp_path: Path) -> None:
    from blackhole.local_jobs import LocalJobQueue

    queue = LocalJobQueue(_settings(tmp_path))
    config = {
        "execution_target": "LOCAL",
        "job_type": "EVALUATION",
        "evaluation_id": "eval-static0x-localgpu-test",
        "model_adapter": "static-0t",
        "checkpoint_run_id": "static0x-run",
        "prompt_set_version": "english-heldout-v1",
        "seed": 1234,
        "temperature": 0.2,
        "top_k": 20,
        "max_new_tokens": 128,
    }
    with pytest.raises(PermissionError, match="EVALUATION_NOT_AUTHORIZED"):
        queue.enqueue_evaluation(config, owner_authorized=False)
    queued = queue.enqueue_evaluation(config, owner_authorized=True)
    assert queued["state"] == "QUEUED"
    assert queued["config"]["job_type"] == "EVALUATION"
    assert queued["dataset_manifest"] == {}
    assert queued["authorization"]["scope"] == "LOCAL_EVALUATION"
    assert queued["authorization"]["command_block_approved"] is True


def test_exact_job_claim_and_safe_cancel_are_bound_to_requested_job(tmp_path: Path) -> None:
    from blackhole.local_jobs import LocalJobQueue

    settings = _settings(tmp_path)
    queue = LocalJobQueue(settings)
    first = queue.enqueue(_config("run-first"), _dataset(settings), owner_authorized=True)
    second = queue.enqueue(_config("run-second"), _dataset(settings), owner_authorized=True)
    queue.announce(_worker("worker-a"))

    claim = queue.claim("worker-a", second["job_id"])
    assert claim["status"] == "CLAIMED"
    assert claim["job"]["job_id"] == second["job_id"]
    assert queue.status(first["job_id"])["state"] == "QUEUED"

    cancelled = queue.request_cancel(first["job_id"])
    assert cancelled["state"] == "CANCELLED"
    assert cancelled["result"]["code"] == "CANCELLED"


def test_resume_job_uses_verified_checkpoint_and_remaining_steps(tmp_path: Path) -> None:
    import hashlib

    from blackhole.local_jobs import LocalJobQueue

    settings = _settings(tmp_path)
    dataset = _dataset(settings)
    source = settings.lab_root / "runs" / "run-source"
    checkpoint = source / "checkpoint-step-00000002"
    checkpoint.mkdir(parents=True)
    files = {}
    hashes = {}
    for role, name in {
        "model": "model.pt",
        "optimizer": "optimizer.pt",
        "scheduler": "scheduler.pt",
        "random_state": "random_state.pt",
    }.items():
        path = checkpoint / name
        path.write_bytes(role.encode())
        files[role] = name
        hashes[role] = hashlib.sha256(path.read_bytes()).hexdigest()
    (source / "RUN_CONFIG.json").write_text(
        json.dumps(
            {
                "schema_version": "blackhole.run-config.v1",
                "run_id": "run-source",
                "model_adapter": "static-0t",
                "seed": 7,
                "train_steps": 5,
                "batch_size": 1,
                "context_length": 256,
                "learning_rate": 0.0003,
                "checkpoint_interval": 2,
                "evaluation_interval": 2,
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "CHECKPOINT_MANIFEST.json").write_text(
        json.dumps(
            {
                "schema_version": "blackhole.checkpoint.v1",
                "checkpoint_id": "run-source-step-2",
                "run_id": "run-source",
                "step": 2,
                "dataset_position": 2,
                "model_adapter": "static-0t",
                "trained": True,
                "files": files,
                "sha256": hashes,
            }
        ),
        encoding="utf-8",
    )

    queue = LocalJobQueue(settings)
    resumed = queue.enqueue_resume(checkpoint, dataset, owner_authorized=True)
    assert resumed["state"] == "QUEUED"
    assert resumed["config"]["resume_step"] == 2
    assert resumed["config"]["dataset_position"] == 2
    assert resumed["config"]["train_steps"] == 3
    assert resumed["config"]["resume_checkpoint_uri"].endswith(
        "/LAB/runs/run-source/checkpoint-step-00000002"
    )


def test_generic_cancel_prefers_running_job_over_newer_queued_job(tmp_path: Path) -> None:
    from blackhole.local_jobs import LocalJobQueue

    settings = _settings(tmp_path)
    queue = LocalJobQueue(settings)
    running = queue.enqueue(_config("run-running"), _dataset(settings), owner_authorized=True)
    queue.announce(_worker("worker-a"))
    claim = queue.claim("worker-a", running["job_id"])
    queue.update(running["job_id"], "worker-a", claim["claim_token"], "RUNNING", {"step": 1})
    queued = queue.enqueue(_config("run-newer-queued"), _dataset(settings), owner_authorized=True)

    stopped = queue.request_cancel()
    assert stopped["job_id"] == running["job_id"]
    assert stopped["state"] == "RUNNING"
    assert stopped["cancel_requested"] is True
    assert queue.status(queued["job_id"])["state"] == "QUEUED"


def test_cohort_job_persists_three_members_and_effective_batch_contract(tmp_path: Path) -> None:
    from blackhole.datasets import DatasetRegistry
    from blackhole.local_jobs import LocalJobQueue

    settings = _settings(tmp_path)
    dataset = DatasetRegistry(settings.data_root).register(
        {
            "dataset_id": "owner/cohort-data",
            "revision": "abc123",
            "license": "apache-2.0",
            "license_review": "approved",
            "quality_review": "approved",
            "content_sha256": "e" * 64,
            "approval_status": "Approved",
            "role": "chat",
            "split": "train",
            "language": "en",
            "rows": 100,
        }
    )
    config = {
        "execution_target": "LOCAL",
        "job_type": "COHORT_TRAIN",
        "cohort_id": "batch2",
        "members": [
            {"member_id": "T", "run_id": "batch2-t", "model_adapter": "static-0t", "seed": 1234},
            {"member_id": "X", "run_id": "batch2-x", "model_adapter": "static-0t", "seed": 2401},
            {"member_id": "K", "run_id": "batch2-k", "model_adapter": "static-0t", "seed": 2402},
        ],
        "train_steps": 10,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "context_length": 256,
        "learning_rate": 0.0003,
        "checkpoint_interval": 5,
        "evaluation_interval": 5,
        "shuffle_seed": 99,
        "dataset_position": 0,
    }
    queue = LocalJobQueue(settings)
    queued = queue.enqueue_cohort(config, dataset, owner_authorized=True)
    assert queued["state"] == "QUEUED"
    assert queued["authorization"]["scope"] == "LOCAL_COHORT_TRAIN"
    assert queued["config"]["batch_size"] == 2
    assert queued["config"]["gradient_accumulation_steps"] == 4
    assert [m["member_id"] for m in queued["config"]["members"]] == ["T", "X", "K"]


def test_cohort_rejects_consuming_more_examples_than_approved_dataset(tmp_path: Path) -> None:
    from blackhole.datasets import DatasetRegistry
    from blackhole.local_jobs import LocalJobQueue

    settings = _settings(tmp_path)
    dataset = DatasetRegistry(settings.data_root).register(
        {
            "dataset_id": "owner/small-cohort-data",
            "revision": "abc123",
            "license": "apache-2.0",
            "license_review": "approved",
            "quality_review": "approved",
            "content_sha256": "f" * 64,
            "approval_status": "Approved",
            "role": "chat",
            "split": "train",
            "language": "en",
            "rows": 100,
        }
    )
    queue = LocalJobQueue(settings)
    with pytest.raises(ValueError, match="requires 10000 examples"):
        queue.enqueue_cohort(
            {
                "execution_target": "LOCAL",
                "job_type": "COHORT_TRAIN",
                "cohort_id": "huge-batch",
                "members": [
                    {"member_id": "T", "run_id": "huge-t", "model_adapter": "static-0t", "seed": 1},
                    {"member_id": "X", "run_id": "huge-x", "model_adapter": "static-0t", "seed": 2},
                ],
                "train_steps": 1,
                "batch_size": 2,
                "gradient_accumulation_steps": 5000,
            },
            dataset,
            owner_authorized=True,
        )
