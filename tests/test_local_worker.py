import json
from pathlib import Path


class FakeAdapter:
    name = "fake"

    def __init__(self) -> None:
        self.evaluated = 0
        self.configured = None
        self.batches = []
        self.spawned = []
        self.seed = None

    def spawn(self, *, seed, learning_rate):
        child = FakeAdapter()
        child.seed = seed
        child.learning_rate = learning_rate
        self.spawned.append(child)
        return child

    def begin_accumulation(self):
        self.accumulating = True

    def accumulate_step(self, batch, *, divisor):
        self.batches.append(batch)
        return {"train_loss": 2.0 + (float(self.seed or 0) / 100000.0)}

    def finish_accumulation(self):
        self.accumulating = False

    def inspect(self):
        return {"name": self.name, "device": "cuda:0"}

    def configure_for_training(self, config):
        self.configured = dict(config)

    def train_step(self, batch):
        self.batches.append(batch)
        return {"train_loss": 2.0}

    def evaluate(self, batch):
        self.evaluated += 1
        return {"val_loss": 1.5}

    def save(self, path):
        path.mkdir(parents=True, exist_ok=True)
        files = {
            "model": "model.pt",
            "optimizer": "optimizer.pt",
            "scheduler": "scheduler.pt",
            "random_state": "random_state.pt",
        }
        for filename in files.values():
            (path / filename).write_bytes(b"test")
        return {"files": files}

    def load(self, path):
        return None

    def generate(self, prompt, *, temperature, top_k, max_new_tokens):
        return "test"


class FakeCuda:
    def __init__(self, *, available: bool = True) -> None:
        self.available = available
        self.selected = None

    def is_available(self):
        return self.available

    def device_count(self):
        return 1 if self.available else 0

    def set_device(self, index):
        self.selected = index

    def get_device_name(self, index):
        return "Test NVIDIA GPU"

    def mem_get_info(self, index):
        return (7_000_000_000, 8_000_000_000)

    def get_device_capability(self, index):
        return (12, 0)


class FakeTorch:
    __version__ = "test"

    def __init__(self, *, available: bool = True) -> None:
        self.cuda = FakeCuda(available=available)


def _authorized_job(job: dict) -> dict:
    import hashlib

    config = job.get("config", {})
    digest = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        **job,
        "authorization": {
            "owner_authorized": True,
            "command_block_approved": True,
            "config_sha256": digest,
        },
    }


class FakeClient:
    def __init__(self, claim=None) -> None:
        self.claim = claim or {"status": "IDLE", "job": None}
        self.calls = []

    def predict(self, payload, *, api_name):
        self.calls.append((api_name, payload))
        if api_name == "/local_worker_announce":
            return {"status": "READY" if payload["cuda_available"] else "BLOCKED_CUDA"}
        if api_name == "/local_worker_claim":
            return self.claim
        if api_name == "/local_worker_update":
            return {"state": payload["state"], "result": payload["details"]}
        raise AssertionError(api_name)


def test_inspect_cuda_reports_real_device_shape_and_refuses_cpu() -> None:
    from blackhole.local_worker import inspect_cuda

    ready = inspect_cuda(FakeTorch())
    assert ready == {
        "cuda_available": True,
        "device": "cuda:0",
        "gpu_name": "Test NVIDIA GPU",
        "vram_free_bytes": 7_000_000_000,
        "vram_total_bytes": 8_000_000_000,
        "compute_capability": "12.0",
        "torch_version": "test",
    }
    blocked = inspect_cuda(FakeTorch(available=False))
    assert blocked["cuda_available"] is False
    assert blocked["status"] == "BLOCKED_CUDA"
    assert "cpu" not in json.dumps(blocked).lower()


def test_check_only_authenticates_and_announces_without_claiming(tmp_path: Path) -> None:
    from blackhole.adapters.registry import AdapterRegistry
    from blackhole.local_worker import LocalWorker

    client = FakeClient()
    worker = LocalWorker(client, AdapterRegistry(), tmp_path, torch_module=FakeTorch(), worker_id="worker-test")
    result = worker.check_only()

    assert result["status"] == "PASS"
    assert result["connection"] == "AUTHENTICATED"
    assert [call[0] for call in client.calls] == ["/local_worker_announce"]


def test_poll_without_execution_gate_never_claims(tmp_path: Path) -> None:
    from blackhole.adapters.registry import AdapterRegistry
    from blackhole.local_worker import LocalWorker

    client = FakeClient()
    worker = LocalWorker(client, AdapterRegistry(), tmp_path, torch_module=FakeTorch(), worker_id="worker-test")
    result = worker.poll_once(execute_training=False)

    assert result["status"] == "TRAINING_DISABLED"
    assert [call[0] for call in client.calls] == ["/local_worker_announce"]


def test_claimed_job_with_missing_adapter_is_blocked_before_training(tmp_path: Path) -> None:
    from blackhole.adapters.registry import AdapterRegistry
    from blackhole.local_worker import LocalWorker

    job = {
        "schema_version": "blackhole.local-job.v1",
        "job_id": "local-abc",
        "execution_target": "LOCAL",
        "config": {
            "execution_target": "LOCAL",
            "run_id": "run-001",
            "model_adapter": "missing",
            "train_steps": 1,
            "batch_size": 1,
            "seed": 7,
        },
        "dataset_manifest": {
            "record_id": "a" * 20,
            "content_sha256": "b" * 64,
            "artifact_uri": "hf://buckets/STATICAI/BLACKHOLE/DATA/train.jsonl",
        },
    }
    client = FakeClient({"status": "CLAIMED", "job": _authorized_job(job), "claim_token": "claim-secret"})
    worker = LocalWorker(client, AdapterRegistry(), tmp_path, torch_module=FakeTorch(), worker_id="worker-test")
    result = worker.poll_once(execute_training=True)

    assert result["state"] == "BLOCKED"
    assert result["result"]["code"] == "BLOCKED_INPUTS"
    assert [call[0] for call in client.calls] == [
        "/local_worker_announce",
        "/local_worker_claim",
        "/local_worker_update",
    ]
    rendered = json.dumps(client.calls)
    assert "claim-secret" in rendered
    assert str(tmp_path) not in rendered


def test_wrong_target_is_blocked_without_adapter_or_filesystem_calls(tmp_path: Path) -> None:
    from blackhole.adapters.registry import AdapterRegistry
    from blackhole.local_worker import LocalWorker

    job = {
        "schema_version": "blackhole.local-job.v1",
        "job_id": "local-abc",
        "execution_target": "CLOUD",
        "config": {},
        "dataset_manifest": {},
    }
    client = FakeClient({"status": "CLAIMED", "job": _authorized_job(job), "claim_token": "token"})
    worker = LocalWorker(client, AdapterRegistry(), tmp_path, torch_module=FakeTorch(), worker_id="worker-test")

    result = worker.poll_once(execute_training=True)
    assert result["state"] == "BLOCKED"
    assert result["result"]["code"] == "BLOCKED_JOB"


def test_claim_validation_rejects_unapproved_validation_artifact(tmp_path: Path) -> None:
    import pytest

    from blackhole.adapters.registry import AdapterRegistry
    from blackhole.local_worker import BlockedJob, LocalWorker

    registry = AdapterRegistry()
    registry.register(FakeAdapter())
    worker = LocalWorker(FakeClient(), registry, tmp_path, torch_module=FakeTorch())
    job = {
        "schema_version": "blackhole.local-job.v1",
        "execution_target": "LOCAL",
        "config": {"execution_target": "LOCAL", "model_adapter": "fake"},
        "dataset_manifest": {
            "artifact_uri": "hf://buckets/STATICAI/BLACKHOLE/DATA/train.jsonl",
            "content_sha256": "a" * 64,
            "validation_artifact_uri": "hf://buckets/STATICAI/OTHER/validation.jsonl",
            "validation_content_sha256": "b" * 64,
        },
    }

    with pytest.raises(BlockedJob, match="validation"):
        worker._validate_claim(job)


def test_claim_applies_job_hyperparameters_before_cuda_inspection(tmp_path: Path) -> None:
    from blackhole.adapters.registry import AdapterRegistry
    from blackhole.local_worker import LocalWorker

    adapter = FakeAdapter()
    registry = AdapterRegistry()
    registry.register(adapter)
    worker = LocalWorker(FakeClient(), registry, tmp_path, torch_module=FakeTorch())
    config = {
        "execution_target": "LOCAL",
        "model_adapter": "fake",
        "seed": 17,
        "learning_rate": 0.004,
        "context_length": 256,
    }
    worker._validate_claim(
        {
            "schema_version": "blackhole.local-job.v1",
            "execution_target": "LOCAL",
            "config": config,
            "dataset_manifest": {
                "artifact_uri": "hf://buckets/STATICAI/BLACKHOLE/DATA/train.jsonl",
                "content_sha256": "a" * 64,
            },
        }
    )

    assert adapter.configured == config


def test_training_evaluates_hash_bound_validation_split_at_interval(tmp_path: Path) -> None:
    from blackhole.adapters.registry import AdapterRegistry
    from blackhole.config import LabSettings
    from blackhole.datasets import DatasetRegistry
    from blackhole.local_worker import LocalWorker

    settings = LabSettings.from_env(
        {
            "BLACKHOLE_ENV": "test",
            "BLACKHOLE_DATA_ROOT": str(tmp_path / "data"),
            "BLACKHOLE_LAB_ROOT": str(tmp_path / "lab"),
        }
    )
    settings.ensure_storage()
    dataset = DatasetRegistry(settings.data_root).register(
        {
            "dataset_id": "example/data",
            "revision": "rev",
            "license": "apache-2.0",
            "license_review": "approved",
            "quality_review": "approved",
            "content_sha256": "a" * 64,
            "approval_status": "Approved",
            "role": "chat",
            "split": "train",
            "language": "en",
            "rows": 2,
        }
    )
    row = json.dumps({"messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]})
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    train_path.write_text(f"{row}\n{row}\n", encoding="utf-8")
    validation_path.write_text(f"{row}\n", encoding="utf-8")
    adapter = FakeAdapter()
    registry = AdapterRegistry()
    registry.register(adapter)
    metrics = []
    worker = LocalWorker(FakeClient(), registry, tmp_path, torch_module=FakeTorch())

    result = worker._run_training(
        settings,
        adapter,
        {
            "run_id": "run-validation",
            "model_adapter": "fake",
            "train_steps": 2,
            "batch_size": 1,
            "evaluation_interval": 2,
        },
        dataset,
        train_path,
        validation_path,
        on_metric=lambda metric: metrics.append(metric) or {"status": "RUNNING"},
    )

    assert adapter.evaluated == 1
    assert result["val_loss"] == 1.5
    # Remote telemetry is buffered/non-fatal; local MetricsStore retains every step.
    assert len(metrics) == 1
    assert metrics[0]["step"] == 2
    assert metrics[0]["val_loss"] == 1.5


def test_powershell_launcher_requires_an_explicit_mode() -> None:
    script = Path("scripts/start_local_gpu_worker.ps1").read_text(encoding="utf-8")
    python_worker = Path("scripts/local_gpu_worker.py").read_text(encoding="utf-8")
    assert "[switch]$CheckOnly" in script
    assert "[switch]$ExecuteTraining" in script
    assert "Exactly one of -CheckOnly or -ExecuteTraining" in script
    assert "[switch]$Once" in script
    assert "Start-Service" not in script
    assert "New-NetFirewallRule" not in script
    assert "truststore.inject_into_ssl()" in python_worker
    assert "ssl_verify=False" not in python_worker
    assert "build_default_registry(device=\"cuda:0\")" in python_worker
    assert "BLACKHOLE_GPU_WORKER" in script
    assert "--job-id" in python_worker
    exact_runner = Path("scripts/run_authorized_local_job.ps1").read_text(encoding="utf-8")
    assert "[string]$JobId" in exact_runner
    assert "--job-id $JobId" in exact_runner


def test_resume_training_skips_consumed_rows_and_preserves_absolute_step(tmp_path: Path) -> None:
    import pytest

    from blackhole.adapters.registry import AdapterRegistry
    from blackhole.config import LabSettings
    from blackhole.datasets import DatasetRegistry
    from blackhole.local_worker import LocalWorker

    settings = LabSettings.from_env(
        {
            "BLACKHOLE_ENV": "test",
            "BLACKHOLE_DATA_ROOT": str(tmp_path / "data"),
            "BLACKHOLE_LAB_ROOT": str(tmp_path / "lab"),
        }
    )
    settings.ensure_storage()
    dataset = DatasetRegistry(settings.data_root).register(
        {
            "dataset_id": "example/resume",
            "revision": "rev",
            "license": "apache-2.0",
            "license_review": "approved",
            "quality_review": "approved",
            "content_sha256": "c" * 64,
            "approval_status": "Approved",
            "role": "chat",
            "split": "train",
            "language": "en",
            "rows": 3,
        }
    )
    rows = [
        json.dumps({"messages": [{"role": "user", "content": str(i)}, {"role": "assistant", "content": "a"}]})
        for i in range(3)
    ]
    train_path = tmp_path / "resume.jsonl"
    train_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    adapter = FakeAdapter()
    registry = AdapterRegistry()
    registry.register(adapter)
    worker = LocalWorker(FakeClient(), registry, tmp_path, torch_module=FakeTorch())

    result = worker._run_training(
        settings,
        adapter,
        {
            "run_id": "resume-run",
            "model_adapter": "fake",
            "train_steps": 1,
            "batch_size": 1,
            "resume_step": 2,
            "dataset_position": 2,
            "dataset_epoch": 0,
            "shuffle_seed": 17,
            "checkpoint_interval": 50,
        },
        dataset,
        train_path,
        None,
        on_metric=lambda metric: {"status": "RUNNING"},
    )

    assert result["step"] == 3
    import random
    order = list(range(3))
    random.Random(17).shuffle(order)
    assert adapter.batches[0][0]["messages"][0]["content"] == str(order[2])
    manifest = json.loads(
        (settings.lab_root / "runs" / "resume-run" / "checkpoint-final" / "CHECKPOINT_MANIFEST.json").read_text()
    )
    assert manifest["step"] == 3


def test_safe_cancel_finalizes_run_as_stopped(tmp_path: Path) -> None:
    import pytest

    from blackhole.adapters.registry import AdapterRegistry
    from blackhole.config import LabSettings
    from blackhole.datasets import DatasetRegistry
    from blackhole.local_worker import CancelledJob, LocalWorker

    settings = LabSettings.from_env(
        {
            "BLACKHOLE_ENV": "test",
            "BLACKHOLE_DATA_ROOT": str(tmp_path / "data"),
            "BLACKHOLE_LAB_ROOT": str(tmp_path / "lab"),
        }
    )
    settings.ensure_storage()
    dataset = DatasetRegistry(settings.data_root).register(
        {
            "dataset_id": "example/cancel",
            "revision": "rev",
            "license": "apache-2.0",
            "license_review": "approved",
            "quality_review": "approved",
            "content_sha256": "d" * 64,
            "approval_status": "Approved",
            "role": "chat",
            "split": "train",
            "language": "en",
            "rows": 2,
        }
    )
    row = json.dumps({"messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]})
    train_path = tmp_path / "cancel.jsonl"
    train_path.write_text(row + "\n" + row + "\n", encoding="utf-8")
    adapter = FakeAdapter()
    registry = AdapterRegistry()
    registry.register(adapter)
    local = LocalWorker(FakeClient(), registry, tmp_path, torch_module=FakeTorch())

    with pytest.raises(CancelledJob):
        local._run_training(
            settings,
            adapter,
            {
                "run_id": "cancel-run",
                "model_adapter": "fake",
                "train_steps": 2,
                "batch_size": 1,
                "checkpoint_interval": 1,
            },
            dataset,
            train_path,
            None,
            on_metric=lambda metric: {"status": "RUNNING"},
            should_cancel=lambda: True,
        )
    manifest = json.loads((settings.lab_root / "runs" / "cancel-run" / "RUN_MANIFEST.json").read_text())
    assert manifest["result"] == "stopped"


def test_cohort_training_uses_one_shared_row_order_for_three_independent_models(tmp_path: Path) -> None:
    from blackhole.adapters.registry import AdapterRegistry
    from blackhole.config import LabSettings
    from blackhole.datasets import DatasetRegistry
    from blackhole.local_worker import LocalWorker

    settings = LabSettings.from_env(
        {
            "BLACKHOLE_ENV": "test",
            "BLACKHOLE_DATA_ROOT": str(tmp_path / "data"),
            "BLACKHOLE_LAB_ROOT": str(tmp_path / "lab"),
        }
    )
    settings.ensure_storage()
    dataset = DatasetRegistry(settings.data_root).register(
        {
            "dataset_id": "example/cohort",
            "revision": "rev",
            "license": "apache-2.0",
            "license_review": "approved",
            "quality_review": "approved",
            "content_sha256": "9" * 64,
            "approval_status": "Approved",
            "role": "chat",
            "split": "train",
            "language": "en",
            "rows": 8,
        }
    )
    rows = [
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": f"row-{i}"},
                    {"role": "assistant", "content": "a"},
                ]
            }
        )
        for i in range(8)
    ]
    train_path = tmp_path / "cohort.jsonl"
    train_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    prototype = FakeAdapter()
    registry = AdapterRegistry()
    registry.register(prototype)
    local = LocalWorker(FakeClient(), registry, tmp_path, torch_module=FakeTorch())
    local._cancel_requested = lambda job_id: False
    local._sync_checkpoint_best_effort = lambda *args, **kwargs: None
    local._flush_pending_sync = lambda *args, **kwargs: None
    local._upload_directory = lambda *args, **kwargs: None

    result = local._run_cohort_training(
        settings,
        {
            "cohort_id": "cohort-test",
            "members": [
                {"member_id": "T", "run_id": "cohort-t", "model_adapter": "fake", "seed": 1},
                {"member_id": "X", "run_id": "cohort-x", "model_adapter": "fake", "seed": 2},
                {"member_id": "K", "run_id": "cohort-k", "model_adapter": "fake", "seed": 3},
            ],
            "train_steps": 2,
            "batch_size": 2,
            "gradient_accumulation_steps": 2,
            "context_length": 256,
            "learning_rate": 0.0003,
            "checkpoint_interval": 10,
            "evaluation_interval": 10,
            "shuffle_seed": 77,
            "dataset_position": 0,
        },
        dataset,
        train_path,
        None,
        job_id="local-cohort-test",
        on_metric=lambda payload: None,
    )

    assert result["effective_batch_size"] == 4
    assert result["examples_seen"] == 8
    assert len(prototype.spawned) == 3
    traces = []
    for spawned in prototype.spawned:
        flattened = [
            row["messages"][0]["content"]
            for batch in spawned.batches
            for row in batch
        ]
        traces.append(flattened)
    assert traces[0] == traces[1] == traces[2]
    assert len(traces[0]) == 8

    for run_id in ("cohort-t", "cohort-x", "cohort-k"):
        checkpoint = settings.lab_root / "runs" / run_id / "checkpoint-final" / "CHECKPOINT_MANIFEST.json"
        manifest = json.loads(checkpoint.read_text())
        assert manifest["dataset_position"] == 8
        assert manifest["step"] == 2
