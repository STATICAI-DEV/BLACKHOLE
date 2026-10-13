from pathlib import Path


class FakeAdapter:
    name = "fake"

    def inspect(self):
        return {"name": self.name, "parameters": 1}

    def train_step(self, batch):
        return {"loss": 1.0}

    def evaluate(self, batch):
        return {"loss": 1.0}

    def save(self, path):
        return {}

    def load(self, path):
        return None

    def generate(self, prompt, *, temperature, top_k, max_new_tokens):
        return "test-only"


def _settings(tmp_path: Path):
    from blackhole.config import LabSettings

    return LabSettings.from_env(
        {
            "BLACKHOLE_ENV": "test",
            "BLACKHOLE_DATA_ROOT": str(tmp_path / "data"),
            "BLACKHOLE_LAB_ROOT": str(tmp_path / "lab"),
        }
    )


def _dataset(tmp_path: Path) -> dict[str, object]:
    from blackhole.datasets import DatasetRegistry
    settings = _settings(tmp_path)
    settings.ensure_storage()
    return DatasetRegistry(settings.data_root).register({
        "dataset_id": "example/data",
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
    })


def _config(run_id: str = "run-001") -> dict[str, object]:
    return {"run_id": run_id, "model_adapter": "fake", "seed": 7, "train_steps": 3}


def test_start_requires_explicit_owner_authorization_before_writes(tmp_path: Path) -> None:
    import pytest

    from blackhole.adapters.registry import AdapterRegistry
    from blackhole.training import TrainingController

    registry = AdapterRegistry()
    registry.register(FakeAdapter())
    controller = TrainingController(_settings(tmp_path), registry)

    with pytest.raises(PermissionError, match="TRAINING_NOT_AUTHORIZED"):
        controller.start(_config(), _dataset(tmp_path), owner_authorized=False)
    assert not (tmp_path / "lab/runs").exists()


def test_start_rejects_unapproved_dataset_or_missing_adapter(tmp_path: Path) -> None:
    import pytest

    from blackhole.adapters.registry import AdapterRegistry
    from blackhole.training import TrainingController

    controller = TrainingController(_settings(tmp_path), AdapterRegistry())
    blocked = {**_dataset(tmp_path), "approval_status": "Blocked", "eligible_for_training": False}
    with pytest.raises(RuntimeError, match="NO_MODEL"):
        controller.start(_config(), blocked, owner_authorized=True)


def test_single_writer_lock_stop_and_status(tmp_path: Path) -> None:
    import pytest

    from blackhole.adapters.registry import AdapterRegistry
    from blackhole.training import TrainingController

    registry = AdapterRegistry()
    registry.register(FakeAdapter())
    first = TrainingController(_settings(tmp_path), registry)
    second = TrainingController(_settings(tmp_path), registry)

    state = first.start(_config(), _dataset(tmp_path), owner_authorized=True)
    assert state["status"] == "running"
    with pytest.raises(RuntimeError, match="TRAINING_ALREADY_ACTIVE"):
        second.start(_config("run-002"), _dataset(tmp_path), owner_authorized=True)

    stopped = first.request_stop()
    assert stopped["status"] == "stopped"
    assert first.status()["status"] == "idle"


def test_resume_refuses_incomplete_checkpoint(tmp_path: Path) -> None:
    import pytest

    from blackhole.adapters.registry import AdapterRegistry
    from blackhole.training import TrainingController

    registry = AdapterRegistry()
    registry.register(FakeAdapter())
    controller = TrainingController(_settings(tmp_path), registry)

    with pytest.raises(RuntimeError, match="RESUME_UNVERIFIED"):
        controller.resume(tmp_path / "missing", _dataset(tmp_path), owner_authorized=True)


def test_training_rejects_forged_or_held_out_manifest(tmp_path: Path) -> None:
    import pytest
    from blackhole.adapters.registry import AdapterRegistry
    from blackhole.training import TrainingController

    registry = AdapterRegistry()
    registry.register(FakeAdapter())
    forged = {**_dataset(tmp_path), "split": "test", "eligible_for_training": True}
    with pytest.raises(RuntimeError, match="DATASET_NOT_APPROVED"):
        TrainingController(_settings(tmp_path), registry).start(_config(), forged, owner_authorized=True)


def test_training_lock_release_requires_matching_nonce(tmp_path: Path) -> None:
    import json
    import pytest
    from blackhole.adapters.registry import AdapterRegistry
    from blackhole.training import TrainingController

    registry = AdapterRegistry()
    registry.register(FakeAdapter())
    controller = TrainingController(_settings(tmp_path), registry)
    controller.start(_config(), _dataset(tmp_path), owner_authorized=True)
    controller.lock_path.write_text(json.dumps({"run_id": "other", "nonce": "other"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="TRAINING_LOCK_OWNERSHIP_LOST"):
        controller.request_stop()
    assert controller.lock_path.exists()


def test_complete_finalizes_run_and_releases_owned_lock(tmp_path: Path) -> None:
    from blackhole.adapters.registry import AdapterRegistry
    from blackhole.training import TrainingController

    registry = AdapterRegistry()
    registry.register(FakeAdapter())
    controller = TrainingController(_settings(tmp_path), registry)
    controller.start(_config(), _dataset(tmp_path), owner_authorized=True)

    completed = controller.complete(actual_metrics={"last_step": 3, "train_loss": 0.5})

    assert completed == {"run_id": "run-001", "status": "completed"}
    assert controller.status() == {"status": "idle", "run_id": None}
    assert not controller.lock_path.exists()
    manifest = (tmp_path / "lab/runs/run-001/RUN_MANIFEST.json").read_text(encoding="utf-8")
    assert '"result": "completed"' in manifest
