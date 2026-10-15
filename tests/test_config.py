from pathlib import Path

import pytest


def test_from_env_uses_isolated_roots_and_creates_test_storage(tmp_path: Path) -> None:
    from blackhole.config import LabSettings

    settings = LabSettings.from_env(
        {
            "BLACKHOLE_ENV": "test",
            "BLACKHOLE_DATA_ROOT": str(tmp_path / "data"),
            "BLACKHOLE_LAB_ROOT": str(tmp_path / "lab"),
        }
    )

    health = settings.ensure_storage()

    assert settings.environment == "test"
    assert health.ready is True
    assert health.code == "READY"
    assert (tmp_path / "data").is_dir()
    assert (tmp_path / "lab").is_dir()


def test_production_refuses_missing_mounts_without_creating_them(tmp_path: Path) -> None:
    from blackhole.config import BlockedStorageError, LabSettings

    data_root = tmp_path / "missing-data"
    lab_root = tmp_path / "missing-lab"
    settings = LabSettings.from_env(
        {
            "BLACKHOLE_ENV": "production",
            "BLACKHOLE_DATA_ROOT": str(data_root),
            "BLACKHOLE_LAB_ROOT": str(lab_root),
        }
    )

    with pytest.raises(BlockedStorageError, match="BLOCKED_STORAGE"):
        settings.ensure_storage()

    assert not data_root.exists()
    assert not lab_root.exists()


def test_boolean_flags_default_off_and_reject_implicit_truthiness(tmp_path: Path) -> None:
    from blackhole.config import LabSettings

    base = {
        "BLACKHOLE_ENV": "test",
        "BLACKHOLE_DATA_ROOT": str(tmp_path / "data"),
        "BLACKHOLE_LAB_ROOT": str(tmp_path / "lab"),
    }
    settings = LabSettings.from_env(base)
    assert settings.auto_train is False
    assert settings.auto_resume is False

    with pytest.raises(ValueError, match="BLACKHOLE_AUTO_TRAIN"):
        LabSettings.from_env({**base, "BLACKHOLE_AUTO_TRAIN": "sometimes"})


def test_environment_is_closed_enum_and_production_requires_mount_markers(tmp_path: Path) -> None:
    import json
    from blackhole.config import BlockedStorageError, LabSettings

    with pytest.raises(ValueError, match="BLACKHOLE_ENV"):
        LabSettings.from_env({"BLACKHOLE_ENV": "prodution"})
    data, lab = tmp_path / "data", tmp_path / "lab"
    data.mkdir(); lab.mkdir()
    settings = LabSettings("production", data, lab)
    with pytest.raises(BlockedStorageError, match="marker"):
        settings.ensure_storage()
    (data / ".blackhole-volume.json").write_text(json.dumps({"schema_version": "blackhole.volume.v1", "mount_path": "/data", "bucket": "STATICAI/BLACKHOLE/DATA"}))
    (lab / ".blackhole-volume.json").write_text(json.dumps({"schema_version": "blackhole.volume.v1", "mount_path": "/wrong", "bucket": "STATICAI/BLACKHOLE/LAB"}))
    with pytest.raises(BlockedStorageError, match="identity"):
        settings.ensure_storage()
    (lab / ".blackhole-volume.json").write_text(json.dumps({"schema_version": "blackhole.volume.v1", "mount_path": "/lab", "bucket": "STATICAI/BLACKHOLE/LAB"}))
    assert settings.ensure_storage().ready is True


@pytest.mark.parametrize("service", ["events", "datasets", "metrics", "evaluations", "runs"])
def test_persistence_services_refuse_missing_base_root(tmp_path: Path, service: str) -> None:
    from blackhole.config import BlockedStorageError
    from blackhole.datasets import DatasetRegistry
    from blackhole.evaluation import EvaluationStore
    from blackhole.metrics import MetricsStore
    from blackhole.provenance import EventStore
    from blackhole.runs import RunStore

    missing = tmp_path / "missing"
    with pytest.raises(BlockedStorageError, match="BLOCKED_STORAGE"):
        if service == "events": EventStore(missing).emit("run-1", "test", "blocked")
        elif service == "datasets": DatasetRegistry(missing).register({})
        elif service == "metrics": MetricsStore(missing).append("run-1", {"step": 0})
        elif service == "evaluations": EvaluationStore(missing).save("eval-1", {"prompt_set_version": "v1", "results": [{}]})
        else: RunStore(missing).create("run-1", {}, {})
