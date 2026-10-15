import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace


class FakeClient:
    def __init__(self, *, fail_catalog: bool = False, fail_space: bool = False, fail_run: bool = False) -> None:
        self.fail_catalog = fail_catalog
        self.fail_space = fail_space
        self.fail_run = fail_run

    def get_space(self):
        if self.fail_space:
            raise RuntimeError("Authorization: Bearer bridge-secret")
        return None

    def get_jobs(self):
        return []

    def get_dataset_catalog(self):
        if self.fail_catalog:
            raise RuntimeError("authorization hf_secretvalue123456 failed")
        return {
            "bucket": "STATICAI/BLACKHOLE/DATA",
            "catalog": "manifests/dataset_catalog_v2.json",
            "primary_language": "en",
            "english_train_pairs": 5240,
            "english_validation_pairs": 289,
            "audit_completed": False,
        }

    def get_latest_run(self):
        if self.fail_run:
            raise RuntimeError("https://x.test/?token=bridge-query-secret")
        return None


def test_configured_volume_records_prefer_space_info_runtime() -> None:
    from bridge.blackhole_status import _configured_volume_records

    volume = SimpleNamespace(
        type="bucket",
        source="STATICAI/BLACKHOLE",
        path="DATA",
        mount_path="/data",
        read_only=False,
    )
    info = SimpleNamespace(runtime=SimpleNamespace(volumes=[volume]))
    runtime = SimpleNamespace(volumes=None)

    assert _configured_volume_records(info, runtime) == [
        {
            "type": "bucket",
            "source": "STATICAI/BLACKHOLE",
            "path": "DATA",
            "mount_path": "/data",
            "read_only": False,
        }
    ]


def test_missing_space_and_metrics_are_truthful_not_deployed_and_unknown() -> None:
    from bridge.blackhole_status import collect_status

    status = collect_status(FakeClient(), observed_at=datetime(2026, 10, 10, 17, 0, tzinfo=UTC))

    assert status["schema_version"] == "blackhole.status.v1"
    assert status["space"]["state"] == "NOT_DEPLOYED"
    assert status["training"]["state"] == "NOT_STARTED"
    assert status["training"]["latest_step"] is None
    assert status["training"]["train_loss"] is None
    assert status["latest_job"] is None
    assert status["source_state"] == "CURRENT"


def test_partial_source_failure_is_blocked_and_redacted() -> None:
    from bridge.blackhole_status import collect_status

    status = collect_status(FakeClient(fail_catalog=True))
    serialized = json.dumps(status)

    assert status["source_state"] == "BLOCKED"
    assert status["datasets"]["state"] == "UNKNOWN"
    assert "hf_secretvalue123456" not in serialized
    assert "[REDACTED]" in serialized


def test_failed_space_and_run_sources_are_unknown_not_invented_absence() -> None:
    from bridge.blackhole_status import collect_status

    status = collect_status(FakeClient(fail_space=True, fail_run=True))
    serialized = json.dumps(status)
    assert status["space"]["state"] == "UNKNOWN"
    assert status["training"]["state"] == "UNKNOWN"
    assert "bridge-secret" not in serialized
    assert "bridge-query-secret" not in serialized


def test_write_outputs_stable_json_and_timestamped_markdown(tmp_path: Path) -> None:
    from bridge.blackhole_status import collect_status, write_outputs

    observed = datetime(2026, 10, 10, 17, 5, tzinfo=UTC)
    status = collect_status(FakeClient(), observed_at=observed)
    report = write_outputs(status, tmp_path / "bridge/status.json", tmp_path / "reports")

    assert json.loads((tmp_path / "bridge/status.json").read_text())["observed_at"] == "2026-10-10T17:05:00Z"
    assert report.name == "2026-10-10_1705.md"
    rendered = report.read_text(encoding="utf-8")
    assert "cached snapshot" in rendered.lower()
    assert "NOT_STARTED" in rendered
