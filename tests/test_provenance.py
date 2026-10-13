import json
from pathlib import Path


def test_emit_creates_unique_redacted_event_files_and_audit_passes(tmp_path: Path) -> None:
    from blackhole.provenance import EventStore

    store = EventStore(tmp_path)
    first = store.emit(
        "run-001",
        "experiment_created",
        "success",
        details={"HF_TOKEN": "hf_privatevalue123", "note": "safe"},
        source_references=["hf://buckets/BluStatic/blackhole-lab/runs/run-001"],
    )
    second = store.emit("run-001", "model_selected", "blocked", details={"reason": "NO_MODEL"})

    assert first.path != second.path
    assert first.path.exists() and second.path.exists()
    payload = json.loads(first.path.read_text(encoding="utf-8"))
    assert payload["details"]["HF_TOKEN"] == "[REDACTED]"
    assert "hf_privatevalue123" not in first.path.read_text(encoding="utf-8")
    assert payload["timestamp_utc"].endswith("Z")
    assert store.audit("run-001").valid is True
    assert store.audit("run-001").event_count == 2


def test_event_store_rejects_path_traversal_run_ids(tmp_path: Path) -> None:
    import pytest

    from blackhole.provenance import EventStore

    with pytest.raises(ValueError, match="run_id"):
        EventStore(tmp_path).emit("../escape", "failure", "blocked")


def test_emit_recursively_redacts_all_persisted_fields(tmp_path: Path) -> None:
    from blackhole.provenance import EventStore

    serialized = EventStore(tmp_path).emit(
        "run-1", "test", "blocked",
        details={"headers": {"Authorization": "Bearer abc.def.ghi"}, "url": "https://x.test/?access_token=topsecret&ok=1"},
        file_hashes={"Cookie": "session=secret"},
        source_references=["Basic dXNlcjpwYXNz", "https://x.test/?api_key=hidden"],
        code_revision="Bearer revision-secret",
    ).path.read_text(encoding="utf-8")
    for secret in ("abc.def.ghi", "topsecret", "session=secret", "dXNlcjpwYXNz", "hidden", "revision-secret"):
        assert secret not in serialized
    assert "[REDACTED]" in serialized
