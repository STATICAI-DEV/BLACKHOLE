from pathlib import Path


def _manifest(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "dataset_id": "OpenAssistant/oasst2",
        "revision": "179dd21fc55192153d94adb0e0ce8f69e222bf75",
        "license": "apache-2.0",
        "license_review": "approved",
        "quality_review": "approved",
        "role": "chat",
        "split": "train",
        "language": "en",
        "rows": 5240,
        "content_sha256": "a" * 64,
        "approval_status": "Approved",
    }
    payload.update(overrides)
    return payload


def test_missing_license_is_blocked_even_when_approval_is_requested(tmp_path: Path) -> None:
    from blackhole.datasets import DatasetRegistry

    record = DatasetRegistry(tmp_path).register(_manifest(license="", approval_status="Approved"))

    assert record["approval_status"] == "Blocked"
    assert record["eligible_for_training"] is False
    assert "license" in record["blockers"]


def test_evaluation_or_test_data_never_becomes_training_eligible(tmp_path: Path) -> None:
    from blackhole.datasets import DatasetRegistry

    evaluation = DatasetRegistry(tmp_path).register(
        _manifest(dataset_id="blackhole/held-out", role="evaluation", split="test")
    )

    assert evaluation["approval_status"] == "Approved"
    assert evaluation["eligible_for_training"] is False
    assert "held_out" in evaluation["blockers"]


def test_approved_training_manifest_round_trips_without_dataset_rows(tmp_path: Path) -> None:
    from blackhole.datasets import DatasetRegistry

    registry = DatasetRegistry(tmp_path)
    record = registry.register(_manifest())

    assert record["eligible_for_training"] is True
    assert registry.list_records() == [record]
    assert "samples" not in record


def test_ambiguous_license_and_invalid_schema_types_are_blocked(tmp_path: Path) -> None:
    from blackhole.datasets import DatasetRegistry

    record = DatasetRegistry(tmp_path).register(_manifest(license="unknown", rows="many"))
    assert record["approval_status"] == "Blocked"
    assert {"license", "rows"}.issubset(record["blockers"])


def test_persisted_record_identity_and_digest_are_verified(tmp_path: Path) -> None:
    import pytest
    from blackhole.datasets import DatasetRegistry

    registry = DatasetRegistry(tmp_path)
    record = registry.register(_manifest())
    assert registry.verify_persisted(record)["record_id"] == record["record_id"]
    with pytest.raises(RuntimeError, match="DATASET_NOT_APPROVED"):
        registry.verify_persisted({**record, "split": "test", "eligible_for_training": True})


def test_optional_local_worker_artifact_uri_is_strictly_allowlisted(tmp_path: Path) -> None:
    from blackhole.datasets import DatasetRegistry

    registry = DatasetRegistry(tmp_path)
    ready = registry.register(
        _manifest(artifact_uri="hf://buckets/BluStatic/BLACKHOLE/DATA/training/oasst2.jsonl")
    )
    blocked = registry.register(
        _manifest(dataset_id="other/data", artifact_uri="https://example.com/arbitrary.py")
    )

    assert ready["eligible_for_training"] is True
    assert ready["artifact_uri"].endswith(".jsonl")
    assert blocked["eligible_for_training"] is False
    assert "artifact_uri" in blocked["blockers"]


def test_optional_validation_artifact_is_hash_bound_and_allowlisted(tmp_path: Path) -> None:
    from blackhole.datasets import DatasetRegistry

    registry = DatasetRegistry(tmp_path)
    ready = registry.register(
        _manifest(
            artifact_uri="hf://buckets/BluStatic/BLACKHOLE/DATA/train.jsonl",
            validation_artifact_uri="hf://buckets/BluStatic/BLACKHOLE/DATA/validation.jsonl",
            validation_content_sha256="b" * 64,
        )
    )
    blocked = registry.register(
        _manifest(
            dataset_id="invalid/validation",
            validation_artifact_uri="hf://buckets/Other/place/validation.jsonl",
            validation_content_sha256="short",
        )
    )

    assert ready["eligible_for_training"] is True
    assert blocked["eligible_for_training"] is False
    assert {"validation_artifact_uri", "validation_content_sha256"} <= set(blocked["blockers"])
