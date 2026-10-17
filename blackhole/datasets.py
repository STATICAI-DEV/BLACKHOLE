"""Metadata-only dataset registration and training eligibility checks."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from .atomic import atomic_write_json
from .config import require_storage_root

_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_TRAINING_ROLES = {"pretraining", "reasoning", "software_engineering", "chat", "instruction", "business", "relationship", "browser_agent", "judgment", "preference"}
_HELD_OUT_ROLES = {"evaluation", "benchmark", "test"}
_AMBIGUOUS_LICENSES = {"", "unknown", "unspecified", "other", "none", "n/a", "tbd"}
_TEXT_FIELDS = ("dataset_id", "revision", "role", "split", "language")
_LOCAL_ARTIFACT_PREFIX = "hf://buckets/STATICAI/BLACKHOLE/DATA/"


class DatasetRegistry:
    def __init__(self, data_root: Path | str) -> None:
        self.data_root = Path(data_root)
        self.root = Path(data_root) / "manifests" / "registered"

    def register(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        require_storage_root(self.data_root, "data")
        record = dict(manifest)
        required = ("dataset_id", "revision", "role", "split", "content_sha256")
        blockers: list[str] = []
        for key in required:
            if record.get(key) in (None, ""):
                blockers.append(key)

        for key in _TEXT_FIELDS:
            if not isinstance(record.get(key), str):
                blockers.append(key)
        if not isinstance(record.get("rows"), int) or record.get("rows", -1) < 0:
            blockers.append("rows")
        license_value = str(record.get("license") or "").strip().lower()
        if license_value in _AMBIGUOUS_LICENSES:
            blockers.append("license")
        if record.get("license_review") != "approved":
            blockers.append("license_review")
        if record.get("quality_review") != "approved":
            blockers.append("quality_review")
        if not _SHA256.fullmatch(str(record.get("content_sha256") or "")):
            blockers.append("content_sha256")
        artifact_uri = record.get("artifact_uri")
        if artifact_uri not in (None, "") and (
            not isinstance(artifact_uri, str)
            or not artifact_uri.startswith(_LOCAL_ARTIFACT_PREFIX)
            or not artifact_uri.endswith(".jsonl")
        ):
            blockers.append("artifact_uri")
        validation_uri = record.get("validation_artifact_uri")
        validation_hash = record.get("validation_content_sha256")
        if validation_uri not in (None, "") or validation_hash not in (None, ""):
            if (
                not isinstance(validation_uri, str)
                or not validation_uri.startswith(_LOCAL_ARTIFACT_PREFIX)
                or not validation_uri.endswith(".jsonl")
            ):
                blockers.append("validation_artifact_uri")
            if not _SHA256.fullmatch(str(validation_hash or "")):
                blockers.append("validation_content_sha256")

        role = str(record.get("role") or "").lower()
        split = str(record.get("split") or "").lower()
        held_out = role in _HELD_OUT_ROLES or split in {"test", "evaluation", "eval"}
        if held_out:
            blockers.append("held_out")

        approval_requested = record.get("approval_status") == "Approved"
        review_blockers = [item for item in blockers if item != "held_out"]
        record["approval_status"] = "Approved" if approval_requested and not review_blockers else "Blocked"
        record["eligible_for_training"] = bool(
            record["approval_status"] == "Approved"
            and role in _TRAINING_ROLES
            and split not in {"test", "evaluation", "eval"}
        )
        record["blockers"] = sorted(set(blockers))
        record["schema_version"] = "blackhole.dataset.v1"

        identity = "|".join(
            str(record.get(key) or "") for key in ("dataset_id", "revision", "role", "split")
        )
        record["record_id"] = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        digest_payload = {key: value for key, value in record.items() if key != "registry_digest"}
        record["registry_digest"] = hashlib.sha256(
            json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        filename = record["record_id"] + ".json"
        atomic_write_json(self.root / filename, record)
        return record

    def verify_persisted(self, supplied: Mapping[str, Any]) -> dict[str, Any]:
        record_id = supplied.get("record_id")
        if not isinstance(record_id, str) or not re.fullmatch(r"[0-9a-f]{20}", record_id):
            raise RuntimeError("DATASET_NOT_APPROVED: persisted record identity is required")
        try:
            persisted = json.loads((self.root / f"{record_id}.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("DATASET_NOT_APPROVED: persisted registry record is unavailable") from exc
        if persisted != dict(supplied):
            raise RuntimeError("DATASET_NOT_APPROVED: supplied manifest differs from registry")
        expected = persisted.get("registry_digest")
        content = {key: value for key, value in persisted.items() if key != "registry_digest"}
        actual = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        role, split = str(persisted.get("role", "")).lower(), str(persisted.get("split", "")).lower()
        if (
            expected != actual
            or persisted.get("approval_status") != "Approved"
            or persisted.get("eligible_for_training") is not True
            or persisted.get("license_review") != "approved"
            or persisted.get("quality_review") != "approved"
            or str(persisted.get("license", "")).strip().lower() in _AMBIGUOUS_LICENSES
            or role not in _TRAINING_ROLES
            or role in _HELD_OUT_ROLES
            or split in {"test", "evaluation", "eval"}
            or not _SHA256.fullmatch(str(persisted.get("content_sha256") or ""))
        ):
            raise RuntimeError("DATASET_NOT_APPROVED: persisted record is not training eligible")
        return persisted

    def list_records(self) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(self.root.glob("*.json"))]
