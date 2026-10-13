"""Checkpoint manifest and artifact integrity guards."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .atomic import sha256_file

_INFERENCE_FILES = {"model"}
_RESUME_FILES = {"model", "optimizer", "scheduler", "random_state"}
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class CheckpointInspection:
    valid: bool
    chat_ready: bool
    resume_ready: bool
    code: str
    manifest: dict[str, Any] | None
    errors: tuple[str, ...]


class CheckpointStore:
    def inspect(
        self,
        checkpoint_root: Path | str,
        *,
        expected_adapter: str | None = None,
    ) -> CheckpointInspection:
        root = Path(checkpoint_root)
        manifest_path = root / "CHECKPOINT_MANIFEST.json"
        if not manifest_path.is_file():
            return CheckpointInspection(False, False, False, "NO_CHECKPOINT", None, ("manifest missing",))
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return CheckpointInspection(False, False, False, "CHECKPOINT_MISMATCH", None, ("manifest unreadable",))

        core_errors: list[str] = []
        resume_errors: list[str] = []
        if manifest.get("schema_version") != "blackhole.checkpoint.v1":
            core_errors.append("unsupported checkpoint schema")
        for key in ("checkpoint_id", "run_id"):
            if not isinstance(manifest.get(key), str) or not _ID.fullmatch(manifest[key]):
                core_errors.append(f"{key} missing or invalid")
        if not isinstance(manifest.get("step"), int) or isinstance(manifest.get("step"), bool) or manifest.get("step", -1) < 0:
            core_errors.append("step missing or invalid")
        adapter = manifest.get("model_adapter")
        if not isinstance(adapter, str) or not adapter:
            core_errors.append("model adapter missing")
        if expected_adapter is not None and adapter != expected_adapter:
            core_errors.append("adapter mismatch")

        files = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
        hashes = manifest.get("sha256") if isinstance(manifest.get("sha256"), dict) else {}
        for key in _RESUME_FILES:
            filename = files.get(key)
            expected_hash = hashes.get(key)
            errors = core_errors if key in _INFERENCE_FILES else resume_errors
            target = None
            if isinstance(filename, str):
                candidate = Path(filename)
                try:
                    resolved_root = root.resolve(strict=True)
                    lexical_target = root / candidate
                    resolved_target = lexical_target.resolve(strict=False)
                    resolved_target.relative_to(resolved_root)
                    relative_parts = lexical_target.relative_to(root).parts
                    symlinked = any((root.joinpath(*relative_parts[:index])).is_symlink() for index in range(1, len(relative_parts) + 1))
                    if candidate.is_absolute() or ".." in candidate.parts or symlinked:
                        raise ValueError
                    target = resolved_target
                except (OSError, ValueError):
                    errors.append(f"{key} unsafe file path")
            if target is None or not target.is_file():
                if not any(message == f"{key} unsafe file path" for message in errors):
                    errors.append(f"{key} file missing")
                continue
            if not isinstance(expected_hash, str) or not _HASH.fullmatch(expected_hash) or sha256_file(target) != expected_hash:
                errors.append(f"{key} hash mismatch")

        trained = manifest.get("trained") is True
        if not trained:
            core_errors.append("checkpoint is not trained")
        chat_ready = trained and not core_errors
        resume_ready = chat_ready and not resume_errors
        all_errors = tuple(core_errors + resume_errors)
        if core_errors:
            code = "CHECKPOINT_MISMATCH"
        elif resume_errors:
            code = "RESUME_UNVERIFIED"
        else:
            code = "READY"
        return CheckpointInspection(not all_errors, chat_ready, resume_ready, code, manifest, all_errors)
