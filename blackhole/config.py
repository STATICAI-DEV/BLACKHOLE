"""Environment configuration and production mount validation."""

from __future__ import annotations

import os
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .types import StorageHealth

_ENVIRONMENTS = {"production", "development", "test"}
_VOLUMES = {
    "/data": "STATICAI/BLACKHOLE/DATA",
    "/lab": "STATICAI/BLACKHOLE/LAB",
}


class BlockedStorageError(RuntimeError):
    """Raised when required durable storage is absent or unwritable."""


def _boolean(env: Mapping[str, str], key: str, default: bool = False) -> bool:
    raw = env.get(key)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{key} must be an explicit boolean")


@dataclass(frozen=True, slots=True)
class LabSettings:
    environment: str
    data_root: Path
    lab_root: Path
    auto_train: bool = False
    auto_resume: bool = False

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "LabSettings":
        values = os.environ if env is None else env
        environment = values.get("BLACKHOLE_ENV", "production").strip().lower()
        if environment not in _ENVIRONMENTS:
            raise ValueError("BLACKHOLE_ENV must be production, development, or test")
        production = environment == "production"
        data_default = "/data" if production else ".blackhole-data"
        lab_default = "/lab" if production else ".blackhole-lab"
        return cls(
            environment=environment,
            data_root=Path(values.get("BLACKHOLE_DATA_ROOT", data_default)),
            lab_root=Path(values.get("BLACKHOLE_LAB_ROOT", lab_default)),
            auto_train=_boolean(values, "BLACKHOLE_AUTO_TRAIN"),
            auto_resume=_boolean(values, "BLACKHOLE_AUTO_RESUME"),
        )

    @property
    def production(self) -> bool:
        return self.environment == "production"

    def ensure_storage(self) -> StorageHealth:
        roots = (self.data_root, self.lab_root)
        if self.production:
            missing = [str(root) for root in roots if not root.is_dir()]
            if missing:
                raise BlockedStorageError(
                    "BLOCKED_STORAGE: required mounted directories are missing: "
                    + ", ".join(missing)
                )
            for root, mount_path in zip(roots, ("/data", "/lab"), strict=True):
                marker = root / ".blackhole-volume.json"
                try:
                    payload = json.loads(marker.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise BlockedStorageError(f"BLOCKED_STORAGE: volume marker missing or unreadable: {marker}") from exc
                expected = {"schema_version": "blackhole.volume.v1", "mount_path": mount_path, "bucket": _VOLUMES[mount_path]}
                if payload != expected:
                    raise BlockedStorageError(f"BLOCKED_STORAGE: volume marker identity mismatch: {root}")
        else:
            for root in roots:
                root.mkdir(parents=True, exist_ok=True)

        for root in roots:
            self._verify_writable(root)
        return StorageHealth(True, "READY", "Dataset and lab storage are writable")

    @staticmethod
    def _verify_writable(root: Path) -> None:
        try:
            with tempfile.NamedTemporaryFile(dir=root, prefix=".blackhole-write-", delete=True):
                pass
        except OSError as exc:
            raise BlockedStorageError(
                f"BLOCKED_STORAGE: required mounted directory is not writable: {root}"
            ) from exc


def require_storage_root(root: Path | str, label: str) -> Path:
    path = Path(root)
    if not path.is_dir():
        raise BlockedStorageError(f"BLOCKED_STORAGE: {label} root is missing: {path}")
    return path
