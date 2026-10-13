"""Shared state types for the BLACKHOLE laboratory."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class OperationalState(StrEnum):
    READY = "READY"
    BLOCKED = "BLOCKED"
    UNKNOWN = "UNKNOWN"
    NOT_DEPLOYED = "NOT_DEPLOYED"


class LabMode(StrEnum):
    IDLE = "idle"
    TRAINING = "training"
    EVALUATING = "evaluating"
    INFERENCE = "inference"


@dataclass(frozen=True, slots=True)
class StorageHealth:
    ready: bool
    code: str
    detail: str

