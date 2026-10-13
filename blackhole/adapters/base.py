"""Small interface implemented by owner-reviewed model adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable


@runtime_checkable
class ModelAdapter(Protocol):
    name: str

    def inspect(self) -> Mapping[str, Any]: ...

    def train_step(self, batch: Any) -> Mapping[str, float]: ...

    def evaluate(self, batch: Any) -> Mapping[str, float]: ...

    def save(self, path: Path) -> Mapping[str, Any]: ...

    def load(self, path: Path) -> None: ...

    def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        top_k: int,
        max_new_tokens: int,
    ) -> str: ...

