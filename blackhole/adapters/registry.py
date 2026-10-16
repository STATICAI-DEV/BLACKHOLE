"""Explicit in-process registry for audited model adapters."""

from __future__ import annotations

from .base import ModelAdapter


class AdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, ModelAdapter] = {}

    def register(self, adapter: ModelAdapter) -> None:
        if not isinstance(adapter, ModelAdapter):
            raise TypeError("adapter does not implement ModelAdapter")
        if not adapter.name or adapter.name in self._adapters:
            raise ValueError("adapter name must be non-empty and unique")
        self._adapters[adapter.name] = adapter

    def names(self) -> list[str]:
        return sorted(self._adapters)

    @property
    def status_code(self) -> str:
        return "READY" if self._adapters else "NO_MODEL"

    def get(self, name: str) -> ModelAdapter:
        try:
            return self._adapters[name]
        except KeyError as exc:
            raise KeyError(f"model adapter is not registered: {name}") from exc

    def spawn(self, name: str, *, seed: int, learning_rate: float) -> ModelAdapter:
        """Create an independent training instance from an audited adapter prototype."""
        prototype = self.get(name)
        factory = getattr(prototype, "spawn", None)
        if not callable(factory):
            raise RuntimeError(f"adapter does not support cohort spawning: {name}")
        spawned = factory(seed=seed, learning_rate=learning_rate)
        if not isinstance(spawned, ModelAdapter):
            raise TypeError("spawned adapter does not implement ModelAdapter")
        return spawned


def build_default_registry(*, device: str) -> AdapterRegistry:
    from .static_0t import Static0TAdapter
    from .ternary_cohort import StaticTernaryAdapter

    registry = AdapterRegistry()
    registry.register(Static0TAdapter(device=device))
    for name in ("STATIC-X", "STATIC-T", "STATIC-K"):
        registry.register(StaticTernaryAdapter(name=name, device=device))
    return registry
