"""Explicit in-process registry for audited model adapters."""

from __future__ import annotations

from collections.abc import Callable

from .base import ModelAdapter


class AdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, ModelAdapter] = {}
        self._factories: dict[str, Callable[[], ModelAdapter]] = {}

    def register(self, adapter: ModelAdapter) -> None:
        if not isinstance(adapter, ModelAdapter):
            raise TypeError("adapter does not implement ModelAdapter")
        if not adapter.name or adapter.name in self._adapters or adapter.name in self._factories:
            raise ValueError("adapter name must be non-empty and unique")
        self._adapters[adapter.name] = adapter

    def register_factory(self, name: str, factory: Callable[[], ModelAdapter]) -> None:
        if not isinstance(name, str) or not name or name in self._adapters or name in self._factories:
            raise ValueError("adapter name must be non-empty and unique")
        if not callable(factory):
            raise TypeError("adapter factory must be callable")
        self._factories[name] = factory

    def names(self) -> list[str]:
        return sorted(set(self._adapters) | set(self._factories))

    @property
    def status_code(self) -> str:
        return "READY" if self.names() else "NO_MODEL"

    def get(self, name: str) -> ModelAdapter:
        adapter = self._adapters.get(name)
        if adapter is not None:
            return adapter
        factory = self._factories.get(name)
        if factory is None:
            raise KeyError(f"model adapter is not registered: {name}")
        created = factory()
        if not isinstance(created, ModelAdapter) or created.name != name:
            raise TypeError(f"adapter factory returned invalid adapter for {name}")
        self._adapters[name] = created
        return created

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
    registry.register_factory("static-0t", lambda: Static0TAdapter(device=device))
    for name in ("STATIC-X", "STATIC-T", "STATIC-K"):
        registry.register_factory(
            name,
            lambda model_name=name: StaticTernaryAdapter(name=model_name, device=device),
        )
    return registry
