"""Independent STATIC-X, STATIC-T, STATIC-K ternary adapters."""

from .static_0t import Static0TAdapter


class StaticTernaryAdapter(Static0TAdapter):
    """Independent seeded model with ternary {-1, 0, +1} projection weights."""

    def __init__(self, *, name: str, device: str, config=None, seed=1234, learning_rate=0.0005):
        if name not in {"STATIC-X", "STATIC-T", "STATIC-K"}:
            raise ValueError("unknown ternary model")
        self.name = name
        super().__init__(device=device, config=config, seed=seed, learning_rate=learning_rate)

    def spawn(self, *, seed: int, learning_rate: float):
        return type(self)(
            name=self.name, device=self.device, config=self.config,
            seed=seed, learning_rate=learning_rate,
        )
