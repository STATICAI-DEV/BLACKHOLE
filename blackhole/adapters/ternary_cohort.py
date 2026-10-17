"""Fresh STATIC-X, STATIC-T and STATIC-K foundation-model adapters."""

from .ternary_transformer import FoundationConfig, StaticFoundationAdapter


class StaticTernaryAdapter(StaticFoundationAdapter):
    """Independent decoder Transformer with ternary {-1, 0, +1} projection weights."""

    def __init__(self, *, name: str, device: str, config=None, seed=1234, learning_rate=3e-4):
        super().__init__(
            name=name,
            device=device,
            config=config or FoundationConfig(),
            seed=seed,
            learning_rate=learning_rate,
        )

    def spawn(self, *, seed: int, learning_rate: float):
        return type(self)(
            name=self.name,
            device=self.device,
            config=self.config,
            seed=seed,
            learning_rate=learning_rate,
        )
