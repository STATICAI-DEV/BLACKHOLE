"""Inference chat guardrails; chat never falls back to another model."""

from __future__ import annotations

from pathlib import Path

from .adapters.registry import AdapterRegistry
from .checkpoints import CheckpointStore

CHAT_UNAVAILABLE = "NO TRAINED CHECKPOINT — Chat unavailable"


class ChatUnavailable(RuntimeError):
    pass


class ChatService:
    def __init__(self, adapters: AdapterRegistry, checkpoints: CheckpointStore | None = None) -> None:
        self.adapters = adapters
        self.checkpoints = checkpoints or CheckpointStore()

    def respond(
        self,
        checkpoint_root: Path | str,
        message: str,
        *,
        temperature: float,
        top_k: int,
        max_new_tokens: int,
    ) -> str:
        if not message.strip():
            raise ValueError("message must not be empty")
        inspection = self.checkpoints.inspect(checkpoint_root)
        if not inspection.chat_ready or inspection.manifest is None:
            raise ChatUnavailable(CHAT_UNAVAILABLE)
        adapter_name = inspection.manifest["model_adapter"]
        try:
            adapter = self.adapters.get(adapter_name)
        except KeyError as exc:
            raise ChatUnavailable(CHAT_UNAVAILABLE) from exc
        adapter.load(Path(checkpoint_root))
        return adapter.generate(
            message,
            temperature=temperature,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
        )
