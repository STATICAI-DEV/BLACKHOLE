from pathlib import Path


def test_empty_registry_reports_no_model() -> None:
    from blackhole.adapters.registry import AdapterRegistry

    registry = AdapterRegistry()
    assert registry.names() == []
    assert registry.status_code == "NO_MODEL"


def test_chat_without_trained_checkpoint_is_truthfully_blocked(tmp_path: Path) -> None:
    import pytest

    from blackhole.adapters.registry import AdapterRegistry
    from blackhole.chat import ChatService, ChatUnavailable

    service = ChatService(AdapterRegistry())

    with pytest.raises(ChatUnavailable, match="NO TRAINED CHECKPOINT — Chat unavailable"):
        service.respond(tmp_path / "missing", "hello", temperature=0.7, top_k=20, max_new_tokens=32)
