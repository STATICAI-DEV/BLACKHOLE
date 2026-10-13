import math
from pathlib import Path

import pytest


def _batch():
    return [
        {
            "messages": [
                {"role": "user", "content": "What is a small test?"},
                {"role": "assistant", "content": "A bounded check of one behavior."},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Why validate data?"},
                {"role": "assistant", "content": "To reject malformed inputs before training."},
            ]
        },
    ]


def test_static_0t_architecture_matches_supplied_asset_contract() -> None:
    from blackhole.adapters.static_0t import (
        StaticConfig,
        StaticLM,
        model_parameters,
        ternary_parameters,
    )

    model = StaticLM(StaticConfig())

    assert model_parameters(model) == 1_909_248
    assert ternary_parameters(model) == 1_867_776
    assert model.cfg.context == 256
    assert model.cfg.vocab_size == 256


def test_static_0t_causal_forward_and_ternary_codes() -> None:
    import torch

    from blackhole.adapters.static_0t import (
        StaticConfig,
        StaticLM,
        TernaryConv1d,
        TernaryLinear,
        ternary_codes_and_scale,
    )

    torch.manual_seed(1742)
    model = StaticLM(StaticConfig(width=16, layers=2, context=32)).eval()
    tokens = torch.randint(0, 256, (2, 24))
    with torch.no_grad():
        output = model(tokens)
        prefix = model(tokens[:, :13])

    assert output.shape == (2, 24, 256)
    assert torch.allclose(output[:, :13], prefix, atol=1e-5)
    for layer in model.modules():
        if isinstance(layer, (TernaryConv1d, TernaryLinear)):
            codes, _ = ternary_codes_and_scale(layer.weight)
            assert torch.isin(codes, torch.tensor([-1, 0, 1], dtype=torch.int8)).all()


def test_adapter_trains_evaluates_and_saves_resumable_artifacts(tmp_path: Path) -> None:
    import torch

    from blackhole.adapters.static_0t import Static0TAdapter, StaticConfig

    adapter = Static0TAdapter(
        device="cpu",
        config=StaticConfig(width=16, layers=1, context=32),
        seed=7,
    )
    before = adapter.model.embedding.weight.detach().clone()
    trained = adapter.train_step(_batch())
    evaluated = adapter.evaluate(_batch())
    saved = adapter.save(tmp_path / "checkpoint")

    assert math.isfinite(trained["train_loss"])
    assert math.isfinite(evaluated["val_loss"])
    assert not torch.equal(before, adapter.model.embedding.weight.detach())
    assert set(saved["files"]) == {"model", "optimizer", "scheduler", "random_state", "packed"}
    assert all((tmp_path / "checkpoint" / name).is_file() for name in saved["files"].values())

    restored = Static0TAdapter(
        device="cpu",
        config=StaticConfig(width=16, layers=1, context=32),
        seed=99,
    )
    torch.manual_seed(999)
    restored.load(tmp_path / "checkpoint")
    assert torch.equal(adapter.model.embedding.weight, restored.model.embedding.weight)
    resumed_random = torch.rand(4)
    saved_state = torch.load(
        tmp_path / "checkpoint" / "random_state.pt",
        map_location="cpu",
        weights_only=True,
    )["torch"]
    torch.set_rng_state(saved_state)
    assert torch.equal(resumed_random, torch.rand(4))


def test_adapter_rejects_malformed_conversation_rows() -> None:
    from blackhole.adapters.static_0t import Static0TAdapter, StaticConfig

    adapter = Static0TAdapter(device="cpu", config=StaticConfig(width=8, layers=1, context=16))
    with pytest.raises(ValueError, match="user and assistant"):
        adapter.train_step([{"messages": [{"role": "assistant", "content": "wrong"}]}])


def test_default_registry_exposes_audited_static_0t_adapter() -> None:
    from blackhole.adapters.registry import build_default_registry

    registry = build_default_registry(device="cpu")
    assert registry.names() == ["static-0t"]
    assert registry.get("static-0t").inspect()["source_archive_sha256"] == (
        "fbc24587333b2f0303dd7e8ee6ada0fb7fb64760e732b34e5035fcbf5bb9e331"
    )


def test_adapter_reconfigures_from_the_claimed_training_contract() -> None:
    import torch

    from blackhole.adapters.static_0t import Static0TAdapter, StaticConfig

    config = StaticConfig(width=8, layers=1, context=16)
    first = Static0TAdapter(device="cpu", config=config, seed=1, learning_rate=0.1)
    first.configure_for_training({"seed": 42, "learning_rate": 0.002, "context_length": 16})
    second = Static0TAdapter(device="cpu", config=config, seed=42, learning_rate=0.002)

    assert first.inspect()["seed"] == 42
    assert first.inspect()["learning_rate"] == 0.002
    assert first.optimizer.param_groups[0]["lr"] == 0.002
    assert torch.equal(first.model.embedding.weight, second.model.embedding.weight)

    with pytest.raises(ValueError, match="context_length"):
        first.configure_for_training({"seed": 42, "learning_rate": 0.002, "context_length": 15})
