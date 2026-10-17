from __future__ import annotations

import torch

from blackhole.adapters.ternary_transformer import FoundationConfig, StaticFoundationAdapter


def small_adapter(recipe: str) -> StaticFoundationAdapter:
    adapter = StaticFoundationAdapter(
        name="STATIC-T",
        device="cpu",
        config=FoundationConfig(
            vocab_size=32000,
            width=64,
            layers=2,
            heads=4,
            ff_hidden=128,
            context=128,
            gradient_checkpointing=False,
        ),
        seed=11,
        learning_rate=3e-4,
    )
    adapter.configure_for_training({
        "seed": 11,
        "learning_rate": 3e-4,
        "train_steps": 4,
        "context_length": 128,
        "training_recipe_version": recipe,
        "weight_decay": 0.1,
        "warmup_steps": 1,
        "min_lr_ratio": 0.1,
    })
    return adapter


def test_foundation_causal_pretrain_step() -> None:
    adapter = small_adapter("causal-pretrain-v1")
    result = adapter.train_step([{
        "text": "English language models learn by predicting the next token in useful text."
    }])
    assert torch.isfinite(torch.tensor(result["train_loss"]))
    assert result["train_loss"] < 20.0


def test_foundation_instruction_masks_user_tokens() -> None:
    adapter = small_adapter("instruction-sft-v1")
    batch = [{
        "messages": [
            {"role": "user", "content": "What is two plus two?"},
            {"role": "assistant", "content": "Four."},
        ]
    }]
    _, targets = adapter._tensors(batch, training=False)
    assert int((targets != -100).sum()) > 0
    assert int((targets == -100).sum()) > 0


def test_foundation_reasoning_supervises_conversation() -> None:
    adapter = small_adapter("reasoning-mid-v1")
    result = adapter.train_step([{
        "messages": [
            {"role": "user", "content": "If x is 2, what is x plus 3?"},
            {"role": "assistant", "content": "x plus 3 equals 5."},
        ]
    }])
    assert torch.isfinite(torch.tensor(result["train_loss"]))


def test_foundation_model_profile() -> None:
    adapter = StaticFoundationAdapter(name="STATIC-X", device="cpu")
    inspection = adapter.inspect()
    assert inspection["parameters"] == 147_284_736
    assert inspection["config"]["context"] == 2048
    assert inspection["config"]["vocab_size"] == 32000
    assert inspection["variant"] == "STATIC-FOUNDATION-TERNARY-v1"


def test_foundation_software_engineering_sft_masks_user_tokens() -> None:
    adapter = small_adapter("software-engineering-sft-v1")
    batch = [{
        "messages": [
            {"role": "user", "content": "Fix the bug in this Python function."},
            {"role": "assistant", "content": "def fixed():\n    return True"},
        ]
    }]
    result = adapter.train_step(batch)
    assert torch.isfinite(torch.tensor(result["train_loss"]))


def test_foundation_judgment_sft_masks_user_tokens() -> None:
    adapter = small_adapter("judgment-sft-v1")
    batch = [{
        "messages": [
            {"role": "user", "content": "Compare response A and response B. Which is more correct?"},
            {"role": "assistant", "content": "Response A is better because it directly satisfies the requirement."},
        ]
    }]
    inputs, targets = adapter._tensors(batch, training=False)
    assert inputs.shape == targets.shape
    assert int((targets != -100).sum()) > 0
    assert int((targets == -100).sum()) > 0


def test_foundation_preference_dpo_uses_frozen_stage_reference(tmp_path) -> None:
    source = small_adapter("judgment-sft-v1")
    source.save(tmp_path)

    adapter = small_adapter("preference-dpo-v1")
    adapter.load(tmp_path, weights_only=True)
    assert adapter.reference_model is not None
    assert all(not parameter.requires_grad for parameter in adapter.reference_model.parameters())

    batch = [{
        "prompt": "Which answer is better for the task: write clear Python code?",
        "chosen": "Use a small function with a descriptive name and a direct return value.",
        "rejected": "Ignore the task and return unrelated text.",
    }]
    result = adapter.train_step(batch)
    assert torch.isfinite(torch.tensor(result["dpo_loss"]))
    assert "preference_margin" in result


def test_foundation_dpo_checkpoint_persists_reference_and_validation_contract(tmp_path) -> None:
    source_root = tmp_path / "source"
    dpo_root = tmp_path / "dpo"

    source = small_adapter("judgment-sft-v1")
    source.save(source_root)

    adapter = small_adapter("preference-dpo-v1")
    adapter.load(source_root, weights_only=True)
    batch = [{
        "prompt": "Choose the better software-engineering response.",
        "chosen": "Inspect the failing test, make the smallest correct patch, then rerun the test.",
        "rejected": "Rewrite unrelated files without checking the failure.",
    }]
    observed = adapter.train_step(batch)
    assert "train_loss" in observed and "dpo_loss" in observed
    saved = adapter.save(dpo_root)
    assert saved["files"]["reference_model"] == "reference_model.pt"

    resumed = small_adapter("preference-dpo-v1")
    resumed.load(dpo_root)
    assert resumed.reference_model is not None
    validation = resumed.evaluate(batch)
    assert set(validation) >= {"val_loss", "dpo_loss", "preference_margin"}
    assert torch.isfinite(torch.tensor(validation["val_loss"]))
