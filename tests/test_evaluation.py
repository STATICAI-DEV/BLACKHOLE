from pathlib import Path


def _evaluation(checkpoint_hash: str, suffix: str) -> dict[str, object]:
    return {
        "prompt_set_version": "english-heldout-v1",
        "checkpoint_sha256": checkpoint_hash,
        "generation_parameters": {"temperature": 0.2, "seed": 7},
        "results": [
            {
                "prompt_id": "vocab-001",
                "category": "basic_vocabulary",
                "prompt": "Name a common fruit.",
                "output": f"apple{suffix}",
                "duration_seconds": 0.1,
            },
            {
                "prompt_id": "format-001",
                "category": "format_compliance",
                "prompt": "Return one word.",
                "output": f"done{suffix}",
                "duration_seconds": 0.1,
            },
        ],
    }


def test_evaluation_round_trip_and_same_prompt_comparison(tmp_path: Path) -> None:
    from blackhole.evaluation import EvaluationStore

    store = EvaluationStore(tmp_path)
    first = store.save("eval-a", _evaluation("a" * 64, "-a"))
    second = store.save("eval-b", _evaluation("b" * 64, "-b"))
    comparison = store.compare("eval-a", "eval-b")

    assert first["evaluation_id"] == "eval-a"
    assert second["evaluation_id"] == "eval-b"
    assert [pair["prompt_id"] for pair in comparison["pairs"]] == ["vocab-001", "format-001"]
    assert comparison["same_prompt_set"] is True


def test_comparison_rejects_different_prompt_ids(tmp_path: Path) -> None:
    import pytest

    from blackhole.evaluation import EvaluationStore

    store = EvaluationStore(tmp_path)
    store.save("eval-a", _evaluation("a" * 64, "-a"))
    changed = _evaluation("b" * 64, "-b")
    changed["results"][0]["prompt_id"] = "different"  # type: ignore[index]
    store.save("eval-b", changed)

    with pytest.raises(ValueError, match="same prompts"):
        store.compare("eval-a", "eval-b")

