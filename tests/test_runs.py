import json
from pathlib import Path


def test_run_store_creates_required_records_and_truthful_summary(tmp_path: Path) -> None:
    from blackhole.runs import RunStore

    store = RunStore(tmp_path)
    created = store.create(
        "run-001",
        {"model_adapter": "fake", "seed": 7, "train_steps": 3},
        {"dataset_id": "example/data", "content_sha256": "a" * 64},
    )
    summary = store.finalize(
        "run-001",
        result="stopped",
        actual_metrics={"last_step": 0},
        failures=[],
    )

    assert created["status"] == "created"
    assert json.loads((tmp_path / "runs/run-001/RUN_CONFIG.json").read_text())["seed"] == 7
    assert (tmp_path / "runs/run-001/DATA_MANIFEST.json").is_file()
    manifest = json.loads((tmp_path / "runs/run-001/RUN_MANIFEST.json").read_text())
    assert manifest["result"] == "stopped"
    assert manifest["actual_metrics"] == {"last_step": 0}
    assert "What was not proven" in summary.read_text(encoding="utf-8")


def test_run_store_rejects_overwriting_an_existing_run(tmp_path: Path) -> None:
    import pytest

    from blackhole.runs import RunStore

    store = RunStore(tmp_path)
    store.create("run-001", {"seed": 1}, {"dataset_id": "example/data"})
    with pytest.raises(FileExistsError, match="run already exists"):
        store.create("run-001", {"seed": 2}, {"dataset_id": "example/data"})

