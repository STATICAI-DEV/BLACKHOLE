from pathlib import Path


def test_metrics_reload_from_persisted_jsonl_and_build_real_series(tmp_path: Path) -> None:
    from blackhole.metrics import MetricsStore

    store = MetricsStore(tmp_path)
    store.append("run-001", {"step": 1, "train_loss": 4.2, "bytes_per_second": 120.0})
    store.append("run-001", {"step": 2, "train_loss": 3.8, "val_loss": 4.0})

    restarted = MetricsStore(tmp_path)
    assert [item["step"] for item in restarted.load("run-001")] == [1, 2]
    assert restarted.series("run-001", "train_loss") == {"step": [1, 2], "value": [4.2, 3.8]}
    assert restarted.series("run-001", "val_loss") == {"step": [2], "value": [4.0]}


def test_metrics_reject_non_monotonic_steps(tmp_path: Path) -> None:
    import pytest

    from blackhole.metrics import MetricsStore

    store = MetricsStore(tmp_path)
    store.append("run-001", {"step": 4, "train_loss": 2.0})
    with pytest.raises(ValueError, match="strictly increase"):
        store.append("run-001", {"step": 4, "train_loss": 1.9})
