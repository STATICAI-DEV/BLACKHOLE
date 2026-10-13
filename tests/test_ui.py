from pathlib import Path


def _settings(tmp_path: Path):
    from blackhole.config import LabSettings

    return LabSettings.from_env(
        {
            "BLACKHOLE_ENV": "test",
            "BLACKHOLE_DATA_ROOT": str(tmp_path / "data"),
            "BLACKHOLE_LAB_ROOT": str(tmp_path / "lab"),
        }
    )


def test_build_app_has_exactly_five_contract_tabs_and_truthful_empty_state(tmp_path: Path) -> None:
    import gradio as gr

    from blackhole.ui import build_app

    app = build_app(_settings(tmp_path))
    assert isinstance(app, gr.Blocks)
    config = app.get_config_file()
    tabs = [
        component["props"].get("label")
        for component in config["components"]
        if component["type"] in {"tab", "tabitem"}
    ]
    assert tabs == ["Overview", "Training", "Datasets", "Testing Arena", "Chat"]

    rendered_values = [str(component["props"].get("value", "")) for component in config["components"]]
    rendered = "\n".join(rendered_values)
    assert "Model adapter:** `READY`" in rendered
    assert "NO TRAINED CHECKPOINT — Chat unavailable" in rendered

    start_buttons = [
        component
        for component in config["components"]
        if component["type"] == "button" and component["props"].get("value") == "Start"
    ]
    assert len(start_buttons) == 1
    assert start_buttons[0]["props"]["interactive"] is False

    execution_targets = [
        component
        for component in config["components"]
        if component["type"] == "dropdown" and component["props"].get("label") == "Execution target"
    ]
    assert len(execution_targets) == 1
    assert execution_targets[0]["props"]["value"] == "LOCAL"
    assert [choice[1] for choice in execution_targets[0]["props"]["choices"]] == ["LOCAL", "CLOUD"]

    api_names = {dependency.get("api_name") for dependency in config["dependencies"]}
    assert {
        "local_dataset_register",
        "local_job_enqueue",
        "local_cohort_enqueue",
        "local_evaluation_enqueue",
        "local_worker_announce",
        "local_worker_claim",
        "local_worker_status",
        "local_worker_update",
    } <= api_names

    labels = {component["props"].get("label") for component in config["components"]}
    assert "DATA bucket JSONL artifact URI (required for LOCAL)" in labels
    assert "I separately authorize CLOUD GPU / cloud compute for this run" in labels
    assert "Cloud authorization phrase" in labels


def test_metric_figure_uses_only_persisted_points(tmp_path: Path) -> None:
    from blackhole.metrics import MetricsStore
    from blackhole.ui import metric_figure

    store = MetricsStore(tmp_path)
    store.append("run-001", {"step": 1, "train_loss": 4.5})
    store.append("run-001", {"step": 2, "train_loss": 4.0})

    figure = metric_figure(store, "run-001", "train_loss", "Training loss")
    assert list(figure.data[0].x) == [1, 2]
    assert list(figure.data[0].y) == [4.5, 4.0]
    assert "TEST-ONLY" not in figure.layout.title.text


def test_safe_error_redacts_credential_like_values() -> None:
    from blackhole.ui import safe_error

    rendered = safe_error(RuntimeError("request failed with hf_secretvalue123456"))
    assert "hf_secretvalue123456" not in rendered
    assert "[REDACTED]" in rendered


def test_chatbot_constructor_is_compatible_without_type_deprecation(tmp_path: Path) -> None:
    import warnings

    from blackhole.ui import build_app

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_app(_settings(tmp_path))

    assert not any("type" in str(item.message) and "chatbot" in str(item.message).lower() for item in caught)


def test_all_action_buttons_are_wired_to_callbacks(tmp_path: Path) -> None:
    from blackhole.ui import build_app

    config = build_app(_settings(tmp_path)).get_config_file()
    button_ids = {item["id"]: item["props"].get("value") for item in config["components"] if item["type"] == "button"}
    wired = {target[0] for dep in config["dependencies"] for target in dep.get("targets", []) if target[1] == "click"}
    for label in ("Start", "Queue synchronized T / X / K cohort", "Pause/Stop safely", "Resume from checkpoint", "Evaluate now", "Register reviewed manifest", "Run held-out evaluation", "Compare checkpoints", "Send", "Reset conversation"):
        matching = [component_id for component_id, value in button_ids.items() if value == label]
        assert matching and matching[0] in wired, f"missing click callback for {label}"


def test_dispatch_training_keeps_cloud_path_and_queues_local_explicitly() -> None:
    from blackhole.ui import dispatch_training

    calls: list[tuple[str, object]] = []

    class Controller:
        def start(self, config, dataset, *, owner_authorized):
            calls.append(("cloud", config))
            return {"status": "running"}

    class Queue:
        def enqueue(self, config, dataset, *, owner_authorized):
            calls.append(("local", config))
            return {"state": "QUEUED"}

    base = {
        "run_id": "run-001",
        "model_adapter": "static-0t",
        "seed": 7,
        "train_steps": 3,
        "batch_size": 1,
        "context_length": 256,
        "learning_rate": 0.0003,
        "checkpoint_interval": 10,
        "evaluation_interval": 10,
    }
    dataset = {"record_id": "a" * 20}

    import pytest

    with pytest.raises(PermissionError, match="CLOUD_GPU_NOT_AUTHORIZED"):
        dispatch_training(Controller(), Queue(), "CLOUD", base, dataset, owner_authorized=True)
    with pytest.raises(RuntimeError, match="CLOUD_EXECUTION_NOT_IMPLEMENTED"):
        dispatch_training(
            Controller(),
            Queue(),
            "CLOUD",
            base,
            dataset,
            owner_authorized=True,
            cloud_gpu_authorized=True,
            cloud_gpu_authorization_text="AUTHORIZE CLOUD GPU",
        )
    assert not any(kind == "cloud" for kind, _ in calls)
    assert dispatch_training(Controller(), Queue(), "LOCAL", base, dataset, owner_authorized=True) == {"state": "QUEUED"}
    assert calls[-1][0] == "local"
    assert calls[-1][1]["execution_target"] == "LOCAL"

    with pytest.raises(ValueError, match="execution target"):
        dispatch_training(Controller(), Queue(), "AUTO", base, dataset, owner_authorized=True)


def test_checkpoint_discovery_uses_run_checkpoint_directories(tmp_path: Path) -> None:
    import hashlib
    import json

    from blackhole.checkpoints import CheckpointStore
    from blackhole.ui import discover_checkpoint_roots

    settings = _settings(tmp_path)
    settings.ensure_storage()
    checkpoint = settings.lab_root / "runs" / "run-001" / "checkpoint-final"
    checkpoint.mkdir(parents=True)
    model = checkpoint / "model.pt"
    model.write_bytes(b"model")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    (checkpoint / "CHECKPOINT_MANIFEST.json").write_text(
        json.dumps(
            {
                "schema_version": "blackhole.checkpoint.v1",
                "checkpoint_id": "run-001-final",
                "run_id": "run-001",
                "step": 10,
                "model_adapter": "static-0t",
                "trained": True,
                "files": {"model": "model.pt"},
                "sha256": {"model": digest},
            }
        ),
        encoding="utf-8",
    )

    roots = discover_checkpoint_roots(settings, CheckpointStore())
    assert roots == [str(checkpoint)]


def test_dispatch_cohort_training_queues_one_local_cohort_job() -> None:
    from blackhole.ui import dispatch_cohort_training

    calls = []

    class Queue:
        def enqueue_cohort(self, config, dataset, *, owner_authorized):
            calls.append((config, dataset, owner_authorized))
            return {"state": "QUEUED", "job_id": "local-cohort"}

    config = {
        "cohort_id": "batch2",
        "members": [
            {"member_id": "T", "run_id": "batch2-t", "model_adapter": "static-0t", "seed": 1234},
            {"member_id": "X", "run_id": "batch2-x", "model_adapter": "static-0t", "seed": 2401},
            {"member_id": "K", "run_id": "batch2-k", "model_adapter": "static-0t", "seed": 2402},
        ],
        "train_steps": 500,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
    }
    result = dispatch_cohort_training(Queue(), config, {"record_id": "a" * 20}, owner_authorized=True)
    assert result["state"] == "QUEUED"
    assert len(calls) == 1
    queued = calls[0][0]
    assert queued["job_type"] == "COHORT_TRAIN"
    assert queued["execution_target"] == "LOCAL"
    assert queued["batch_size"] * queued["gradient_accumulation_steps"] == 8


def test_checkpoint_selector_state_refreshes_after_new_checkpoint(tmp_path: Path) -> None:
    import hashlib
    import json

    from blackhole.checkpoints import CheckpointStore
    from blackhole.ui import checkpoint_selector_state

    settings = _settings(tmp_path)
    settings.ensure_storage()
    store = CheckpointStore()

    empty = checkpoint_selector_state(settings, store, has_approved_dataset=True)
    assert empty == {"final": [], "recovery": [], "has_final": False, "can_resume": False}

    checkpoint = settings.lab_root / "runs" / "fresh-run" / "checkpoint-final"
    checkpoint.mkdir(parents=True)
    files = {}
    hashes = {}
    for role in ("model", "optimizer", "scheduler", "random_state"):
        target = checkpoint / f"{role}.pt"
        target.write_bytes(role.encode("utf-8"))
        files[role] = target.name
        hashes[role] = hashlib.sha256(target.read_bytes()).hexdigest()
    (checkpoint / "CHECKPOINT_MANIFEST.json").write_text(
        json.dumps(
            {
                "schema_version": "blackhole.checkpoint.v1",
                "checkpoint_id": "fresh-run-final",
                "run_id": "fresh-run",
                "step": 10,
                "dataset_position": 20,
                "model_adapter": "static-0t",
                "trained": True,
                "files": files,
                "sha256": hashes,
            }
        ),
        encoding="utf-8",
    )

    refreshed = checkpoint_selector_state(settings, store, has_approved_dataset=True)
    assert refreshed["final"] == [str(checkpoint)]
    assert refreshed["has_final"] is True


def test_build_app_wires_periodic_checkpoint_refresh(tmp_path: Path) -> None:
    from blackhole.ui import build_app

    config = build_app(_settings(tmp_path)).get_config_file()
    timers = [component for component in config["components"] if component["type"] == "timer"]
    assert len(timers) == 1
    timer_id = timers[0]["id"]
    tick_dependencies = [
        dependency
        for dependency in config["dependencies"]
        if (timer_id, "tick") in [tuple(target) for target in dependency.get("targets", [])]
    ]
    assert len(tick_dependencies) == 1
    assert len(tick_dependencies[0].get("outputs", [])) == 14
