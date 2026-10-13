"""Exactly-five-tab Gradio user interface for the BLACKHOLE lab."""

from __future__ import annotations

import inspect
import json
import uuid
from pathlib import Path
from typing import Any, Mapping

import gradio as gr
import plotly.graph_objects as go

from .adapters.registry import AdapterRegistry, build_default_registry
from .chat import CHAT_UNAVAILABLE
from .chat import ChatService
from .config import LabSettings
from .datasets import DatasetRegistry
from .evaluation import EvaluationStore
from .local_jobs import LocalJobQueue, sanitize_remote_value
from .metrics import MetricsStore
from .redaction import redact_text
from .system import inspect_system
from .training import TrainingController


CLOUD_GPU_AUTHORIZATION_PHRASE = "AUTHORIZE CLOUD GPU"


def dispatch_training(
    controller: TrainingController,
    local_jobs: LocalJobQueue,
    execution_target: str,
    config: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    *,
    owner_authorized: bool,
    cloud_gpu_authorized: bool = False,
    cloud_gpu_authorization_text: str = "",
) -> dict[str, Any]:
    """Route explicit LOCAL/CLOUD execution with a fail-closed cloud GPU gate."""
    base = {
        "run_id": config.get("run_id"),
        "model_adapter": config.get("model_adapter"),
        "seed": config.get("seed"),
        "train_steps": config.get("train_steps"),
    }
    if execution_target == "CLOUD":
        if not owner_authorized:
            raise PermissionError("TRAINING_NOT_AUTHORIZED: explicit owner authorization is required")
        if not cloud_gpu_authorized or cloud_gpu_authorization_text.strip() != CLOUD_GPU_AUTHORIZATION_PHRASE:
            raise PermissionError(
                "CLOUD_GPU_NOT_AUTHORIZED: cloud execution requires separate explicit user authorization "
                f"and the exact phrase {CLOUD_GPU_AUTHORIZATION_PHRASE!r}"
            )
        raise RuntimeError(
            "CLOUD_EXECUTION_NOT_IMPLEMENTED: cloud GPU authorization was supplied, "
            "but BLACKHOLE has no cloud training worker; no run was started"
        )
    if execution_target == "LOCAL":
        local_config = {**dict(config), "execution_target": "LOCAL"}
        return local_jobs.enqueue(local_config, dataset_manifest, owner_authorized=owner_authorized)
    raise ValueError("execution target must be CLOUD or LOCAL")


def dispatch_cohort_training(
    local_jobs: LocalJobQueue,
    config: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    *,
    owner_authorized: bool,
) -> dict[str, Any]:
    """Queue one synchronized T/X/K-style LOCAL cohort job."""
    return local_jobs.enqueue_cohort(
        {**dict(config), "execution_target": "LOCAL", "job_type": "COHORT_TRAIN"},
        dataset_manifest,
        owner_authorized=owner_authorized,
    )


def _worker_api(call) -> dict[str, Any]:
    try:
        return call()
    except Exception as exc:
        return {"status": "ERROR", "error": sanitize_remote_value(safe_error(exc))}

def safe_error(error: BaseException) -> str:
    return redact_text(error, limit=240)


def metric_figure(store: MetricsStore, run_id: str, field: str, title: str) -> go.Figure:
    figure = go.Figure()
    if run_id.strip():
        try:
            series = store.series(run_id.strip(), field)
        except (OSError, ValueError):
            series = {"step": [], "value": []}
        if series["step"]:
            figure.add_trace(
                go.Scatter(x=series["step"], y=series["value"], mode="lines+markers", name=field)
            )
    figure.update_layout(
        title=title,
        xaxis_title="Step",
        yaxis_title=field.replace("_", " ").title(),
        template="plotly_dark",
        height=280,
        margin={"l": 40, "r": 20, "t": 48, "b": 40},
    )
    return figure


def _overview_markdown(settings: LabSettings, adapters: AdapterRegistry) -> str:
    snapshot = inspect_system(settings)
    model_state = "READY" if adapters.names() else "NO_MODEL — BLOCKED_MODEL_ASSET"
    return (
        "# BLACKHOLE\n\n"
        "Private evidence-first language-model research laboratory.\n\n"
        f"- **Mode:** `idle`\n"
        f"- **Model adapter:** `{model_state}`\n"
        f"- **Storage:** `{snapshot.storage_code}` — {snapshot.storage_detail}\n"
        f"- **CPU:** {snapshot.cpu_percent:.1f}%\n"
        f"- **RAM:** {snapshot.ram_used_bytes / (1024**3):.2f} / "
        f"{snapshot.ram_total_bytes / (1024**3):.2f} GiB\n"
        f"- **Checkpoints found:** {snapshot.checkpoints_found}\n"
        f"- **Observed:** {snapshot.observed_at}\n\n"
        "No training is active. AUTO_TRAIN and AUTO_RESUME are off."
    )


def _dataset_rows(registry: DatasetRegistry) -> list[list[object]]:
    try:
        records = registry.list_records()
    except (OSError, ValueError):
        return []
    return [
        [
            record.get("dataset_id"),
            record.get("revision"),
            record.get("license"),
            record.get("role"),
            record.get("split"),
            record.get("language"),
            record.get("rows"),
            record.get("approval_status"),
            record.get("eligible_for_training"),
            record.get("content_sha256"),
        ]
        for record in records
    ]


def discover_checkpoint_roots(
    settings: LabSettings,
    checkpoints,
    *,
    resume_only: bool = False,
) -> list[str]:
    """Discover verified checkpoints where BLACKHOLE actually stores them."""
    runs_root = settings.lab_root / "runs"
    if not runs_root.is_dir():
        return []
    roots: list[str] = []
    patterns = ("checkpoint-step-*",) if resume_only else ("checkpoint-final",)
    for run_root in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        for pattern in patterns:
            for path in sorted(run_root.glob(pattern)):
                if not path.is_dir():
                    continue
                inspection = checkpoints.inspect(path)
                ready = inspection.resume_ready if resume_only else inspection.chat_ready
                if ready:
                    roots.append(str(path))
    return roots


def checkpoint_selector_state(
    settings: LabSettings,
    checkpoints,
    *,
    has_approved_dataset: bool,
) -> dict[str, Any]:
    """Return current verified checkpoint choices and control availability."""
    final = discover_checkpoint_roots(settings, checkpoints)
    recovery = discover_checkpoint_roots(settings, checkpoints, resume_only=True)
    return {
        "final": final,
        "recovery": recovery,
        "has_final": bool(final),
        "can_resume": bool(recovery and has_approved_dataset),
    }


def stop_active_training(controller: TrainingController, local_jobs: LocalJobQueue) -> dict[str, Any]:
    """Stop the actual active execution path, preferring LOCAL queue work."""
    try:
        return local_jobs.request_cancel()
    except RuntimeError as exc:
        if str(exc) != "NO_ACTIVE_LOCAL_JOB":
            raise
    try:
        return controller.request_stop()
    except RuntimeError as exc:
        if str(exc) == "NO_ACTIVE_TRAINING":
            return {"status": "idle", "code": "NO_ACTIVE_TRAINING"}
        raise


def build_app(
    settings: LabSettings | None = None,
    adapters: AdapterRegistry | None = None,
) -> gr.Blocks:
    settings = settings or LabSettings.from_env()
    adapters = adapters or build_default_registry(device="cpu")
    snapshot = inspect_system(settings)
    datasets = DatasetRegistry(settings.data_root)
    metrics = MetricsStore(settings.lab_root)
    controller = TrainingController(settings, adapters)
    local_jobs = LocalJobQueue(settings)
    evaluations = EvaluationStore(settings.lab_root)
    chat_service = ChatService(adapters)
    dataset_records = datasets.list_records() if settings.data_root.is_dir() else []
    approved = [record for record in dataset_records if record.get("eligible_for_training")]
    approved_by_id = {str(record.get("record_id")): record for record in approved}
    checkpoint_state = checkpoint_selector_state(
        settings,
        chat_service.checkpoints,
        has_approved_dataset=bool(approved),
    )
    checkpoint_roots = checkpoint_state["final"]
    resume_checkpoint_roots = checkpoint_state["recovery"]
    evaluation_ids = sorted(
        path.parent.name
        for path in (settings.lab_root / "evaluations").glob("*/EVALUATION_RESULTS.json")
        if path.is_file()
    ) if settings.lab_root.is_dir() else []
    can_train = snapshot.storage_ready and bool(adapters.names()) and bool(approved)
    empty_plot = lambda field, title: metric_figure(metrics, "", field, title)

    def refresh_checkpoint_controls():
        state = checkpoint_selector_state(
            settings,
            chat_service.checkpoints,
            has_approved_dataset=bool(approved),
        )
        final = state["final"]
        recovery = state["recovery"]
        final_enabled = state["has_final"]
        resume_enabled = state["can_resume"]
        return (
            gr.update(choices=recovery, interactive=bool(recovery)),
            gr.update(interactive=resume_enabled),
            gr.update(choices=final, interactive=final_enabled),
            gr.update(interactive=final_enabled),
            gr.update(choices=final, interactive=final_enabled),
            gr.update(choices=final, interactive=final_enabled),
            gr.update(interactive=final_enabled),
            gr.update(interactive=final_enabled),
            gr.update(choices=final, interactive=final_enabled),
            gr.update(interactive=final_enabled),
            gr.update(interactive=final_enabled),
            gr.update(interactive=final_enabled),
            gr.update(interactive=final_enabled),
            gr.update(interactive=final_enabled),
        )

    def queue_local_evaluation(checkpoint: str, authorized: bool) -> dict[str, Any]:
        inspection = chat_service.checkpoints.inspect(checkpoint)
        if not inspection.chat_ready or inspection.manifest is None:
            raise RuntimeError(CHAT_UNAVAILABLE)
        manifest = inspection.manifest
        return local_jobs.enqueue_evaluation(
            {
                "execution_target": "LOCAL",
                "job_type": "EVALUATION",
                "evaluation_id": f"eval-{manifest['run_id']}-{uuid.uuid4().hex[:8]}",
                "model_adapter": manifest["model_adapter"],
                "checkpoint_run_id": manifest["run_id"],
                "prompt_set_version": "english-heldout-v1",
                "seed": 1234,
                "temperature": 0.2,
                "top_k": 20,
                "max_new_tokens": 128,
            },
            owner_authorized=authorized,
        )

    def compare_checkpoint_evaluations(first: str, second: str) -> dict[str, Any]:
        def latest_for(checkpoint: str) -> str:
            inspection = chat_service.checkpoints.inspect(checkpoint)
            if not inspection.chat_ready or inspection.manifest is None:
                raise RuntimeError(CHAT_UNAVAILABLE)
            run_id = inspection.manifest["run_id"]
            matches: list[tuple[float, str]] = []
            for path in (settings.lab_root / "evaluations").glob("*/EVALUATION_RESULTS.json"):
                try:
                    record = evaluations.load(path.parent.name)
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
                if record.get("train_run_id") == run_id or str(record.get("checkpoint", "")).endswith(
                    f"/runs/{run_id}/checkpoint-final"
                ):
                    matches.append((path.stat().st_mtime, path.parent.name))
            if not matches:
                raise RuntimeError(f"NO_EVALUATION: {run_id}")
            return max(matches)[1]
        return evaluations.compare(latest_for(first), latest_for(second))

    with gr.Blocks(title="BLACKHOLE") as demo:
        with gr.Tab("Overview"):
            overview = gr.Markdown(_overview_markdown(settings, adapters))
            refresh_overview = gr.Button("Refresh overview")
            refresh_overview.click(lambda: _overview_markdown(settings, adapters), outputs=overview)

        with gr.Tab("Training"):
            gr.Markdown(
                "## Training control\n"
                "Controls remain disabled until durable storage, an audited adapter, and an approved training manifest exist."
            )
            with gr.Row():
                execution_target = gr.Dropdown(["LOCAL", "CLOUD"], value="LOCAL", label="Execution target")
                adapter_choice = gr.Dropdown(adapters.names(), label="Model adapter", interactive=bool(adapters.names()))
                dataset_choice = gr.Dropdown(
                    list(approved_by_id),
                    label="Approved dataset manifest",
                    interactive=bool(approved),
                )
            with gr.Row():
                seed = gr.Number(value=7, precision=0, label="Seed")
                batch_size = gr.Number(value=2, minimum=1, precision=0, label="Physical batch size")
                context_length = gr.Number(value=256, minimum=8, precision=0, label="Context length")
                learning_rate = gr.Number(value=0.0003, minimum=0, label="Learning rate")
            with gr.Row():
                train_steps = gr.Number(value=500, minimum=1, precision=0, label="Optimizer steps")
                checkpoint_interval = gr.Number(value=50, minimum=1, precision=0, label="Checkpoint interval")
                evaluation_interval = gr.Number(value=10, minimum=1, precision=0, label="Evaluation interval")
            run_name = gr.Textbox(label="New single-model run ID", value="run-001")
            owner_authorized = gr.Checkbox(label="I explicitly authorize this training run", value=False)

            gr.Markdown(
                "### Synchronized cohort training\n"
                "One LOCAL worker, one CUDA process, one deterministic shuffled dataset stream, "
                "three independent models. Physical batch × accumulation = effective batch."
            )
            with gr.Row():
                cohort_id = gr.Textbox(label="Cohort ID", value="batch2")
                gradient_accumulation_steps = gr.Number(
                    value=4, minimum=1, precision=0, label="Gradient accumulation"
                )
                shuffle_seed = gr.Number(value=1234, minimum=0, precision=0, label="Shuffle seed")
                dataset_position = gr.Number(
                    value=0,
                    minimum=0,
                    precision=0,
                    label="Dataset cursor (examples already consumed)",
                )
            with gr.Row():
                cohort_seed_t = gr.Number(value=1234, minimum=0, precision=0, label="T seed")
                cohort_seed_x = gr.Number(value=2401, minimum=0, precision=0, label="X seed")
                cohort_seed_k = gr.Number(value=2402, minimum=0, precision=0, label="K seed")
            cohort_button = gr.Button(
                "Queue synchronized T / X / K cohort",
                variant="primary",
                interactive=can_train,
            )
            gr.Markdown(
                "**Cloud GPU lock:** selecting CLOUD is blocked unless the user separately authorizes cloud compute "
                "and types the exact phrase `AUTHORIZE CLOUD GPU`. Local approval never implies cloud approval."
            )
            with gr.Row():
                cloud_gpu_authorized = gr.Checkbox(
                    label="I separately authorize CLOUD GPU / cloud compute for this run",
                    value=False,
                )
                cloud_gpu_authorization_text = gr.Textbox(
                    label="Cloud authorization phrase",
                    placeholder="AUTHORIZE CLOUD GPU",
                )
            with gr.Row():
                start_button = gr.Button("Start", variant="primary", interactive=can_train)
                stop_button = gr.Button("Pause/Stop safely", interactive=can_train)
                resume_button = gr.Button("Resume from checkpoint", interactive=bool(resume_checkpoint_roots and approved))
                evaluate_button = gr.Button("Evaluate now", interactive=bool(checkpoint_roots))
                refresh_training = gr.Button("Refresh")
            training_status = gr.JSON(value=controller.status(), label="Training action result")
            resume_checkpoint = gr.Dropdown(
                resume_checkpoint_roots,
                label="Recovery checkpoint to resume",
                interactive=bool(resume_checkpoint_roots),
            )
            evaluate_checkpoint_choice = gr.Dropdown(
                checkpoint_roots,
                label="Final checkpoint to evaluate",
                interactive=bool(checkpoint_roots),
            )
            start_button.click(
                lambda target, name, adapter, dataset_id, selected_seed, steps, selected_batch_size, selected_context, selected_learning_rate, selected_checkpoint_interval, selected_evaluation_interval, authorized, cloud_authorized, cloud_phrase: dispatch_training(
                    controller,
                    local_jobs,
                    str(target),
                    {
                        "run_id": name,
                        "model_adapter": adapter,
                        "seed": int(selected_seed),
                        "train_steps": int(steps),
                        "batch_size": int(selected_batch_size),
                        "context_length": int(selected_context),
                        "learning_rate": float(selected_learning_rate),
                        "checkpoint_interval": int(selected_checkpoint_interval),
                        "evaluation_interval": int(selected_evaluation_interval),
                    },
                    approved_by_id.get(str(dataset_id), {}),
                    owner_authorized=bool(authorized),
                    cloud_gpu_authorized=bool(cloud_authorized),
                    cloud_gpu_authorization_text=str(cloud_phrase or ""),
                ),
                inputs=[execution_target, run_name, adapter_choice, dataset_choice, seed, train_steps, batch_size, context_length, learning_rate, checkpoint_interval, evaluation_interval, owner_authorized, cloud_gpu_authorized, cloud_gpu_authorization_text], outputs=training_status,
            )
            cohort_button.click(
                lambda cohort, adapter, dataset_id, steps, physical_batch, accumulation, selected_context, selected_learning_rate, selected_checkpoint_interval, selected_evaluation_interval, selected_shuffle_seed, cursor, seed_t, seed_x, seed_k, authorized: dispatch_cohort_training(
                    local_jobs,
                    {
                        "cohort_id": str(cohort),
                        "members": [
                            {
                                "member_id": "T",
                                "run_id": f"{cohort}-t",
                                "model_adapter": str(adapter),
                                "seed": int(seed_t),
                            },
                            {
                                "member_id": "X",
                                "run_id": f"{cohort}-x",
                                "model_adapter": str(adapter),
                                "seed": int(seed_x),
                            },
                            {
                                "member_id": "K",
                                "run_id": f"{cohort}-k",
                                "model_adapter": str(adapter),
                                "seed": int(seed_k),
                            },
                        ],
                        "train_steps": int(steps),
                        "batch_size": int(physical_batch),
                        "gradient_accumulation_steps": int(accumulation),
                        "context_length": int(selected_context),
                        "learning_rate": float(selected_learning_rate),
                        "checkpoint_interval": int(selected_checkpoint_interval),
                        "evaluation_interval": int(selected_evaluation_interval),
                        "shuffle_seed": int(selected_shuffle_seed),
                        "dataset_position": int(cursor),
                    },
                    approved_by_id.get(str(dataset_id), {}),
                    owner_authorized=bool(authorized),
                ),
                inputs=[
                    cohort_id,
                    adapter_choice,
                    dataset_choice,
                    train_steps,
                    batch_size,
                    gradient_accumulation_steps,
                    context_length,
                    learning_rate,
                    checkpoint_interval,
                    evaluation_interval,
                    shuffle_seed,
                    dataset_position,
                    cohort_seed_t,
                    cohort_seed_x,
                    cohort_seed_k,
                    owner_authorized,
                ],
                outputs=training_status,
            )
            stop_button.click(
                lambda: stop_active_training(controller, local_jobs),
                outputs=training_status,
            )
            resume_button.click(
                lambda checkpoint, dataset_id, authorized: local_jobs.enqueue_resume(
                    checkpoint,
                    approved_by_id.get(str(dataset_id), {}),
                    owner_authorized=bool(authorized),
                ),
                inputs=[resume_checkpoint, dataset_choice, owner_authorized],
                outputs=training_status,
            )
            evaluate_button.click(
                lambda checkpoint, authorized: queue_local_evaluation(checkpoint, bool(authorized)),
                inputs=[evaluate_checkpoint_choice, owner_authorized],
                outputs=training_status,
            )
            refresh_training.click(controller.status, outputs=training_status)
            run_id = gr.Textbox(label="Run ID for persisted graphs", placeholder="No run selected")
            with gr.Row():
                loss_plot = gr.Plot(empty_plot("train_loss", "Training loss"))
                validation_plot = gr.Plot(empty_plot("val_loss", "Validation loss"))
            with gr.Row():
                throughput_plot = gr.Plot(empty_plot("bytes_per_second", "Throughput (bytes/s)"))
                step_time_plot = gr.Plot(empty_plot("seconds_per_step", "Time per step (seconds)"))
            refresh_graphs = gr.Button("Load persisted metrics")
            refresh_graphs.click(
                lambda selected: (
                    metric_figure(metrics, selected or "", "train_loss", "Training loss"),
                    metric_figure(metrics, selected or "", "val_loss", "Validation loss"),
                    metric_figure(metrics, selected or "", "bytes_per_second", "Throughput (bytes/s)"),
                    metric_figure(metrics, selected or "", "seconds_per_step", "Time per step (seconds)"),
                ),
                inputs=run_id,
                outputs=[loss_plot, validation_plot, throughput_plot, step_time_plot],
            )

        with gr.Tab("Datasets"):
            gr.Markdown(
                "## Dataset registry\n"
                "Registration stores reviewed metadata only. Missing or ambiguous licensing remains Blocked; no bulk import occurs here."
            )
            table = gr.Dataframe(
                headers=[
                    "Dataset",
                    "Revision",
                    "License",
                    "Role",
                    "Split",
                    "Language",
                    "Rows",
                    "Status",
                    "Train eligible",
                    "SHA-256",
                ],
                value=_dataset_rows(datasets),
                interactive=False,
            )
            with gr.Row():
                dataset_id_input = gr.Textbox(label="Hugging Face dataset ID")
                revision_input = gr.Textbox(label="Exact revision")
                gr.Textbox(label="Config")
                split_input = gr.Textbox(label="Split", value="train")
            artifact_uri_input = gr.Textbox(
                label="DATA bucket JSONL artifact URI (required for LOCAL)",
                placeholder="hf://buckets/BluStatic/BLACKHOLE/DATA/path/file.jsonl",
            )
            with gr.Row():
                license_input = gr.Textbox(label="Exact license")
                role_input = gr.Dropdown(["pretraining", "chat", "instruction", "evaluation", "benchmark"], value="chat", label="Role")
                language_input = gr.Textbox(value="en", label="Language")
                rows_input = gr.Number(value=0, minimum=0, precision=0, label="Rows")
            content_hash_input = gr.Textbox(label="Content SHA-256")
            with gr.Row():
                license_review = gr.Checkbox(label="License review approved")
                quality_review = gr.Checkbox(label="Quality review approved")
                approval_requested = gr.Checkbox(label="Approve manifest")
            register_button = gr.Button("Register reviewed manifest", interactive=True)
            registration_status = gr.JSON(label="Registration result")
            def register_manifest(dataset_id, revision, split, artifact_uri, license_value, role, language, rows, content_hash, license_ok, quality_ok, approved_request):
                settings.ensure_storage()
                return datasets.register({"dataset_id": dataset_id, "revision": revision, "split": split, "artifact_uri": artifact_uri, "license": license_value, "role": role, "language": language, "rows": int(rows), "content_sha256": content_hash, "license_review": "approved" if license_ok else "blocked", "quality_review": "approved" if quality_ok else "blocked", "approval_status": "Approved" if approved_request else "Blocked"})
            register_button.click(register_manifest, inputs=[dataset_id_input, revision_input, split_input, artifact_uri_input, license_input, role_input, language_input, rows_input, content_hash_input, license_review, quality_review, approval_requested], outputs=registration_status)
            refresh_datasets = gr.Button("Refresh dataset registry")
            refresh_datasets.click(lambda: _dataset_rows(datasets), outputs=table)

        with gr.Tab("Testing Arena"):
            gr.Markdown(
                "## Held-out testing arena\n"
                "Arena runs are queued for the authorized LOCAL CUDA worker. "
                "The Space CPU does not perform model generation."
            )
            with gr.Row():
                checkpoint_a = gr.Dropdown(checkpoint_roots, label="Checkpoint A", interactive=bool(checkpoint_roots))
                checkpoint_b = gr.Dropdown(checkpoint_roots, label="Checkpoint B", interactive=bool(checkpoint_roots))
            arena_authorized = gr.Checkbox(label="I explicitly authorize this evaluation", value=False)
            run_evaluation = gr.Button("Run held-out evaluation", interactive=bool(checkpoint_roots))
            compare_evaluations = gr.Button("Compare checkpoints", interactive=bool(checkpoint_roots))
            evaluation_result = gr.JSON(
                value={"status": "READY", "persisted_evaluations": evaluation_ids},
                label="Arena result",
            )
            run_evaluation.click(
                lambda checkpoint, authorized: queue_local_evaluation(checkpoint, bool(authorized)),
                inputs=[checkpoint_a, arena_authorized],
                outputs=evaluation_result,
            )
            compare_evaluations.click(
                compare_checkpoint_evaluations,
                inputs=[checkpoint_a, checkpoint_b],
                outputs=evaluation_result,
            )

        with gr.Tab("Chat"):
            chat_status = gr.Markdown(f"## {CHAT_UNAVAILABLE}\nNo cloud fallback or scripted response is enabled.")
            chatbot_options = {"type": "messages"} if "type" in inspect.signature(gr.Chatbot.__init__).parameters else {}
            chatbot = gr.Chatbot(label="Inference chat", value=[], **chatbot_options)
            chat_checkpoint = gr.Dropdown(checkpoint_roots, label="Trained checkpoint", interactive=bool(checkpoint_roots))
            message = gr.Textbox(label="Message", interactive=bool(checkpoint_roots))
            with gr.Row():
                temperature = gr.Slider(0.0, 2.0, value=0.7, label="Temperature", interactive=bool(checkpoint_roots))
                top_k = gr.Slider(1, 200, value=20, step=1, label="Top-k", interactive=bool(checkpoint_roots))
                max_tokens = gr.Slider(1, 1024, value=128, step=1, label="Max new tokens", interactive=bool(checkpoint_roots))
            with gr.Row():
                send_button = gr.Button("Send", interactive=bool(checkpoint_roots))
                reset_button = gr.Button("Reset conversation", interactive=True)
            def send_message(checkpoint, text, history, temp, selected_top_k, selected_max):
                response = chat_service.respond(checkpoint, text, temperature=float(temp), top_k=int(selected_top_k), max_new_tokens=int(selected_max))
                return list(history or []) + [{"role": "user", "content": text}, {"role": "assistant", "content": response}], ""
            send_button.click(send_message, inputs=[chat_checkpoint, message, chatbot, temperature, top_k, max_tokens], outputs=[chatbot, message])
            reset_button.click(lambda: ([], ""), outputs=[chatbot, message])
            message.submit(lambda: CHAT_UNAVAILABLE, outputs=chat_status)

        checkpoint_refresh_timer = gr.Timer(value=15.0, active=True)
        checkpoint_refresh_outputs = [
            resume_checkpoint,
            resume_button,
            evaluate_checkpoint_choice,
            evaluate_button,
            checkpoint_a,
            checkpoint_b,
            run_evaluation,
            compare_evaluations,
            chat_checkpoint,
            message,
            temperature,
            top_k,
            max_tokens,
            send_button,
        ]
        checkpoint_refresh_timer.tick(
            refresh_checkpoint_controls,
            outputs=checkpoint_refresh_outputs,
            show_progress="hidden",
        )
        refresh_training.click(
            refresh_checkpoint_controls,
            outputs=checkpoint_refresh_outputs,
            show_progress="hidden",
        )

        announce_payload = gr.JSON(visible=False)
        announce_result = gr.JSON(visible=False)
        announce_button = gr.Button("Local worker announce API", visible=False)
        announce_button.click(
            lambda payload: _worker_api(lambda: local_jobs.announce(payload if isinstance(payload, Mapping) else {})),
            inputs=announce_payload,
            outputs=announce_result,
            api_name="local_worker_announce",
        )
        claim_payload = gr.JSON(visible=False)
        claim_result = gr.JSON(visible=False)
        claim_button = gr.Button("Local worker claim API", visible=False)
        claim_button.click(
            lambda payload: _worker_api(
                lambda: local_jobs.claim(
                    str((payload or {}).get("worker_id") or ""),
                    str((payload or {}).get("job_id") or "") or None,
                )
            ),
            inputs=claim_payload,
            outputs=claim_result,
            api_name="local_worker_claim",
        )
        status_payload = gr.JSON(visible=False)
        status_result = gr.JSON(visible=False)
        status_button = gr.Button("Local worker status API", visible=False)
        status_button.click(
            lambda payload: _worker_api(
                lambda: local_jobs.status(str((payload or {}).get("job_id") or ""))
            ),
            inputs=status_payload,
            outputs=status_result,
            api_name="local_worker_status",
        )
        update_payload = gr.JSON(visible=False)
        update_result = gr.JSON(visible=False)
        update_button = gr.Button("Local worker update API", visible=False)
        update_button.click(
            lambda payload: _worker_api(
                lambda: local_jobs.update(
                    str((payload or {}).get("job_id") or ""),
                    str((payload or {}).get("worker_id") or ""),
                    str((payload or {}).get("claim_token") or ""),
                    str((payload or {}).get("state") or ""),
                    (payload or {}).get("details") if isinstance((payload or {}).get("details"), Mapping) else {},
                )
            ),
            inputs=update_payload,
            outputs=update_result,
            api_name="local_worker_update",
        )
        local_evaluation_payload = gr.JSON(visible=False)
        local_evaluation_result = gr.JSON(visible=False)
        local_evaluation_button = gr.Button("Local evaluation enqueue API", visible=False)
        local_evaluation_button.click(
            lambda payload: _worker_api(
                lambda: local_jobs.enqueue_evaluation(
                    (payload or {}).get("config")
                    if isinstance((payload or {}).get("config"), Mapping)
                    else {},
                    owner_authorized=(payload or {}).get("owner_authorized") is True,
                )
            ),
            inputs=local_evaluation_payload,
            outputs=local_evaluation_result,
            api_name="local_evaluation_enqueue",
        )
        cohort_payload = gr.JSON(visible=False)
        cohort_result = gr.JSON(visible=False)
        cohort_enqueue_button = gr.Button("Local cohort enqueue API", visible=False)
        cohort_enqueue_button.click(
            lambda payload: _worker_api(
                lambda: local_jobs.enqueue_cohort(
                    (payload or {}).get("config")
                    if isinstance((payload or {}).get("config"), Mapping)
                    else {},
                    (payload or {}).get("dataset_manifest")
                    if isinstance((payload or {}).get("dataset_manifest"), Mapping)
                    else {},
                    owner_authorized=(payload or {}).get("owner_authorized") is True,
                )
            ),
            inputs=cohort_payload,
            outputs=cohort_result,
            api_name="local_cohort_enqueue",
        )
        dataset_payload = gr.JSON(visible=False)
        dataset_result = gr.JSON(visible=False)
        dataset_button = gr.Button("Local dataset register API", visible=False)
        dataset_button.click(
            lambda payload: _worker_api(
                lambda: datasets.register(payload if isinstance(payload, Mapping) else {})
            ),
            inputs=dataset_payload,
            outputs=dataset_result,
            api_name="local_dataset_register",
        )
        enqueue_payload = gr.JSON(visible=False)
        enqueue_result = gr.JSON(visible=False)
        enqueue_button = gr.Button("Local job enqueue API", visible=False)
        enqueue_button.click(
            lambda payload: _worker_api(
                lambda: local_jobs.enqueue(
                    (payload or {}).get("config") if isinstance((payload or {}).get("config"), Mapping) else {},
                    (payload or {}).get("dataset_manifest")
                    if isinstance((payload or {}).get("dataset_manifest"), Mapping)
                    else {},
                    owner_authorized=(payload or {}).get("owner_authorized") is True,
                )
            ),
            inputs=enqueue_payload,
            outputs=enqueue_result,
            api_name="local_job_enqueue",
        )

    return demo
