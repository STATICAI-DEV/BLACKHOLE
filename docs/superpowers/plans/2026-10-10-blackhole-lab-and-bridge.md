# BLACKHOLE Lab and Reporting Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a truthful, private-ready BLACKHOLE Gradio laboratory and read-only GitHub reporting bridge without starting training or paid compute.

**Architecture:** A focused `blackhole` Python package provides durable local-file services used by a five-tab Gradio UI. Hugging Face storage remains canonical, while a separate bridge normalizes small sanitized status artifacts into GitHub snapshots.

**Tech Stack:** Python 3.11, Gradio, Plotly, psutil, huggingface_hub, pytest, PyYAML, GitHub Actions

**Spec:** `docs/superpowers/specs/2026-10-10-blackhole-lab-and-bridge-design.md`

## Global Constraints

- Exactly five UI tabs: Overview, Training, Datasets, Testing Arena and Chat.
- Default to `NO_MODEL`, no dataset and `NO TRAINED CHECKPOINT — Chat unavailable` without crashing.
- `AUTO_TRAIN` and `AUTO_RESUME` remain off; no real training without separate owner authorization.
- No model/dataset downloads, paid APIs, paid Jobs, hardware upgrade or public deployment.
- `/data` and `/lab` are required production mounts; missing persistence blocks training.
- Store no raw datasets, checkpoints, credentials or private prompts in GitHub.
- The bridge is reporting-only and manual-dispatch-only.

## Review Focus

- Missing/unwritable production mounts must produce `BLOCKED_STORAGE`, never an ephemeral fallback.
- Malformed or ambiguously licensed dataset manifests must remain blocked and evaluation splits must never become training inputs.
- Corrupt, mismatched or incomplete checkpoints must not enable chat/resume.
- Concurrent start attempts and stale lock files must fail closed without corrupting run evidence.
- Partial HF/API failures must preserve unknowns and errors without leaking tokens or inventing healthy state.

---

### Task 1: Project contract and persistence foundation

**Files:**
- Create: `pyproject.toml`, `requirements.txt`, `.gitignore`
- Create: `blackhole/config.py`, `blackhole/atomic.py`, `blackhole/types.py`, `blackhole/__init__.py`
- Test: `tests/test_config.py`, `tests/test_atomic.py`

**Interfaces:**
- Produces: `LabSettings.from_env()`, `atomic_write_json(path, payload)`, `sha256_file(path)` and typed status enums.

- [ ] Write failing tests for environment overrides, production mount refusal, atomic replacement and stable SHA-256.
- [ ] Run the focused tests and confirm failures are caused by missing production code.
- [ ] Implement the minimum typed settings and atomic file primitives.
- [ ] Run focused tests and the full suite; commit the foundation.

### Task 2: Provenance, datasets and persisted metrics

**Files:**
- Create: `blackhole/provenance.py`, `blackhole/datasets.py`, `blackhole/metrics.py`
- Test: `tests/test_provenance.py`, `tests/test_datasets.py`, `tests/test_metrics.py`

**Interfaces:**
- Consumes: Task 1 atomic primitives and settings.
- Produces: `EventStore.emit/audit`, `DatasetRegistry.register/list`, `MetricsStore.append/load/series`.

- [ ] Write failing tests for unique immutable event files, audit hashes, license/split approval rules and restart-loaded metric series.
- [ ] Verify the expected red failures.
- [ ] Implement the three stores with redaction, validation and atomic/index-safe persistence.
- [ ] Run focused tests and the full suite; commit the data services.

### Task 3: Model, checkpoint, evaluation and chat guardrails

**Files:**
- Create: `blackhole/adapters/base.py`, `blackhole/adapters/registry.py`, `blackhole/checkpoints.py`, `blackhole/evaluation.py`, `blackhole/chat.py`
- Test: `tests/test_checkpoints.py`, `tests/test_evaluation.py`, `tests/test_chat.py`

**Interfaces:**
- Consumes: Task 1 hashes/atomic writes and Task 2 provenance.
- Produces: `ModelAdapter` protocol, `AdapterRegistry`, `CheckpointStore.inspect`, `EvaluationStore.save/compare`, `ChatService.respond`.

- [ ] Write failing tests for the empty adapter state, hash/model mismatch, exact-resume components, real persisted evaluation comparison and chat denial.
- [ ] Verify the expected red failures.
- [ ] Implement only the adapter interface and guarded stores; do not invent STATIC-0T.
- [ ] Run focused tests and the full suite; commit model-facing guardrails.

### Task 4: Single-writer training control and run finalization

**Files:**
- Create: `blackhole/training.py`, `blackhole/runs.py`
- Test: `tests/test_training.py`, `tests/test_runs.py`

**Interfaces:**
- Consumes: Tasks 1-3 settings, registry, checkpoint, metric and event services.
- Produces: `TrainingController.start/request_stop/resume/status` and `RunStore.create/finalize`.

- [ ] Write failing tests for explicit authorization, readiness checks, concurrent-run rejection, safe stop, exact-resume refusal and truthful final manifests.
- [ ] Verify the expected red failures.
- [ ] Implement the state machine and manifest/summary generation without any bundled trainer.
- [ ] Run focused tests and the full suite; commit training control.

### Task 5: Exactly-five-tab Gradio application

**Files:**
- Create: `app.py`, `blackhole/ui.py`, `blackhole/system.py`
- Test: `tests/test_ui.py`, `tests/test_app_import.py`

**Interfaces:**
- Consumes: Tasks 1-4 service APIs.
- Produces: `build_app(settings=None) -> gr.Blocks` and pure callback helpers returning UI-safe values/plots.

- [ ] Write failing tests for exact tab labels, empty-state copy, disabled controls, persisted metric plots and sanitized errors.
- [ ] Verify the expected red failures.
- [ ] Implement responsive Gradio blocks and callbacks with no import-time launch or cloud mutation.
- [ ] Run focused tests, full suite and a bounded local launch smoke; commit the UI.

### Task 6: Read-only GitHub status bridge

**Files:**
- Create: `bridge/blackhole_status.py`, `bridge/status.json`, `bridge/README.md`, `.github/workflows/blackhole-status.yml`
- Create: `reports/2026-10-10_*.md`
- Test: `tests/test_bridge.py`, `tests/test_workflow.py`

**Interfaces:**
- Consumes: HF metadata plus small JSON/JSONL artifacts through an injectable read-only client.
- Produces: schema `blackhole.status.v1`, stable `bridge/status.json` and timestamped Markdown reports.

- [ ] Write failing tests for `NOT_DEPLOYED`, `UNKNOWN`, partial-source `BLOCKED`, token redaction, stable schema and manual-only workflow permissions.
- [ ] Verify the expected red failures.
- [ ] Implement normalization, atomic outputs, CLI and the manual Actions workflow.
- [ ] Run focused tests, fixture CLI and full suite; commit the bridge.

### Task 7: Operator provisioning and complete documentation

**Files:**
- Create: `scripts/provision_hf.py`, `scripts/verify_deployment.py`
- Create stable docs required by the contracts under `docs/`
- Modify: `README.md`
- Test: `tests/test_provisioning.py`, `tests/test_docs.py`

**Interfaces:**
- Consumes: current `huggingface_hub` APIs and all repository runtime contracts.
- Produces: dry-run-first provisioning plan, authenticated apply mode and redacted deployment verification report.

- [ ] Write failing tests for identity mismatch, private resource requirements, volume replacement safety, dry-run default and documentation completeness.
- [ ] Verify the expected red failures.
- [ ] Implement provisioning/verification without accepting tokens as command-line values; populate all documents truthfully.
- [ ] Run focused tests and the full suite; commit operator tooling/docs.

### Task 8: Release verification and publication

**Files:**
- Modify: `docs/CHANGELOG.md`, `bridge/status.json`, initial current report as evidence permits.

**Interfaces:**
- Consumes: all tasks.
- Produces: a reproducible local verification record and published Git revision; cloud evidence only if authentication permits.

- [ ] Create an isolated virtual environment and install only pinned dependencies.
- [ ] Run full pytest, compile/import, launch smoke, YAML parsing and secret scan; record exact pass/fail/skip counts.
- [ ] Review `git diff`, commit the verified implementation and push `main` to the already verified private repository.
- [ ] If HF authentication is available, dry-run then apply private Space/bucket/volume provisioning, verify boot and restart persistence, provision `HF_READ_TOKEN`, manually dispatch the workflow and verify its commit.
- [ ] If authentication or a secret is absent, leave deployment untouched and report precise `BLOCKED_AUTH`/`BLOCKED_SECRET` evidence without weakening local completion claims.
