# BLACKHOLE Local GPU Worker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fail-closed Windows CUDA worker controlled by the private BLACKHOLE Space without starting training or allocating paid compute.

**Architecture:** The CPU Space remains the single queue controller and persists jobs under its mounted LAB bucket. A manually launched Windows process uses the private Gradio API over authenticated outbound HTTPS, validates CUDA and approved inputs, and delegates batches only to the existing in-project adapter interface.

**Tech Stack:** Python 3.11, Gradio/`gradio_client` 6.30.0, `huggingface_hub` 2.2.0, optional PyTorch 2.14.1 CUDA runtime, PowerShell, pytest.

**Spec:** `docs/superpowers/specs/2026-10-11-local-gpu-worker-design.md`

## Global Constraints

- Do not start training during setup or verification.
- No model or dataset download during setup verification.
- No inbound port, public tunnel, remote shell, arbitrary command, CPU fallback, framework switch, Windows service, or paid Hugging Face GPU.
- Preserve the existing CLOUD path; LOCAL and CLOUD remain explicit.
- Tokens, passwords, authorization headers, claim tokens, and local paths must not appear in persisted public status or logs.

## Review Focus

- Concurrent claims: exactly one worker owns a queued job and non-owner updates fail.
- Stale or malformed jobs: reject unsupported schema, unsafe IDs, wrong target, and invalid transitions without execution.
- CUDA unavailable or PyTorch missing: return `BLOCKED_CUDA`, never CPU execution.
- Unavailable/mismatched inputs: return `BLOCKED_INPUTS` before the training controller or adapter is invoked.
- Error leakage: redact credentials and local paths before remote persistence.

---

### Task 1: Durable local-job controller

**Files:**
- Create: `blackhole/local_jobs.py`
- Create: `tests/test_local_jobs.py`

**Interfaces:**
- Produces: `LocalJobQueue.enqueue(...)`, `announce(...)`, `claim(...)`, and `update(...) -> dict[str, Any]`.
- Consumes: `LabSettings`, `DatasetRegistry.verify_persisted`, atomic JSON persistence, and redaction.

- [ ] Write failing tests for strict LOCAL schema, unique job IDs, exclusive claim ownership, terminal-state immutability, redaction, and local-path removal.
- [ ] Run `pytest tests/test_local_jobs.py -v` and confirm the missing-module failure.
- [ ] Implement the minimal durable queue, claim lock/token, transition validation, and sanitized status records.
- [ ] Run `pytest tests/test_local_jobs.py -v` and confirm PASS.
- [ ] Commit the task.

### Task 2: Private Space dispatch and worker API

**Files:**
- Modify: `blackhole/ui.py`
- Modify: `tests/test_ui.py`

**Interfaces:**
- Consumes: `LocalJobQueue` from Task 1.
- Produces: explicit `LOCAL`/`CLOUD` dispatch plus named private worker announce/claim/update Gradio endpoints.

- [ ] Write failing UI tests for explicit target selection, preserved five tabs, named worker API routes, and a LOCAL callback that queues rather than invokes cloud training.
- [ ] Run the focused UI tests and confirm failure.
- [ ] Add the target selector, dispatch callback, hidden API components, and fail-closed error rendering while leaving current CLOUD behavior intact.
- [ ] Run the focused UI tests and confirm PASS.
- [ ] Commit the task.

### Task 3: CUDA worker and PowerShell launcher

**Files:**
- Create: `blackhole/local_worker.py`
- Create: `scripts/local_gpu_worker.py`
- Create: `scripts/start_local_gpu_worker.ps1`
- Create: `tests/test_local_worker.py`

**Interfaces:**
- Consumes: named Gradio APIs, `TrainingController`, `AdapterRegistry`, `MetricsStore`, `RunStore`, `HfFileSystem`.
- Produces: `inspect_cuda(...)`, `LocalWorker.check_only()`, `poll_once(execute_training: bool)`, and a one-command PowerShell launcher.

- [ ] Write failing tests for CUDA capability reports, PyTorch absence, no CPU fallback, check-only non-claiming, target/schema validation, missing inputs before adapter calls, and redacted remote results.
- [ ] Run `pytest tests/test_local_worker.py -v` and confirm the missing-module failure.
- [ ] Implement authenticated client transport, CUDA guard, explicit execution gate, JSONL/hash validation, adapter-loop execution, artifact upload, and signal-safe polling.
- [ ] Add the PowerShell wrapper with `-CheckOnly` and `-ExecuteTraining` as mutually exclusive modes.
- [ ] Run worker and launcher tests and confirm PASS without training or downloads.
- [ ] Commit the task.

### Task 4: Documentation, deployment, and evidence

**Files:**
- Modify: `README.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/TRAINING_RUNBOOK.md`
- Modify: `docs/VERIFICATION.md`
- Modify: `docs/CHANGELOG.md`
- Modify: `tests/test_docs.py`

**Interfaces:**
- Consumes: completed controller and worker commands.
- Produces: exact setup/start/stop instructions and PASS/BLOCKED evidence language.

- [ ] Write failing documentation tests for the exact command, explicit training gate, LOCAL/CLOUD distinction, no CPU fallback, and no paid Space GPU.
- [ ] Update the operator documentation and changelog without claiming a training run.
- [ ] Run all tests and confirm PASS.
- [ ] Install only required project/runtime dependencies; do not fetch models or datasets.
- [ ] Deploy the private Space code without changing hardware, then verify authenticated API connectivity, Space `cpu-basic`, and bucket persistence.
- [ ] Run `-CheckOnly`; record authenticated connection plus real CUDA name/free/total VRAM and report `BLOCKED_INPUTS` if no eligible job exists.
- [ ] Review the final diff for secrets, local paths, unrelated changes, and false completion claims; commit the task.

