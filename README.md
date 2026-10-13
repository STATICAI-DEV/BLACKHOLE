---
title: BLACKHOLE
emoji: 🕳️
colorFrom: gray
colorTo: indigo
sdk: gradio
sdk_version: 6.30.0
python_version: 3.11
app_file: app.py
pinned: false
fullWidth: true
short_description: Evidence-first language-model research laboratory
---

# BLACKHOLE

BLACKHOLE is an evidence-first research laboratory for controlled, from-scratch language-model experiments. It combines a Gradio control interface with guarded dataset handling, durable training and evaluation records, checkpoint recovery, provenance tracking, and a local CUDA worker for running authorized training jobs on a Windows GPU.

This repository contains the **clean validated software build only**. It intentionally contains **no trained models, checkpoints, datasets, training logs, run metrics, credentials, or virtual environments**.

## Validation status

The packaged build was validated on 13 October 2026 before publication.

**SOURCE DEPLOYMENT AT VALIDATION: PRIVATE SPACE RUNNING**  
**BASELINE TRAINING: 3 completed runs (STATIC-0T, STATIC-0X, STATIC-0K)**

These lines describe the validated source deployment, not the privacy state of this public GitHub repository.

- Automated regression suite: **99/99 passed**
- Three-model acceptance run: **3/3 passed**
- Local CUDA training: **passed**
- Intentional interruption and safe cancellation: **passed**
- Recovery checkpoint persistence and hash verification: **passed**
- Optimizer, scheduler, RNG and dataset-position restoration: **passed**
- Resume-to-completion: **passed**
- Live checkpoint discovery: **passed**
- 500-step bounded GPU stability run: **passed**

The acceptance run used three disposable experimental models (T/X/K). Their weights and runtime artifacts are **not included** in this repository.

## What BLACKHOLE provides

- Gradio-based training, evaluation, Arena and chat controls
- A static-0t model adapter for the validated experimental architecture
- Explicit LOCAL versus CLOUD execution targets
- Windows local-GPU worker with CUDA preflight checks
- Durable job queue and authorization gates
- Recovery and final checkpoints with integrity hashes
- Exact dataset-position continuation after interruption
- Evaluation persistence and checkpoint comparison
- Provenance and governance records
- Optional sanitized GitHub status bridge
- Fail-closed behavior when required storage, authorization or execution support is unavailable

## Requirements

The validated build used Python 3.11, Gradio 6.30.0, PyTorch 2.14.1, huggingface_hub 2.2.0, Plotly 7.1.0 and psutil 7.2.2.

See `requirements.txt` and `pyproject.toml` for the repository's declared dependencies.

## Install

~~~bash
git clone https://github.com/STATICAI-DEV/BLACKHOLE.git
cd BLACKHOLE
python -m venv .venv
~~~

Windows PowerShell:

~~~powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q tests
~~~

Linux/macOS:

~~~bash
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests
~~~

The clean release should report **99 passing tests** before deployment-specific configuration is introduced.

## Local development

Use explicit non-production storage roots:

~~~powershell
$env:BLACKHOLE_ENV = "test"
$env:BLACKHOLE_DATA_ROOT = ".blackhole-data"
$env:BLACKHOLE_LAB_ROOT = ".blackhole-lab"
python app.py
~~~

BLACKHOLE does not silently substitute ephemeral storage when required production mounts are absent. Missing or unwritable storage blocks training rather than pretending the job is safe to run.

## Hugging Face deployment

Authenticate with Hugging Face using the normal CLI credential store:

~~~bash
hf auth login
~~~

Provisioning is dry-run by default:

~~~bash
python scripts/provision_hf.py
~~~

Apply only after reviewing the plan:

~~~bash
python scripts/provision_hf.py --apply
~~~

Verify a deployment with:

~~~bash
python scripts/verify_deployment.py
~~~

This repository was packaged from a validated private BLACKHOLE deployment. **Before deploying a fresh instance, review the Hugging Face owner, Space and bucket identifiers in the configuration and scripts for your target account.** Do not reuse another deployment's runtime state, checkpoints, datasets or credentials.

## Windows local GPU worker

The Hugging Face Space acts as the controller while training can execute on an authorized local Windows CUDA GPU over outbound authenticated HTTPS. The worker opens no inbound port and does not create a public VRAM interface.

Preflight only:

~~~powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_local_gpu_worker.ps1 -CheckOnly
~~~

Execute one explicitly authorized job:

~~~powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_authorized_local_job.ps1 -JobId <authorized-job-id>
~~~

Continuous polling is available with:

~~~powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_local_gpu_worker.ps1 -ExecuteTraining
~~~

The validated controller deployment used a `cpu-basic` Space and the LOCAL worker had **no CPU fallback** and **no paid Hugging Face GPU** path. Stop the continuous worker with `Ctrl+C` when appropriate.

Local worker state is stored under `%LOCALAPPDATA%\BLACKHOLE_GPU_WORKER`.

## Recovery model

A running LOCAL job can receive a durable stop request. BLACKHOLE then saves a recovery checkpoint and records the job as `CANCELLED`.

A resume operation reloads the verified model state, optimizer state, scheduler state, RNG state and dataset position. Training then continues from the recorded optimizer step and exact dataset cursor rather than silently restarting from fresh weights or row zero.

## Synchronized cohort training

`COHORT_TRAIN` supports synchronized multi-model experiments using one local CUDA worker. Cohort members keep independent weights, optimizers, schedulers, seeds, metrics and checkpoints while sharing the deterministic dataset order and update boundaries.

The dataset cursor is measured in **examples consumed from the fixed shuffled order**, not raw source row numbers.

## Repository layout

~~~text
blackhole/        Core application, training, checkpoint and worker logic
bridge/           Optional sanitized status bridge
docs/             Architecture, governance, protocols and runbooks
scripts/          Provisioning, verification and local-worker entry points
tests/            Automated regression suite
app.py            Gradio application entry point
requirements.txt  Runtime dependencies
pyproject.toml     Python project metadata
~~~

## Security and operating principles

- No credentials belong in source control.
- Training requires explicit authorization.
- Missing storage or unsupported execution paths fail closed.
- Checkpoint hashes provide traceability for mutable storage.
- Runtime status files and generated reports should not be treated as live telemetry.
- Public source code does not include private datasets, models or operational secrets.

## Documentation

Start with:

- [Architecture](docs/ARCHITECTURE.md)
- [Training runbook](docs/TRAINING_RUNBOOK.md)
- [Dataset register](docs/DATASET_REGISTER.md)
- [Evaluation protocol](docs/EVALUATION_PROTOCOL.md)
- [Execution authorization policy](docs/EXECUTION_AUTHORIZATION_POLICY.md)
- [Governance and provenance](docs/GOVERNANCE_PROVENANCE.md)
- [Status bridge](bridge/README.md)

## Release provenance

The clean package used to create this repository records its source and validation metadata in `PACKAGE_MANIFEST.json`.

Source snapshot: `9534bd282ad1bb4ab16a9e1b128c297a6977a74d`

## License

See [LICENSE.txt](LICENSE.txt).
