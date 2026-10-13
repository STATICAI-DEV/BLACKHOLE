# BLACKHOLE Verification — 2026-10-11

This record covers the local repository and the authenticated private Hugging Face deployment. It is not evidence of a trained model, imported dataset, GitHub Actions secret or completed training run.

## Verified environment

- Python 3.11 isolated virtual environment
- Gradio 6.30.0
- huggingface_hub 2.2.0
- Plotly 7.1.0
- psutil 7.2.2
- PyYAML 6.0.2
- pytest 9.1.1
- Local worker PyTorch 2.14.1+cu130; the CPU Space keeps the platform-resolved `torch==2.14.1` requirement.

## Results

- `python -m pip check`: no broken core requirements.
- `python -m pytest -q`: **75 passed, 0 failed, 0 skipped**.
- `python -m compileall -q app.py blackhole bridge scripts`: passed.
- Gradio bounded launch on localhost: **HTTP 200**, then server closed.
- Workflow YAML/manual-only permission contract: passed in the test suite.
- Git diff whitespace validation: passed.
- Tracked non-test credential-pattern scan: **0 matches**.
- `scripts/provision_hf.py`: authenticated dry-run matched the approved private Space and bucket-path plan before apply.
- `scripts/verify_deployment.py`: live privacy, runtime and mount contract passed; restart probe passed separately.

## Live private deployment

- Authenticated Hugging Face identity: `BluStatic`.
- Private Space: `BluStatic/BLACKHOLE`; the authenticated deployment verifier confirms the currently served private revision without hard-coding a stale SHA here.
- Runtime: `RUNNING` on `cpu-basic`; no hardware change requested.
- Private bucket: `BluStatic/BLACKHOLE`.
- Configured mounts: bucket path `DATA` at `/data` and bucket path `LAB` at `/lab`, both read/write.
- Authenticated Space request: **HTTP 200**, `text/html`, 15,086 bytes.
- Restart persistence probe: **PASS**; the uniquely named 32-byte LAB probe matched before and after restart and was removed afterward.
- Boot evidence included the Gradio listener on `0.0.0.0:7860`. A non-fatal asyncio invalid-file-descriptor traceback appeared during lifecycle handling; the Space nevertheless returned to `RUNNING` and served HTTP 200.

## Local GPU worker setup

- Official PyTorch CUDA channel install: `torch==2.14.1+cu130`; `python -m pip check`: no broken requirements.
- Real CUDA detection: `cuda_available=true`, CUDA runtime `13.0`, NVIDIA GeForce RTX 5050 Laptop GPU, compute capability `12.0`.
- VRAM at check time: 7,413,432,320 bytes free of 8,518,041,600 bytes total. This is a point-in-time measurement, not a reservation.
- Exact `-CheckOnly` launcher command: **PASS**. It authenticated to the private Space, announced the capability record, and returned `training_started=false`.
- Durable controller evidence: `hf://buckets/BluStatic/BLACKHOLE/LAB/local-worker/workers/worker-b10a64b9e890363c.json` existed and matched the announced GPU report.
- Queue evidence: no `LAB/local-worker/jobs/*/JOB.json` objects existed; check-only made no claim.
- Input evidence: `DATA/` contained only `.blackhole-volume.json`; there was no approved JSONL training artifact and no concrete model adapter.
- Result: controller connection **PASS**, CUDA **PASS**, execution readiness **BLOCKED_INPUTS**, training **NOT STARTED**.
- Space hardware remained `cpu-basic`; provisioning reported `hardware_change: null` and created no bucket or Space.

## Not verified

The GitHub `HF_READ_TOKEN`, manual workflow execution and ChatGPT GitHub connector access remain unverified. The `DATA/` prefix has no training payload; legacy dataset artifacts were not copied from `BluStatic/BLACKHOLE-DATASETS`. No real job could be claimed because the required adapter and approved dataset artifact are absent. Training remains **NOT STARTED** and STATIC-0T remains **BLOCKED_MODEL_ASSET**.
