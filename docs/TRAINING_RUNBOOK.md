# BLACKHOLE Training Runbook

Before Start, verify both mounted buckets, an audited model adapter, an Approved training manifest, fixed experiment protocol, explicit owner authorization and a free `cpu-basic` resource plan. Confirm no active lock and that provenance events can be persisted. AUTO_TRAIN and AUTO_RESUME remain off.

During a run, append real step metrics to `/lab/runs/<run_id>/metrics.jsonl`, persist checkpoints at the configured interval and record model, optimizer, scheduler and random-state hashes. Pause/Stop safely flushes metrics, emits an event and finalizes `RUN_SUMMARY.md`/`RUN_MANIFEST.json`. A missing heartbeat is stale/unknown, not evidence that training continues.

Resume requires model plus optimizer, scheduler and random-state integrity; otherwise use `RESUME_UNVERIFIED` or create a new run. Free CPU Spaces may sleep after inactivity (commonly about 48 hours) or restart, so training is best effort while awake. HF Jobs are a separately approved potentially paid alternative and are never auto-enabled. Troubleshoot mount/provenance failures before compute resumes.

## Windows LOCAL worker

Keep `LOCAL` and `CLOUD` explicit. `LOCAL` uses the laptop GPU and cannot resize the Space or create a paid job; `CLOUD` retains the existing controller behavior. Before allowing any local execution, confirm the Space is still private on `cpu-basic`, `hf auth login` identifies `STATICAI`, the adapter is audited and reports `cuda:0`, and the approved dataset manifest names a hash-matched JSONL object under the BLACKHOLE `DATA` bucket.

On this Windows/CUDA 13 machine, install the pinned CUDA wheel from the official PyTorch index once (the ordinary PyPI wheel is CPU-only):

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps torch==2.14.1+cu130 --index-url https://download.pytorch.org/whl/cu130
```

Verify setup without claiming a job or starting training:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_local_gpu_worker.ps1 -CheckOnly
```

The result must show an authenticated controller connection, actual NVIDIA device name, and free/total VRAM. `BLOCKED_CUDA` is a stop condition; there is no CPU fallback. `BLOCKED_INPUTS` is the truthful current result while the adapter or approved artifact is absent.

After setup is final and the owner separately authorizes the intended run, launch the foreground poller:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_local_gpu_worker.ps1 -ExecuteTraining
```

Stop with `Ctrl+C`. No service, scheduled task, inbound port or public tunnel is installed. Do not put a token on the command line. Inspect `/lab/local-worker/jobs/<job_id>/JOB.json`, `/lab/runs/<run_id>/`, hashes and provenance before reporting completion. A claim without a completed run manifest and remote artifact references is not a successful training run.


## English SFT recipe v2

New conversational runs use `training_recipe_version=assistant-response-v2`. The user prompt and conversation markers remain model context, but supervised loss is calculated only over the assistant response and the terminating `[END]` sequence. Rows that do not fit the 256-byte STATIC-0T context are rejected during dataset preparation; they are never randomly cropped into malformed training examples.

Training order is deterministic per dataset epoch using `shuffle_seed + dataset_epoch`. `dataset_position` means examples consumed inside the current shuffled epoch and is stored separately from the cumulative optimizer step. Recovery checkpoints preserve both values.

Validation is evaluated over a fixed bounded validation sample (`validation_max_rows`, default 256) and reported as an aggregate loss rather than a single batch.

### English-only foundation data

`scripts/prepare_english_foundation_v1.py` prepares `STATICAI/STATIC-ENGLISH-FOUNDATION-V1` without starting training. External OpenAssistant rows are accepted only when both source messages are explicitly tagged `lang == en`, reviewed, undeleted, structurally valid and short enough for the model context. Conversation trees are assigned wholly to one split to prevent train/validation/test leakage.

Run dataset preparation from a separate preparation environment, not the BLACKHOLE runtime venv. On Windows, loading PyArrow through Gradio/Pandas before CUDA PyTorch can cause DLL initialization conflicts.

```powershell
python -m venv .data-venv
.\.data-venv\Scripts\python.exe -m pip install -r .\scripts\requirements-data.txt
$env:PYTHONPATH = (Get-Location).Path
.\.data-venv\Scripts\python.exe -m scripts.prepare_english_foundation_v1
```

The approved V1 split is 4,000 training rows, 500 validation rows and 500 held-out test rows. The held-out test artifact is not eligible for training.

### Arena v2

`english-heldout-v2` contains 50 deterministic English smoke-test prompts across vocabulary, question answering, instruction following, basic reasoning and conversation. It records per-prompt pass/fail plus overall and category scores. Arena scores are evaluation evidence, not a substitute for inspecting generated text.
