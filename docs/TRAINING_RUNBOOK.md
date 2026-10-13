# BLACKHOLE Training Runbook

Before Start, verify both mounted buckets, an audited model adapter, an Approved training manifest, fixed experiment protocol, explicit owner authorization and a free `cpu-basic` resource plan. Confirm no active lock and that provenance events can be persisted. AUTO_TRAIN and AUTO_RESUME remain off.

During a run, append real step metrics to `/lab/runs/<run_id>/metrics.jsonl`, persist checkpoints at the configured interval and record model, optimizer, scheduler and random-state hashes. Pause/Stop safely flushes metrics, emits an event and finalizes `RUN_SUMMARY.md`/`RUN_MANIFEST.json`. A missing heartbeat is stale/unknown, not evidence that training continues.

Resume requires model plus optimizer, scheduler and random-state integrity; otherwise use `RESUME_UNVERIFIED` or create a new run. Free CPU Spaces may sleep after inactivity (commonly about 48 hours) or restart, so training is best effort while awake. HF Jobs are a separately approved potentially paid alternative and are never auto-enabled. Troubleshoot mount/provenance failures before compute resumes.

## Windows LOCAL worker

Keep `LOCAL` and `CLOUD` explicit. `LOCAL` uses the laptop GPU and cannot resize the Space or create a paid job; `CLOUD` retains the existing controller behavior. Before allowing any local execution, confirm the Space is still private on `cpu-basic`, `hf auth login` identifies `BluStatic`, the adapter is audited and reports `cuda:0`, and the approved dataset manifest names a hash-matched JSONL object under the BLACKHOLE `DATA` bucket.

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
