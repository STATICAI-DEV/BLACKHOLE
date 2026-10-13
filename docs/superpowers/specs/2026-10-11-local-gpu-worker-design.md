# BLACKHOLE Local GPU Worker Design

**State:** Approved for setup; training remains prohibited during setup.

## Objective

Keep the private `BluStatic/BLACKHOLE` CPU Space as the controller while an operator-launched Windows worker executes explicitly `LOCAL` training jobs on the laptop NVIDIA GPU. The worker uses authenticated outbound HTTPS only. It opens no inbound port, exposes no public tunnel, does not share VRAM remotely, and never provisions Hugging Face GPU hardware.

## Existing boundaries

- Reuse `TrainingController`, `DatasetRegistry`, `AdapterRegistry`, `MetricsStore`, `RunStore`, checkpoint conventions, redaction, and the mounted `BluStatic/BLACKHOLE/LAB` storage.
- Preserve the current `CLOUD` path. Dispatch targets are explicit `LOCAL` or `CLOUD`; a local dispatch cannot create or resize a Hugging Face Job or Space.
- No model adapter or dataset payload is bundled. Missing, incompatible, unapproved, or hash-mismatched inputs are `BLOCKED`, never substituted.
- No framework switch, remote shell, arbitrary command field, CPU fallback, model download, dataset download during setup, Windows service, or background orchestrator.

## Controller protocol

The Space owns a durable queue at `/lab/local-worker/jobs/<job_id>/JOB.json` and exposes private Gradio API endpoints for capability announcement, claim, heartbeat/status, and completion. The private Space token authenticates HTTPS. A server-side lock plus atomic files serializes claim and terminal transitions. Each claim receives a random claim token; updates without the matching worker ID and token are rejected.

Jobs use `blackhole.local-job.v1`, contain a unique safe job ID, `execution_target: LOCAL`, a bounded training configuration, and an already-persisted approved dataset manifest. The protocol has no command, script, shell, URL, or local-path execution field. Returned errors and logs pass through the existing redactor. Remotely persisted artifact references use `hf://buckets/BluStatic/BLACKHOLE/LAB/...`, not local paths.

## Windows worker

`scripts/local_gpu_worker.py` connects with `gradio_client.Client(..., token=get_token())`, announces a stable pseudonymous worker ID plus CUDA capability, polls, and claims only `LOCAL` jobs. It requires PyTorch CUDA, selects `cuda:0`, reports GPU name and free/total VRAM, and refuses silent CPU fallback.

Before training it requires an explicitly registered in-project adapter whose inspection reports the selected CUDA device, an approved persisted dataset manifest, and an allowlisted bucket artifact under `hf://buckets/BluStatic/BLACKHOLE/DATA/` whose SHA-256 matches the manifest. Dataset retrieval and training occur only after explicit `-ExecuteTraining`; setup verification uses `-CheckOnly` and performs neither.

The executor streams JSONL batches to the existing adapter `train_step` method, persists metrics/runs using existing stores, saves adapter artifacts, uploads sanitized run artifacts to the existing LAB bucket, and reports only remote artifact references and hashes. Unsupported formats or missing assets are terminal `BLOCKED` results.

## Operator lifecycle

- Setup/auth/CUDA verification: `powershell -ExecutionPolicy Bypass -File .\scripts\start_local_gpu_worker.ps1 -CheckOnly`
- Later execution (only after separate training authorization and valid inputs): `powershell -ExecutionPolicy Bypass -File .\scripts\start_local_gpu_worker.ps1 -ExecuteTraining`
- Stop: press `Ctrl+C`; no service or scheduled task remains.

## Acceptance

- Authenticated private Space connection is proven.
- CUDA device name and free/total VRAM are read from PyTorch; `nvidia-smi` is corroborating evidence only.
- A real claim is reported only if a real eligible job exists. With no adapter/dataset asset, verification reports `BLOCKED_INPUTS` without creating or claiming a training job.
- Space hardware remains `cpu-basic`; no paid GPU allocation or Hugging Face training Job is created.
- Tests cover schema/target rejection, exclusive claims, claim-token ownership, redaction, no CPU fallback, and check-only behavior that cannot train.

