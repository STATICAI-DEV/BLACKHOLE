# BLACKHOLE Changelog

## 2026-10-11 — Private Space deployment

- Created private Space `BluStatic/BLACKHOLE` on `cpu-basic`; no hardware upgrade was requested.
- Created private bucket `BluStatic/BLACKHOLE` with isolated `DATA/` and `LAB/` prefixes mounted read/write at `/data` and `/lab`.
- Verified the live Space reached `RUNNING`, returned authenticated HTTP 200, and retained a unique LAB probe across a controlled restart before the probe was removed.
- Corrected Space volume serialization to send the bucket ID as `source` and the subfolder as `path`; corrected verification to read configured volumes from `space_info().runtime`.
- Left the legacy `BluStatic/BLACKHOLE-DATASETS` bucket untouched; no datasets or models were copied or downloaded.
- TRAINING remains NOT STARTED and STATIC-0T remains BLOCKED_MODEL_ASSET.

## 2026-10-11 — Windows local GPU worker setup

- Added an explicit `LOCAL` dispatch target while preserving the existing `CLOUD` controller path.
- Added private Space API endpoints for CUDA capability announcements, exclusive durable claims, status/metric updates and claim-token-bound completion.
- Added the foreground PowerShell worker with structurally separate `-CheckOnly` and `-ExecuteTraining` modes, outbound HTTPS only, and no CPU fallback.
- Restricted local inputs to approved, SHA-256-matched JSONL artifacts inside `BluStatic/BLACKHOLE/DATA`; arbitrary commands, scripts, URLs and local paths are rejected.
- Kept the Space on `cpu-basic`; no paid Hugging Face GPU, Job, public tunnel, model download, dataset download or training run was started by setup.

## 2026-10-10 — Executable lab and bridge

- Added storage-safe Python services, provenance, dataset registry, persisted metrics, checkpoint/evaluation guards and single-writer training lifecycle.
- Added exactly five Gradio tabs with truthful empty/no-model/no-checkpoint states and persisted charts.
- Added stable read-only status bridge plus manual-only GitHub workflow.
- Added dry-run-first private HF provisioning and verification tools using current volume APIs.
- Preserved the earlier dated research-ledger documents and baseline snapshot for provenance.
- Deployment remains **NOT DEPLOYED** because this environment is not authenticated to Hugging Face.
- TRAINING: NOT STARTED. No dataset/model download, dataset import, paid compute, Job or training run occurred.
- Clean core environment verification: 41 tests passed, compile/import passed, dependency check passed and local Gradio HTTP returned 200 using Gradio 6.30.0 and huggingface_hub 2.2.0. Hugging Face provisioning/deployment remained `BLOCKED_AUTH`.
