# BLACKHOLE Lab and Reporting Bridge Design

**Status:** Approved for implementation by the owner's 2026-10-10 execution contracts and the explicit instruction to build them.

## Intent

BLACKHOLE is an independent, private Hugging Face Gradio laboratory for evidence-based language-model experiments. It starts safely with no model, no approved local dataset and no checkpoint. A separate GitHub ledger exposes sanitized, versioned status snapshots without owning training data, checkpoints or raw logs.

The release must be useful before any training is authorized: operators can inspect mount health, register and review dataset manifests, inspect persisted metrics, exercise checkpoint/evaluation guardrails, and see a truthful chat-disabled state. No cloud fallback, canned model output, paid compute, unattended polling or automatic training is permitted.

## Architecture

The repository contains two deployable entry points sharing small, focused Python services:

1. `app.py` creates exactly five Gradio tabs: Overview, Training, Datasets, Testing Arena and Chat.
2. The `blackhole` package owns configuration, atomic persistence, provenance events, dataset validation, metric loading, checkpoint inspection, evaluation records and the model-adapter protocol.
3. Runtime state is rooted at `/data` and `/lab` in the Space. Environment overrides permit isolated local tests. Missing or unwritable mounts fail closed with `BLOCKED_STORAGE`; local development never masquerades as mounted production storage.
4. A single-process training controller enforces one active run, explicit authorization and adapter/dataset readiness. It supports safe stop and exact-resume validation, but ships with no model adapter because the owner-supplied STATIC-0T asset is unavailable.
5. `bridge/blackhole_status.py` independently reads Hugging Face metadata and small bucket artifacts, normalizes unknowns to `null`/`UNKNOWN`, writes `bridge/status.json` plus a timestamped Markdown report atomically, and never changes training state.
6. A manual-only GitHub Actions workflow runs the bridge with `HF_READ_TOKEN` and commits changed sanitized outputs using narrowly scoped `contents: write` permission.

## State and evidence boundaries

- Hugging Face buckets remain canonical for data and run artifacts. GitHub receives only documentation, hashes, source pointers and sanitized summaries.
- Provenance events are one-file-per-event under `/lab/governance/provenance/<run_id>/EVENTS/`, using UTC timestamp plus a unique ID. Persistence failure blocks the corresponding operation.
- Metrics and evaluation charts load only persisted records. Test fixtures are visibly marked `TEST-ONLY` and are never bundled as real run evidence.
- A checkpoint is usable only when its manifest, declared files, hashes, model identity and training status validate. Exact resume additionally requires optimizer, scheduler and random-state artifacts.
- Chat remains disabled with `NO TRAINED CHECKPOINT — Chat unavailable` until a trained compatible adapter/checkpoint pair is verified.
- Dataset registration records source revision, license, role, split, approval and content hash. Ambiguous/missing licensing is `Blocked`; evaluation/test roles are rejected for training.

## Deployment and provisioning

The target is private Space `BluStatic/BLACKHOLE`, Gradio SDK, `cpu-basic`. The target uses one private bucket, `BluStatic/BLACKHOLE`: its `DATA/` prefix mounts at `/data` and its `LAB/` prefix mounts at `/lab`. Provisioning must inspect the exact bucket identity and visibility before changing mounts. The legacy `BluStatic/BLACKHOLE-DATASETS` bucket is not copied or deleted automatically.

Provisioning uses current `huggingface_hub` volume APIs (`Volume`, `create_repo`, `set_space_volumes`) and verifies runtime volumes. It is a separately invoked operator script: no import-time resource creation, hardware changes or credential output. Deployment stops as `BLOCKED_AUTH` when no authenticated HF identity is available.

## Error handling

Public service functions return typed status records or raise narrow validation/persistence exceptions. UI callbacks catch those exceptions and show actionable, redacted messages. Atomic writes use a same-directory temporary file followed by replace. Locks use exclusive lock files with owner metadata and stale-lock refusal rather than silent takeover.

Bridge source failures do not erase prior evidence. A fresh output records overall `BLOCKED`, per-source error codes and `null` unknowns. Logs never include tokens, request headers, private prompt text or dataset rows.

## Testing

Pytest covers empty startup, mount health, atomic persistence, event audit, manifest/license/split rules, metric reload, checkpoint hash and resume guards, evaluation persistence, chat blocking, singleton training control, the five-tab UI contract, bridge normalization, missing Space/bucket behavior and secret redaction. Tests use temporary directories and fake clients; they do not train, download datasets or contact paid services.

Local verification runs the full test suite, compile/import checks, a launch smoke test, workflow YAML parsing, bridge CLI against deterministic fixtures, and a secret-pattern scan. Cloud verification is reported separately and requires authenticated HF access: private Space state, both mounted volumes, write/read probe across a safe restart, live boot log, and manually dispatched bridge workflow after `HF_READ_TOKEN` is provisioned.

## Explicit exclusions

No STATICOS/STATIC_CORE/STATIC_GPT integration, multimodal work, model invention, dataset download, model download, real training, paid Jobs, GPU switch, always-on service, public tunnel, or scheduled polling is included.
