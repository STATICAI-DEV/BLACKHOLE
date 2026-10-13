# BLACKHOLE-specific GitHub Status Bridge
**State:** GitHub read interface ready; Hugging Face -> GitHub automatic writer NOT DEPLOYED.

## Contract
Each reporting event commits a new UTF-8 JSON file at `bridge/status_YYYY-MM-DD_HHMM.json` (Africa/Johannesburg local wall time) on this repository's `main` branch, preserving old snapshots. Only trusted Codex/lab publisher should write these files.

Required fields: `schema_version`, `observed_at` (ISO-8601 with offset), `report_type`, `source`, `sources_checked`, `lab`, `training`, `datasets`, `alerts`, `evidence`. Distinguish `true` evidence-backed values from `null` unknowns. A missing or old heartbeat NEVER means training is in progress.

Training object, when evidence exists: `run_id`, `stage` (not_started, prepared, running, paused, failed, completed, unknown), `global_step`, `max_steps`, `train_loss`, `val_loss`, `checkpoint_sha256`, `last_metric_at`, `hardware`. Null for unavailable values, no estimated-loss fabrication.

## Practical integration for Codex
1. After deploying the Hugging Face Space and reading actual metrics/manifest/checkpoint, validate and sanitize an output JSON.
2. Use a scoped GitHub token (contents:write on **BLACKHOLE only**), git HTTPS, or `gh` with appropriate authenticated permission. Never commit tokens or local dataset samples.
3. Create a new dated status snapshot; commit it to `main` with recorded source evidence; do not edit previous snapshots.
4. Emit the corresponding timestamped `reports/run_*.md` only after verified run evidence exists. Retry push conflicts without deleting snapshots.
5. Confirm a fresh file was committed and ChatGPT GitHub read connector can read it. Avoid paid recurring jobs until approved.

## ChatGPT report procedure
List `bridge/`, choose newest dated status filename, read JSON, check observed_at and source URL, optionally read the Hugging Face bucket/Space directly, and present only verified results. If snapshot is stale or live sync not installed, clearly say so. This needs no additional repository-creation connector capability.
