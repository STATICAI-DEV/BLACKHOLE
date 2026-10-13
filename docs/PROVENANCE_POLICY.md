# BLACKHOLE GitHub Provenance Policy

Hugging Face storage is the source of truth for dataset objects, checkpoints, original run metrics and detailed private evidence. This GitHub repository stores only versioned documentation, code, schemas, checksums, source pointers and sanitized summaries. Raw datasets, weights, credentials, private prompts and unredacted logs are prohibited.

Every published snapshot records an observed UTC timestamp, source identifiers, relevant commit/checkpoint SHA values and explicit error or stale state. Historical dated snapshots are append-only by convention. `bridge/status.json` is the current cached view and may be replaced atomically; it must always identify itself as cached rather than live telemetry.

Retention follows research value and owner direction. Mutable bucket deletion or overwrite cannot be recovered from bucket history; finalized significant manifests and reports may therefore be committed here after redaction. Source read, authentication or validation failures are represented as `BLOCKED`/`UNKNOWN` and never converted into invented zero metrics or successful states.
