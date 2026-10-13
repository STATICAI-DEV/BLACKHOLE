# BLACKHOLE Governance and Provenance

Every run has immutable-by-convention event files under `/lab/governance/provenance/<run_id>/EVENTS/`, each named with UTC timestamp plus unique ID. Events cover experiment creation, model selection, dataset approval, training start/config change, validation, checkpoint, evaluation, stop, resume, failure and finalization. They record source references, relevant hashes and software revision while redacting credentials/private content.

Run start snapshots architecture/parameter and numeric semantics, dataset manifest and fingerprint, hyperparameters/seed/environment, prompt-set version, success/failure criteria and stopping rules. Checkpoints record step, metrics, SHA-256 and whether model and optimizer state were saved. Final `RUN_SUMMARY.md` and `RUN_MANIFEST.json` say what was and was not proven.

HF buckets are mutable and non-versioned. Hashes and append-by-convention EVENTS provide traceability, not a tamper-proof audit trail. Significant finalized manifests may be published to a private versioned repository without storing large checkpoint history. A provenance write failure is `BLOCKED_PROVENANCE` and compute must stop.

