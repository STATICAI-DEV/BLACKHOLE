# BLACKHOLE Governance and Provenance
**State:** Reference | **Scope:** BLACKHOLE laboratory only.

## Immutable research record
For every dataset and experiment, capture:
- owner, source URI, license, source revision, collection date and approval;
- preprocessing code/config revision, excluded-row counts, corpus checksums and splits;
- model code revision, initialization seed, trained weights provenance, quantization/export method;
- job owner, start/end times, hardware, software versions, configuration, actual costs if known;
- step-indexed metrics, checkpoints with SHA-256 and generated evaluation outputs;
- explicit errors, termination causes, retries, changed hypotheses and human decisions.

**Evidence-of-completion rule:** never promote `PLANNED` or `ACTIVE` to `COMPLETED` merely because a script exited without error. Require traceable artifact(s), checksums and evaluation or explicitly record that verification is pending.

## Storage strategy
Hugging Face bucket `BLACKHOLE-DATASETS` holds data/manifests and current early provenance. Codex may designate a separate checkpoint bucket **only with explicit scope and account approval**. GitHub BLACKHOLE holds versioned descriptions and sanitized reports. Since bucket data are mutable, publish an append-only revision log or snapshots in GitHub after independent verification.

## Boundaries
No secrets/tokens, hidden conversation content, raw user data or unlicensed corpora in Git. Retain versioned records; do not silently overwrite past experiment evidence. Status report must carry source, observation time, evidence pointer and stale indicator.

## Dated source records
- `hf://buckets/BluStatic/BLACKHOLE-DATASETS/governance/provenance/BLACKHOLE_Dataset_Intake_2026-10-10_1755.md`
- `hf://buckets/BluStatic/BLACKHOLE-DATASETS/governance/provenance/BLACKHOLE_Language_Swap_2026-10-10.md`
