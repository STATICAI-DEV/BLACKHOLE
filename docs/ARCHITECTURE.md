# BLACKHOLE Architecture

The private Gradio Space exposes exactly five tabs: Overview, Training, Datasets, Testing Arena and Chat. `app.py` composes focused services in `blackhole/`; model-specific code implements the small adapter protocol without changing the lab infrastructure.

`/data` is the read/write mount for `DATA/` in the private `BluStatic/BLACKHOLE` bucket. `/lab` mounts the isolated `LAB/` prefix from the same bucket and contains `runs/`, `checkpoints/`, `evaluations/`, `governance/provenance/`, `governance/docs/` and `exports/`. Production refuses missing/unwritable mounts rather than silently using ephemeral disk.

Data flows from reviewed manifests into a single-writer training controller, then to persisted metrics/checkpoints and held-out evaluation records. The GitHub bridge reads only sanitized metadata and small manifests. Raw data, weights, prompts, tokens and private logs never cross that security boundary. Atomic writes and one-file-per-event provenance reduce collision risk; bucket mutability means hashes provide traceability, not immutability.

## Local GPU control plane

Training dispatch has two explicit targets. `CLOUD` preserves the existing Space-side controller path. `LOCAL` writes a `blackhole.local-job.v1` record below `/lab/local-worker/jobs/`; it never creates a Hugging Face Job or changes Space hardware. The private Space serializes claims and returns a random claim token that binds status and completion updates to one announced worker.

The Windows worker uses only authenticated outbound HTTPS to the private Space. It opens no inbound listener, public tunnel or remote VRAM-sharing interface. Capability checks require PyTorch to expose `cuda:0`, the adapter must report that same device, and there is no CPU fallback. The allowlisted data input is an approved, hash-matched JSONL object below `hf://buckets/BluStatic/BLACKHOLE/DATA/`; arbitrary URLs, commands, scripts and local paths are not job fields.

Status, metrics, redacted errors and final `hf://buckets/BluStatic/BLACKHOLE/LAB/` artifact references return through the controller. Local filesystem paths, tokens, passwords and authorization headers are removed before persistence. The Space stays on `cpu-basic`, so this control plane allocates no paid Hugging Face GPU.
