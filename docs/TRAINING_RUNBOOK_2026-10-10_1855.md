# BLACKHOLE Training Runbook
**State:** Planned. Codex is responsible for Hugging Face lab changes.

## Before a run
Check Space or job exists; storage bucket mounted with appropriate permissions; dataset/architecture hash validated; cost/time limits specified; training permissions approved; metrics output and checkpoint destination writable; reproduce seed and config; record model card + revision.

## During a run
Append timestamped true step/loss/token throughput, GPU/CPU utilization where available, checkpoint IDs and heartbeat. Never send private dataset records or credentials to GitHub. UI charts must read genuine persisted metrics rather than synthetic defaults. A disappearing heartbeat means **stale/unknown**, not automatically running.

## On interruption
Checkpoint and resume optimizer, scheduler, model and RNG state where feasible; verify hashes; record last committed step. A sleeping Space is not an uptime guarantee. Avoid paid keepalive jobs by default.

## Completion
Record end time, actual token count and wall-clock compute, metrics, checkpoint digest, inference smoke test and any failures. Archive immutable evidence in Hugging Face bucket; publish only sanitized summary to GitHub. Status reporter contract: `../bridge/BRIDGE_PROTOCOL_2026-10-10_1855.md`.
