# BLACKHOLE status bridge

The bridge is an **optional, read-only reporting path** for publishing a sanitized BLACKHOLE status snapshot to GitHub.

It does not start, stop, resume or modify training.

## Data flow

~~~text
Hugging Face Space / Jobs / buckets
              |
              v
     bridge.blackhole_status
              |
              v
 status.json + timestamped report
~~~

`status.json` is a cached `blackhole.status.v1` snapshot. It is **not live telemetry** and must not be represented as such.

## Run manually

Provide a fine-grained Hugging Face read token through the environment:

~~~bash
HF_READ_TOKEN=... python -m bridge.blackhole_status
~~~

Never pass a token as a command-line argument and never commit a token to the repository.

## GitHub Actions

The included workflow is manual (`workflow_dispatch`) only. If used, configure the repository secret `HF_READ_TOKEN`.

Missing credentials must produce `BLOCKED_SECRET`, not a false successful status.

## Interpreting output

Always inspect `observed_at`, `source_state`, `blockers`, and the source identifiers.

Unknown data remains `null` or `UNKNOWN`. A missing Space is reported as `NOT_DEPLOYED`; source failures are reported as blocked details.

Generated bridge status and reports are runtime artifacts, not authoritative proof that training is currently active.
