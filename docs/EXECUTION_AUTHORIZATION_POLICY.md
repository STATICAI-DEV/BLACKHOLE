# BLACKHOLE Execution Authorization Policy

**Status:** Enforced project rule  
**Scope:** BLACKHOLE training, evaluation, local workers, command blocks, Hugging Face Jobs and any external AI/operator acting on the project.

## 1. Local execution is the default

BLACKHOLE defaults to `LOCAL`. A LOCAL job may execute only on an explicitly local worker that has announced CUDA capability and claimed the durable authorized job.

A LOCAL authorization MUST NOT be converted, upgraded, substituted or silently rerouted to cloud compute.

## 2. Strict cloud GPU authorization

Cloud GPU compute is **deny-by-default**.

No model, assistant, agent, worker, automation, connector or operator may start, schedule, rent, reserve or submit cloud GPU compute for BLACKHOLE unless the user gives a separate explicit authorization for cloud GPU execution at the time of the requested operation.

The following do **not** authorize cloud GPU use:

- approval to train;
- approval to evaluate;
- approval to use the local RTX GPU;
- approval of a dataset;
- approval of a command block;
- a previous cloud GPU authorization;
- a generic instruction to "run", "retry", "continue", "test" or "make it faster";
- an assistant's inference that cloud hardware would be convenient.

Within the BLACKHOLE Space UI, cloud execution requires both:

1. the normal explicit training authorization; and
2. a separate cloud-compute authorization with the exact phrase `AUTHORIZE CLOUD GPU`.

If either is absent, execution must fail closed with `CLOUD_GPU_NOT_AUTHORIZED`.

For external Hugging Face Jobs or similar services, an AI/operator must obtain fresh explicit user authorization that clearly states cloud GPU use before submitting a GPU-flavor job. CPU-only read/test jobs are not cloud GPU authorization.

## 3. Authorized command blocks

Once the user explicitly authorizes a specific durable LOCAL job, BLACKHOLE may generate an execution command block for that exact job.

That command block is automatically approved for execution and MUST NOT require a second BLACKHOLE authorization prompt, provided:

- the durable job remains authorized;
- its configuration SHA-256 still matches the authorized configuration;
- the execution target remains `LOCAL`; and
- the command does not add or alter training/evaluation parameters.

Any configuration change invalidates the automatic command approval and requires a new authorization.

## 4. Evidence rule

`COMPLETED` may be recorded only after durable result artifacts exist and are referenced by the job record. Planning, queueing, claiming, process start or command submission is not completion.

## 5. Network failures

Local CUDA training must not be terminated merely because metric/telemetry synchronization is temporarily unavailable. Local metrics and recovery checkpoints remain authoritative until remote synchronization succeeds.

## 6. Recovery checkpoints

Local training uses recovery checkpoints at the configured interval, default 50 steps. Recovery checkpoints contain the model and all adapter-provided resume state. BLACKHOLE retains the latest three recovery checkpoints plus the final checkpoint.

## 7. Testing Arena

Held-out Arena evaluation is a first-class LOCAL job type and uses the same authorization, claim ownership, evidence and completion rules as training.

---

This policy is intentionally fail-closed. Convenience is not authorization.


## Local synchronized cohorts

A `COHORT_TRAIN` authorization covers the complete named cohort configuration, including all member run IDs/seeds, dataset identity, physical batch size, gradient accumulation, shuffle seed and dataset cursor. Any configuration change invalidates the stored configuration hash.

Cohort execution is LOCAL only. One worker/CUDA process owns the cohort job. Members receive identical examples in identical shuffled order and update on the same optimizer boundaries while retaining independent model, optimizer, scheduler, RNG, metrics and checkpoint state.

The dataset cursor means examples consumed in the deterministic shuffled order. BLACKHOLE must not silently wrap to row zero when the requested optimizer steps and effective batch exhaust the approved dataset.
