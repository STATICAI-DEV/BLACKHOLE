# BLACKHOLE Experiment Protocol

Before an experiment, freeze the adapter/code revision, architecture and parameter counts, tokenizer semantics, dataset source/revision/license/hash, tree-aware train/validation/test split, seed, optimizer, learning rate, batch/context size, step budget, checkpoint/evaluation intervals and held-out prompt-set version. The baseline and candidate must differ by one declared controlled variable and share resource limits.

Acceptance criteria are set before start: successful forward/backward diagnostics, finite gradients, persisted metrics from real approved data, validation behavior better than the declared baseline, deterministic reload and verified checkpoint hashes. Conversational acceptance uses fixed held-out prompts and does not rely on an invented IQ score.

Failure criteria include non-finite loss, provenance persistence failure, mount failure, split contamination, corrupt/mismatched checkpoint, memory limit breach or inability to reproduce the exact run configuration. Stop safely, flush evidence and label the run failed/blocked; never silently continue. No real smoke run occurs until explicit training authorization.

