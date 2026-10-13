# STATIC-0T Experiment Protocol
**State:** Reference (experiment not started or verified)

1. Audit starter source and confirm model architecture, trainable parameter count, forward/backward correctness, weight-quantization method and reproducible model initialization. Do not assume the downloaded starter is correct without inspecting it.
2. Fix random seed, training/validation corpus revisions and checksums, tokenizer, prompt format, train/test separation and evaluation prompts **before** training.
3. Run short, real CPU diagnostic (e.g. 10–100 steps) to confirm declining loss and stable gradients; never substitute mock loss curves.
4. Record wall-clock duration, optimizer configuration, batch size, accumulation, sequence length, dtype, token counts, gradient norm, memory, and hardware.
5. Train and checkpoint on fixed intervals, then evaluate held-out loss and generated responses.
6. Compare equivalent-resource ternary vs FP32 baseline before making efficiency claims.
7. Preserve all run configs, command lines, dataset manifests, logs, checkpoint hashes, actual measures and explicit conclusions.

## Gates
- No automatic training merely because data is staged.
- Paid accelerator use requires separate approval; no indefinite keep-awake service by default.
- No fabricated benchmarks, numerical performance claims or claims of 'first of its kind' absent research validation.
- Mark each claim **measured**, **inferred** or **unverified**.
