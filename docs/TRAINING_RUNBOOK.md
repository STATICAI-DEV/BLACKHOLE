# BLACKHOLE Training Runbook

Before Start, verify both mounted buckets, an audited model adapter, an Approved training manifest, fixed experiment protocol, explicit owner authorization and a free `cpu-basic` resource plan. Confirm no active lock and that provenance events can be persisted. AUTO_TRAIN and AUTO_RESUME remain off.

During a run, append real step metrics to `/lab/runs/<run_id>/metrics.jsonl`, persist checkpoints at the configured interval and record model, optimizer, scheduler and random-state hashes. Pause/Stop safely flushes metrics, emits an event and finalizes `RUN_SUMMARY.md`/`RUN_MANIFEST.json`. A missing heartbeat is stale/unknown, not evidence that training continues.

Resume requires model plus optimizer, scheduler and random-state integrity; otherwise use `RESUME_UNVERIFIED` or create a new run. Free CPU Spaces may sleep after inactivity (commonly about 48 hours) or restart, so training is best effort while awake. HF Jobs are a separately approved potentially paid alternative and are never auto-enabled. Troubleshoot mount/provenance failures before compute resumes.

## Windows LOCAL worker

Keep `LOCAL` and `CLOUD` explicit. `LOCAL` uses the laptop GPU and cannot resize the Space or create a paid job; `CLOUD` retains the existing controller behavior. Before allowing any local execution, confirm the Space is still private on `cpu-basic`, `hf auth login` identifies `STATICAI`, the adapter is audited and reports `cuda:0`, and the approved dataset manifest names a hash-matched JSONL object under the BLACKHOLE `DATA` bucket.

On this Windows/CUDA 13 machine, install the pinned CUDA wheel from the official PyTorch index once (the ordinary PyPI wheel is CPU-only):

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps torch==2.14.1+cu130 --index-url https://download.pytorch.org/whl/cu130
```

Verify setup without claiming a job or starting training:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_local_gpu_worker.ps1 -CheckOnly
```

The result must show an authenticated controller connection, actual NVIDIA device name, and free/total VRAM. `BLOCKED_CUDA` is a stop condition; there is no CPU fallback. `BLOCKED_INPUTS` is the truthful current result while the adapter or approved artifact is absent.

After setup is final and the owner separately authorizes the intended run, launch the foreground poller:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_local_gpu_worker.ps1 -ExecuteTraining
```

Stop with `Ctrl+C`. No service, scheduled task, inbound port or public tunnel is installed. Do not put a token on the command line. Inspect `/lab/local-worker/jobs/<job_id>/JOB.json`, `/lab/runs/<run_id>/`, hashes and provenance before reporting completion. A claim without a completed run manifest and remote artifact references is not a successful training run.


## English SFT recipe v2

New conversational runs use `training_recipe_version=assistant-response-v2`. The user prompt and conversation markers remain model context, but supervised loss is calculated only over the assistant response and the terminating `[END]` sequence. Rows that do not fit the 256-byte STATIC-0T context are rejected during dataset preparation; they are never randomly cropped into malformed training examples.

Training order is deterministic per dataset epoch using `shuffle_seed + dataset_epoch`. `dataset_position` means examples consumed inside the current shuffled epoch and is stored separately from the cumulative optimizer step. Recovery checkpoints preserve both values.

Validation is evaluated over a fixed bounded validation sample (`validation_max_rows`, default 256) and reported as an aggregate loss rather than a single batch.

### English-only foundation data

`scripts/prepare_english_foundation_v1.py` prepares `STATICAI/STATIC-ENGLISH-FOUNDATION-V1` without starting training. External OpenAssistant rows are accepted only when both source messages are explicitly tagged `lang == en`, reviewed, undeleted, structurally valid and short enough for the model context. Conversation trees are assigned wholly to one split to prevent train/validation/test leakage.

Run dataset preparation from a separate preparation environment, not the BLACKHOLE runtime venv. On Windows, loading PyArrow through Gradio/Pandas before CUDA PyTorch can cause DLL initialization conflicts.

```powershell
python -m venv .data-venv
.\.data-venv\Scripts\python.exe -m pip install -r .\scripts\requirements-data.txt
$env:PYTHONPATH = (Get-Location).Path
.\.data-venv\Scripts\python.exe -m scripts.prepare_english_foundation_v1
```

The approved V1 split is 4,000 training rows, 500 validation rows and 500 held-out test rows. The held-out test artifact is not eligible for training.

### Arena v2

`english-heldout-v2` contains 50 deterministic English smoke-test prompts across vocabulary, question answering, instruction following, basic reasoning and conversation. It records per-prompt pass/fail plus overall and category scores. Arena scores are evaluation evidence, not a substitute for inspecting generated text.


## Foundation generation V2

Fresh STATIC-X, STATIC-T and STATIC-K foundation models use `STATIC-FOUNDATION-TERNARY-v1`, not the legacy 256-byte STATIC-0T training path.

### Local model profile

- decoder-only causal Transformer
- 147,284,736 parameters
- 122,683,392 ternary projection parameters
- 32,000-piece SentencePiece BPE vocabulary
- 16 layers, width 768, 12 attention heads, FFN width 2304
- 2,048-token context
- RoPE positional encoding
- RMSNorm and gated feed-forward blocks
- gradient checkpointing
- AdamW with stage-specific warmup/decay
- lazy adapter construction so only the requested model occupies CUDA memory

A full-context causal smoke test must begin near random-language-model cross entropy (approximately ln(32000)) before a foundation run is accepted. The validated initialization produces loss near 10.6 and the local RTX 5050 training probe fits comfortably.

### Six-stage curriculum

1. **Foundation pretraining** (`causal-pretrain-v1`)
   - FineWeb-Edu English educational text
   - curated English/code material
   - FineMath educational mathematics
   - next-token causal language-model objective

2. **Reasoning mid-training** (`reasoning-mid-v1`)
   - English-filtered OpenThoughts-114k reasoning traces covering mathematics, science, code and puzzles
   - complete conversation supervision
   - long examples that do not fit the model context are rejected, never truncated into malformed reasoning

3. **Software engineering SFT** (`software-engineering-sft-v1`)
   - SWE-smith Python repository issue-to-patch tasks
   - SWE-Gym real repository repair tasks
   - English issue descriptions, source-code patches retained as code

4. **Instruction SFT** (`instruction-sft-v1`)
   - English general instruction following
   - English CodeAlpaca code-generation and programming instructions
   - assistant-response supervision

5. **Judgment SFT** (`judgment-sft-v1`)
   - HelpSteer3 human preference reasoning
   - critique/feedback
   - edited improved responses
   - explicit principles such as correctness, relevance, clarity and factual accuracy

6. **Preference optimization** (`preference-dpo-v1`)
   - HelpSteer3 reviewer-chosen/rejected response pairs
   - LMArena single-turn non-tie human preference pairs
   - OpenR1 Math correct-vs-incorrect reasoning pairs using machine verification
   - frozen copy of the stage-5 model is the DPO reference policy
   - DPO beta defaults to 0.1
   - local profile uses physical batch 1 with gradient accumulation 8

Each stage starts from the previous model weights but creates a fresh optimizer. Stage provenance and hashes are retained separately.

### Language policy

Natural-language training is English-only. Rows from multilingual human-language subsets are excluded. Programming languages such as Python, JavaScript, SQL, C/C++ and shell are code and may be retained.

### Scale warning

This local profile exists to train and validate the STATIC architecture on the RTX 5050. It is not represented as frontier-scale. Market-leading software-engineering or reasoning capability requires substantially larger model capacity, context, token budgets and compute; BLACKHOLE's staged dataset/recipe contracts are designed so larger profiles can reuse the same curriculum without rewriting the training logic.


### Held-out capability benchmarks

- Judgment: RewardBench 2, 1,776 English preference/judge cases. It is evaluation-only and is never registered as a training stage.
- Software engineering: SWE-bench Verified, 500 human-validated repository issues. Benchmark tasks and expected patches/tests are stored separately. Evaluation must execute repository tests; text similarity is not a valid score.
- Benchmark files are SHA-256 checked at publication and marked `training_eligible: false`.


### SWE-smith repair semantics

The upstream SWE-smith `patch` field is the bug-injection diff, not the repair. BLACKHOLE never trains directly on that field. During preparation, the unified diff is deterministically inverted and the inverse is used as the assistant repair target. Binary patches are rejected. SWE-Gym follows SWE-bench semantics and its forward reference patch is retained unchanged.

This distinction is regression-tested because reversing it would train a coding model to introduce the reported bug instead of fixing it.


## Controlled three-model capability curriculum V3

BLACKHOLE trains exactly three fresh models: STATIC-T, STATIC-X and STATIC-K. They use the same architecture, the same nine dataset artifacts, the same hashes, the same stage order, the same data shuffle seed, the same optimizer/hyperparameter schedule, the same validation policy and the same local CUDA worker. The only intentional experimental variable is the model initialization seed.

Stage order:

1. causal pretraining - English, code and mathematics
2. reasoning mid-training - difficult complete reasoning traces
3. software-engineering SFT - repository repair
4. general instruction SFT - chat and code instruction following
5. business SFT - commerce, finance, pricing, operations, unit economics and executive decisions
6. relationship SFT - adult relationship conversation, persona continuity, boundaries, trust, affection and non-explicit flirting
7. browser/research SFT - BrowserGym page-state reasoning and executable actions plus source-grounded web research synthesis
8. judgment SFT - critique, comparison, error repair and principle-based judgment
9. preference DPO - human preferences and verified correct-vs-wrong reasoning

V3 training rows: 342,056. At effective batch 8 this is 42,757 optimizer steps per model for one complete pass.

The V3 business corpus contains 16,000 training rows. The relationship corpus contains 12,000 training rows, including 4,000 unique STATICAI-authored relationship situations and 8,000 persona-consistent multi-turn conversations. The browser/research corpus contains 8,000 training rows: 6,000 MIT-licensed BrowserGym reasoning/action examples plus 2,000 CC-BY-4.0 EntiWeave source-grounded research examples.

The experiment must never vary dataset order by model seed. DATA_ORDER_SEED is fixed at 20,261,017; only the initialization seeds differ.
