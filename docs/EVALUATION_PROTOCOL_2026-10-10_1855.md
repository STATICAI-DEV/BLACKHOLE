# BLACKHOLE Evaluation Protocol
**State:** Reference; no results yet.

## Metrics
1. **Learning:** train/validation token cross-entropy, perplexity where defined, loss trend, overfitting gap and held-out text.
2. **Conversation:** fixed untouched English prompts (greetings, follow-up questions, brief instructions, context continuity); preserve raw generated responses for blinded comparison where permitted.
3. **Size/speed:** packed checkpoint bytes, load-time RAM, peak inference RAM, first-token latency and tokens/s on identified hardware.
4. **Reliability:** deterministic reload, seeded reproduction, errors and exact checkpoint hashes.
5. **Comparisons:** identical evaluation sets, hardware, token budgets and generation settings for ternary versus FP32.

## Contamination controls
Split conversation trees, not individual messages, to reduce leakage; perform cross-split duplicate checks before the first scored evaluation. Do not train on benchmark answers or casually expand eval prompts into the training corpus.

## Honesty
A decreasing loss is evidence of learning, **not** evidence of fluent chatbot behavior. Unrun benchmarks stay blank/null. Record regressions as well as improvements.
