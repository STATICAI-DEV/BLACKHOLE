# BLACKHOLE Evaluation Protocol

Use a versioned held-out prompt suite covering vocabulary, dialogue turns, coherence, instruction following, repetition, elementary reasoning, robustness and format compliance. Record prompt, exact output, generation parameters, duration, seed when deterministic, adapter/checkpoint identity and SHA-256. Optional human ratings are subjective and remain separate from automatic metrics.

Checkpoint comparison uses the same prompts in the same order, prompt-set version, generation settings and identified hardware. Compare two persisted results side by side; never train on the prompt suite or its answers. Test/benchmark splits remain isolated by conversation tree or group where available.

Byte-level cross-entropy/perplexity is only comparable under matching tokenizer semantics. A declining validation loss is learning evidence, not proof of fluent chat. Unrun metrics remain null, failures remain visible, and multimodal evaluation is out of scope for this release.

