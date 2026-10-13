# STATIC-0T (provisional model card)
**State:** Reference. No pretrained or trained model weights have been validated from BLACKHOLE.

## Intended use
Research into small from-scratch conversational language models and native ternary-weight implementations. Not yet suitable for reliable answers or deployment.

## Prototype description (from earlier starter discussion, not independently audited here)
- Name: STATIC-0T, described as gated causal convolutions with ternary core weights in the forward pass.
- Earlier reported parameter count: **1,909,248** total; **1,867,776** ternary core.
- Earlier reported untrained packed core plus nonternary states: about **0.60 MB**.
- Training process purported to maintain higher-precision optimization states; final packed inference may require dequantization.
These are *starter claims to verify against actual source code, packed artifact and tests*, not measurements reproduced by BLACKHOLE.

## Restrictions and known limitations
Small capacity, no broad language pretraining, no verified conversation performance, and no audited multilingual capabilities. No certified safety, commercial-readiness or architectural novelty assertion. Keep future measurements and model versions separate.
