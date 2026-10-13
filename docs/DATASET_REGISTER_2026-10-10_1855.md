# BLACKHOLE Dataset Register
**Source of truth:** `hf://buckets/BluStatic/BLACKHOLE-DATASETS/manifests/dataset_catalog_v2.json`
**Read on:** 2026-10-10 | **State:** Reference

| Source | License as published | SHA source revision | Stored / intended use |
|---|---|---|---|
| OpenAssistant/oasst1 | Apache-2.0 | `fdf72ae0827c1cda404aff25b6603abec9e3399b` | Raw multilingual message-tree archive, not active training |
| OpenAssistant/oasst2 | Apache-2.0 | `179dd21fc55192153d94adb0e0ce8f69e222bf75` | Raw archive; filtered English dialogue pairs prepared |
| Raftico/instructional-dialogues-multilingual | CC-BY-4.0, attribution required | `11ddda73e515ed0f936091ba02215cf4b6e2f2af` | Raw 10-language instructional examples; candidate only |
| jojo-ai-mst/Roleplay-Afrikaans | MIT as declared by publisher | `b8ee74dfa8e14a79d0b3285b27be2cb8be358ad8` | Raw translated roleplay; **quality/rights review pending** |

## Initial English corpus
Source OASST2 revision `179dd21fc55192153d94adb0e0ce8f69e222bf75`, Apache-2.0.

| Split | Pairs | Bucket file | SHA-256 |
|---|---:|---|---|
| train | 5240 | `curated/en/english_conversation_train_2026-10-10_1754.jsonl` | `64c92e8a34b3b1bef9b4cc12ee7f6c3cd742be7acc7051ef4a5c50403c49c853` |
| validation | 289 | `curated/en/english_conversation_validation_2026-10-10_1754.jsonl` | `66e1e4cfeb9ff492d29b0a913c55fe94c4d76cb37fcf44b94df8abb451a74028` |

Preparation filters in bucket `manifests/english_conversation_2026-10-10_1754.json`: reviewed, non-deleted, non-synthetic English user/assistant exchange, label and length thresholds, within-split exact deduplication. **Not yet verified:** privacy audit, near-duplicate/cross-split contamination and human spot-check. This is conversation data, not sufficient evidence of full pretraining readiness.

## Language plan
1 English (en) **only selected training language**; 2 Mandarin (zh); 3 Afrikaans (af); 4 Spanish (es); 5 Arabic (ar); 6 French (fr); 7 Portuguese (pt); 8 Russian (ru); 9 German (de); 10 Japanese (ja).

Hindi (hi) was explicitly replaced by Afrikaans in the active plan. Raftico's historic `hi` file remains archived; do not silently include it in training. Languages listed here do not necessarily each have a sufficiently large or suitable corpus.

## Provenance rules
Keep original source URL, owner, immutable revision, license, acquisition timestamp, file checksum, transformation parameters, splits, excluded rows summary, approval and training-run linkage. Do not train from sources without permission/quality review. Source cards:
- https://huggingface.co/datasets/OpenAssistant/oasst1
- https://huggingface.co/datasets/OpenAssistant/oasst2
- https://huggingface.co/datasets/Raftico/instructional-dialogues-multilingual
- https://huggingface.co/datasets/jojo-ai-mst/Roleplay-Afrikaans
