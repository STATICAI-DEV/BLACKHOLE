# BLACKHOLE Dataset Register

The canonical catalog is `hf://buckets/BluStatic/BLACKHOLE-DATASETS/manifests/dataset_catalog_v2.json`. Prepared English OASST2 data contains **5,240** training pairs and **289** validation pairs at the recorded 2026-10-10 revisions/hashes. It is prepared, not trained, and still requires privacy, near-duplicate, cross-split contamination and human quality review.

Registered sources include OASST1 (Apache-2.0, archival), OASST2 (Apache-2.0, English chat candidate), Raftico multilingual instructional dialogues (CC-BY-4.0 with attribution) and the Afrikaans candidate `jojo-ai-mst/Roleplay-Afrikaans` (publisher-declared MIT, Google-translated, quality/rights review pending). Afrikaans replaced Hindi in the active language plan; historical Hindi raw files remain provenance archives and are excluded from training.

Each registration records exact source ID/revision, license and review, role, split, language, rows/bytes, preprocessing version, dedup history and SHA-256. Missing/ambiguous licensing is `Blocked`. Evaluation/benchmark/test data is held out and never training-eligible. Registration stores metadata only and performs no bulk download.

