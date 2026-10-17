from __future__ import annotations

import json

from scripts.foundation_data_common import (
    CONTEXT,
    REASONING_ROOT,
    SOURCES,
    conversation_length,
    normalize_open_thought,
    split_for,
    stable_hash,
    stream,
    tokenizer,
    write_jsonl,
    save_provenance,
)

TRAIN_TARGET = 8_000
VAL_TARGET = 160

sp = tokenizer()
train_rows: list[dict] = []
val_rows: list[dict] = []
seen: set[str] = set()
scanned = 0
rejected_context = 0
rejected_language = 0

for row in stream("reasoning"):
    scanned += 1
    messages = normalize_open_thought(row)
    if messages is None:
        rejected_language += 1
        continue
    if conversation_length(messages, sp) > CONTEXT + 1:
        rejected_context += 1
        continue

    identity = stable_hash(messages[0]["content"] + "\0" + messages[1]["content"])
    if identity in seen:
        continue
    seen.add(identity)
    split = split_for("reasoning114k|" + stable_hash(messages[0]["content"]))
    target = val_rows if split == "validation" else train_rows
    limit = VAL_TARGET if split == "validation" else TRAIN_TARGET
    if len(target) >= limit:
        continue
    target.append({
        "messages": messages,
        "source": SOURCES["reasoning"]["repo"],
        "domain": "mixed_reasoning",
        "language": "en",
        "pair_sha256": identity,
    })

    if len(train_rows) >= TRAIN_TARGET and len(val_rows) >= VAL_TARGET:
        break
    if scanned % 5_000 == 0:
        print(json.dumps({
            "stage": "REASONING_PROGRESS",
            "scanned": scanned,
            "train": len(train_rows),
            "validation": len(val_rows),
            "rejected_context": rejected_context,
            "rejected_language": rejected_language,
        }), flush=True)

if len(train_rows) < TRAIN_TARGET or len(val_rows) < VAL_TARGET:
    raise RuntimeError(
        f"reasoning target incomplete: train={len(train_rows)}/{TRAIN_TARGET}, "
        f"validation={len(val_rows)}/{VAL_TARGET}, scanned={scanned}, "
        f"context={rejected_context}, language={rejected_language}"
    )

train_ids = {row["pair_sha256"] for row in train_rows}
val_ids = {row["pair_sha256"] for row in val_rows}
if len(train_ids) != len(train_rows) or len(val_ids) != len(val_rows):
    raise RuntimeError("duplicate reasoning rows detected")
if train_ids & val_ids:
    raise RuntimeError("reasoning train/validation leakage detected")

train_rows.sort(key=lambda row: stable_hash("reason-train|" + row["pair_sha256"]))
val_rows.sort(key=lambda row: stable_hash("reason-val|" + row["pair_sha256"]))
train_n, train_sha = write_jsonl(REASONING_ROOT / "train.jsonl", train_rows)
val_n, val_sha = write_jsonl(REASONING_ROOT / "validation.jsonl", val_rows)
prov = {
    "schema_version": "staticai.reasoning-mid.v2",
    "dataset_id": "STATICAI/STATIC-REASONING-MID-V2",
    "recipe": "reasoning-mid-v1",
    "source": SOURCES["reasoning"],
    "language_policy": "English prompts only; mathematical notation and source code are retained.",
    "coverage": ["mathematics", "science", "code", "puzzles"],
    "train_rows": train_n,
    "validation_rows": val_n,
    "train_sha256": train_sha,
    "validation_sha256": val_sha,
    "scanned": scanned,
    "rejected_context": rejected_context,
    "rejected_language": rejected_language,
}
path = save_provenance("reasoning_v1", prov)
print(json.dumps({"status": "REASONING_READY", "provenance": str(path), **prov}), flush=True)
