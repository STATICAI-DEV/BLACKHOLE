from __future__ import annotations
import json
from collections import Counter

from scripts.foundation_data_common import (
    PRETRAIN_ROOT, SOURCES, clean_text, is_english, split_for, stable_hash,
    stream, token_chunks, tokenizer, write_jsonl, save_provenance,
)

TARGETS = {
    "fineweb": (120000, 1000),
    "code": (30000, 300),
    "finemath": (25000, 250),
}

sp = tokenizer()
all_train = []
all_val = []
evidence = {}

for source, (train_target, val_target) in TARGETS.items():
    train_rows = []
    val_rows = []
    seen = set()
    scanned = 0
    rejected_language = 0
    for row in stream(source):
        scanned += 1
        if source == "fineweb":
            text = row.get("text")
            key = str(row.get("id") or row.get("url") or scanned)
            if row.get("language") not in (None, "en"):
                rejected_language += 1
                continue
        elif source == "code":
            text = row.get("content")
            key = str(row.get("repo_name") or "") + "|" + str(row.get("path") or scanned)
        else:
            text = row.get("text")
            key = str(row.get("url") or scanned)
            if row.get("language") != "en":
                rejected_language += 1
                continue
            score = row.get("language_score")
            if isinstance(score, (int, float)) and score < 0.75:
                rejected_language += 1
                continue
        if not isinstance(text, str):
            continue
        text = clean_text(text)
        if len(text) < 300:
            continue
        if source in {"fineweb", "finemath"} and not is_english(text):
            rejected_language += 1
            continue
        split = split_for(source + "|" + key)
        target = val_rows if split == "validation" else train_rows
        limit = val_target if split == "validation" else train_target
        if len(target) >= limit:
            continue
        for chunk in token_chunks(text, sp, max_tokens=1024):
            identity = stable_hash(chunk)
            if identity in seen:
                continue
            seen.add(identity)
            target.append({
                "text": chunk,
                "source": SOURCES[source]["repo"],
                "source_subset": source,
                "language": "code+en" if source == "code" else "en",
                "content_sha256": identity,
            })
            if len(target) >= limit:
                break
        if len(train_rows) >= train_target and len(val_rows) >= val_target:
            break
        if scanned % 10000 == 0:
            print(json.dumps({
                "stage": "PRETRAIN_PROGRESS", "source": source,
                "scanned": scanned, "train": len(train_rows),
                "validation": len(val_rows),
            }), flush=True)
    if len(train_rows) < train_target or len(val_rows) < val_target:
        raise RuntimeError(
            f"{source} incomplete: train={len(train_rows)}/{train_target}, "
            f"validation={len(val_rows)}/{val_target}"
        )
    all_train.extend(train_rows)
    all_val.extend(val_rows)
    evidence[source] = {
        "source": SOURCES[source],
        "scanned": scanned,
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "rejected_language": rejected_language,
    }
    print(json.dumps({
        "stage": "PRETRAIN_SOURCE_READY", "source": source,
        "train": len(train_rows), "validation": len(val_rows),
    }), flush=True)

train_ids = {row["content_sha256"] for row in all_train}
val_ids = {row["content_sha256"] for row in all_val}
if len(train_ids) != len(all_train) or len(val_ids) != len(all_val):
    raise RuntimeError("duplicate pretraining chunks detected")
if train_ids & val_ids:
    raise RuntimeError("pretraining train/validation leakage detected")

all_train.sort(key=lambda row: stable_hash("train|" + row["content_sha256"]))
all_val.sort(key=lambda row: stable_hash("validation|" + row["content_sha256"]))
train_n, train_sha = write_jsonl(PRETRAIN_ROOT / "train.jsonl", all_train)
val_n, val_sha = write_jsonl(PRETRAIN_ROOT / "validation.jsonl", all_val)
prov = {
    "schema_version": "staticai.foundation-pretrain.v1",
    "dataset_id": "STATICAI/STATIC-FOUNDATION-PRETRAIN-V1",
    "recipe": "causal-pretrain-v1",
    "language_policy": "English natural-language data; English-tagged code/explanations.",
    "train_rows": train_n,
    "validation_rows": val_n,
    "train_sha256": train_sha,
    "validation_sha256": val_sha,
    "mixture": {key: {"train": v[0], "validation": v[1]} for key, v in TARGETS.items()},
    "evidence": evidence,
}
path = save_provenance("pretrain_v1", prov)
print(json.dumps({"status": "PRETRAIN_READY", "provenance": str(path), **prov}, default=str), flush=True)
