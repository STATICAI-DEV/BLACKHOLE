from __future__ import annotations

import json
from pathlib import Path

from scripts.foundation_data_common import (
    CONTEXT,
    JUDGMENT_ROOT,
    SOURCES,
    clean_text,
    is_english,
    split_for,
    stable_hash,
    stream,
    tokenizer,
    write_jsonl,
    save_provenance,
)

ARENA_TRAIN_TARGET = 8_000
ARENA_VAL_TARGET = 250
root = JUDGMENT_ROOT / "preference"
base_train_path = root / "train.jsonl"
base_val_path = root / "validation.jsonl"
if not base_train_path.is_file() or not base_val_path.is_file():
    raise RuntimeError("HelpSteer preference corpus must be prepared before Arena augmentation")

def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows

sp = tokenizer()

def preference_length(prompt: str, response: str) -> int:
    return (
        3
        + len(sp.encode("\n" + prompt.strip() + "\n", out_type=int))
        + len(sp.encode("\n" + response.strip() + "\n", out_type=int))
    )

base_train = load_jsonl(base_train_path)
base_val = load_jsonl(base_val_path)
base_ids = {str(row.get("pair_sha256") or "") for row in base_train + base_val}

arena_train: list[dict] = []
arena_val: list[dict] = []
seen = set(base_ids)
scanned = 0
rejected_tie = 0
rejected_multiturn = 0
rejected_context = 0
rejected_language = 0
rejected_parse = 0

for row in stream("arena_preference"):
    scanned += 1
    try:
        prompts = json.loads(row.get("prompt") or "[]")
        responses_a = json.loads(row.get("response_a") or "[]")
        responses_b = json.loads(row.get("response_b") or "[]")
    except (TypeError, json.JSONDecodeError):
        rejected_parse += 1
        continue
    if not (
        isinstance(prompts, list)
        and isinstance(responses_a, list)
        and isinstance(responses_b, list)
        and len(prompts) == len(responses_a) == len(responses_b) == 1
    ):
        rejected_multiturn += 1
        continue
    winner_a = int(row.get("winner_model_a") or 0)
    winner_b = int(row.get("winner_model_b") or 0)
    tie = int(row.get("winner_tie") or 0)
    if tie or winner_a == winner_b:
        rejected_tie += 1
        continue

    prompt = clean_text(str(prompts[0] or ""))
    response_a = clean_text(str(responses_a[0] or ""))
    response_b = clean_text(str(responses_b[0] or ""))
    if not prompt or not response_a or not response_b or not is_english(prompt):
        rejected_language += 1
        continue
    chosen, rejected = (response_a, response_b) if winner_a else (response_b, response_a)
    if max(preference_length(prompt, chosen), preference_length(prompt, rejected)) > CONTEXT + 1:
        rejected_context += 1
        continue

    identity = stable_hash("arena|" + prompt + "\0" + chosen + "\0" + rejected)
    if identity in seen:
        continue
    seen.add(identity)
    split = split_for("arena-pref|" + str(row.get("id") or identity), validation_percent=3)
    target = arena_val if split == "validation" else arena_train
    limit = ARENA_VAL_TARGET if split == "validation" else ARENA_TRAIN_TARGET
    if len(target) >= limit:
        continue
    target.append({
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "source": SOURCES["arena_preference"]["repo"],
        "domain": "human_preference",
        "language": "en",
        "pair_sha256": identity,
    })
    if len(arena_train) >= ARENA_TRAIN_TARGET and len(arena_val) >= ARENA_VAL_TARGET:
        break
    if scanned % 10_000 == 0:
        print(json.dumps({
            "stage": "ARENA_PREFERENCE_PROGRESS",
            "scanned": scanned,
            "train": len(arena_train),
            "validation": len(arena_val),
            "rejected_multiturn": rejected_multiturn,
            "rejected_context": rejected_context,
        }), flush=True)

if len(arena_train) < ARENA_TRAIN_TARGET or len(arena_val) < ARENA_VAL_TARGET:
    raise RuntimeError(
        f"Arena preference target incomplete: train={len(arena_train)}/{ARENA_TRAIN_TARGET}, "
        f"validation={len(arena_val)}/{ARENA_VAL_TARGET}, scanned={scanned}"
    )

combined_train = base_train + arena_train
combined_val = base_val + arena_val
combined_train.sort(key=lambda row: stable_hash("preference-v2-train|" + row["pair_sha256"]))
combined_val.sort(key=lambda row: stable_hash("preference-v2-val|" + row["pair_sha256"]))

train_ids = {row["pair_sha256"] for row in combined_train}
val_ids = {row["pair_sha256"] for row in combined_val}
if len(train_ids) != len(combined_train) or len(val_ids) != len(combined_val) or train_ids & val_ids:
    raise RuntimeError("preference-v2 duplicate or train/validation leakage detected")

train_n, train_sha = write_jsonl(base_train_path, combined_train)
val_n, val_sha = write_jsonl(base_val_path, combined_val)
prov = {
    "schema_version": "staticai.preference-dpo.v2",
    "dataset_id": "STATICAI/STATIC-JUDGMENT-PREFERENCE-V2",
    "recipe": "preference-dpo-v1",
    "license": "mixed-cc-by-4.0-apache-2.0",
    "language_policy": (
        "English natural-language preference pairs only. HelpSteer multilingual-domain rows are excluded. "
        "LMArena contribution is restricted to single-turn, non-tie English human preference battles."
    ),
    "train_rows": train_n,
    "validation_rows": val_n,
    "train_sha256": train_sha,
    "validation_sha256": val_sha,
    "mixture": {
        "helpsteer3": {"train": len(base_train), "validation": len(base_val)},
        "lmarena_human": {"train": len(arena_train), "validation": len(arena_val)},
    },
    "sources": {
        "helpsteer3": SOURCES["helpsteer_preference"],
        "lmarena_human": SOURCES["arena_preference"],
    },
    "arena_evidence": {
        "scanned": scanned,
        "rejected_tie": rejected_tie,
        "rejected_multiturn": rejected_multiturn,
        "rejected_context": rejected_context,
        "rejected_language": rejected_language,
        "rejected_parse": rejected_parse,
    },
}
prov_path = save_provenance("preference_v1", prov)
print(json.dumps({"status": "PREFERENCE_V2_READY", "provenance": str(prov_path), **prov}), flush=True)
