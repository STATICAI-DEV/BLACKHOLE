from __future__ import annotations

import json
from pathlib import Path

from scripts.foundation_data_common import (
    CONTEXT,
    JUDGMENT_ROOT,
    PROVENANCE_ROOT,
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

OPENR1_TRAIN_TARGET = 256
OPENR1_VAL_TARGET = 5
root = JUDGMENT_ROOT / "preference"
train_path = root / "train.jsonl"
val_path = root / "validation.jsonl"
prov_path = PROVENANCE_ROOT / "preference_v1.json"

if not train_path.is_file() or not val_path.is_file() or not prov_path.is_file():
    raise RuntimeError("preference V2 must exist before OpenR1 reasoning augmentation")

def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows

base_train = load_jsonl(train_path)
base_val = load_jsonl(val_path)
base_prov = json.loads(prov_path.read_text(encoding="utf-8"))
sp = tokenizer()

def preference_length(prompt: str, response: str) -> int:
    if len(prompt) + len(response) > 12_000:
        return CONTEXT + 2
    return (
        3
        + len(sp.encode("\n" + prompt.strip() + "\n", out_type=int))
        + len(sp.encode("\n" + response.strip() + "\n", out_type=int))
    )

openr1_train: list[dict] = []
openr1_val: list[dict] = []
seen = {str(row.get("pair_sha256") or "") for row in base_train + base_val}
scanned = 0
rejected_no_pair = 0
rejected_context = 0
rejected_language = 0

for row in stream("openr1_reasoning_preference"):
    scanned += 1
    problem = row.get("problem")
    generations = row.get("generations")
    correctness = row.get("correctness_math_verify")
    complete = row.get("is_reasoning_complete")
    if not isinstance(problem, str) or not problem.strip() or not is_english(problem):
        rejected_language += 1
        continue
    if not isinstance(generations, list) or not isinstance(correctness, list):
        rejected_no_pair += 1
        continue
    if len(generations) != len(correctness):
        rejected_no_pair += 1
        continue
    if not isinstance(complete, list) or len(complete) != len(generations):
        complete = [True] * len(generations)

    chosen = None
    rejected = None
    for generation, is_correct, is_complete in zip(generations, correctness, complete):
        if not isinstance(generation, str) or not generation.strip():
            continue
        if is_correct is True and is_complete is True and chosen is None:
            chosen = clean_text(generation)
        elif is_correct is False and rejected is None:
            rejected = clean_text(generation)
        if chosen and rejected:
            break
    if not chosen or not rejected:
        rejected_no_pair += 1
        continue

    prompt = clean_text(problem)
    if max(preference_length(prompt, chosen), preference_length(prompt, rejected)) > CONTEXT + 1:
        rejected_context += 1
        continue

    identity = stable_hash("openr1|" + prompt + "\0" + chosen + "\0" + rejected)
    if identity in seen:
        continue
    seen.add(identity)
    split = split_for("openr1-pref|" + str(row.get("uuid") or identity), validation_percent=3)
    target = openr1_val if split == "validation" else openr1_train
    limit = OPENR1_VAL_TARGET if split == "validation" else OPENR1_TRAIN_TARGET
    if len(target) >= limit:
        continue
    target.append({
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "source": SOURCES["openr1_reasoning_preference"]["repo"],
        "domain": "reasoning_judgment_math",
        "language": "en",
        "pair_sha256": identity,
    })

    if len(openr1_train) >= OPENR1_TRAIN_TARGET and len(openr1_val) >= OPENR1_VAL_TARGET:
        break
    if scanned % 10_000 == 0:
        print(json.dumps({
            "stage": "OPENR1_PREFERENCE_PROGRESS",
            "scanned": scanned,
            "train": len(openr1_train),
            "validation": len(openr1_val),
            "rejected_no_pair": rejected_no_pair,
            "rejected_context": rejected_context,
        }), flush=True)

if len(openr1_train) < OPENR1_TRAIN_TARGET or len(openr1_val) < OPENR1_VAL_TARGET:
    raise RuntimeError(
        f"OpenR1 preference target incomplete: train={len(openr1_train)}/{OPENR1_TRAIN_TARGET}, "
        f"validation={len(openr1_val)}/{OPENR1_VAL_TARGET}, scanned={scanned}, "
        f"no_pair={rejected_no_pair}, context={rejected_context}"
    )

combined_train = base_train + openr1_train
combined_val = base_val + openr1_val
combined_train.sort(key=lambda row: stable_hash("preference-v3-train|" + row["pair_sha256"]))
combined_val.sort(key=lambda row: stable_hash("preference-v3-val|" + row["pair_sha256"]))

train_ids = {row["pair_sha256"] for row in combined_train}
val_ids = {row["pair_sha256"] for row in combined_val}
if len(train_ids) != len(combined_train) or len(val_ids) != len(combined_val) or train_ids & val_ids:
    raise RuntimeError("preference-v3 duplicate or train/validation leakage detected")

train_n, train_sha = write_jsonl(train_path, combined_train)
val_n, val_sha = write_jsonl(val_path, combined_val)
prior_mixture = dict(base_prov.get("mixture") or {})
prior_mixture["openr1_verified_reasoning"] = {
    "train": len(openr1_train),
    "validation": len(openr1_val),
}
prior_sources = dict(base_prov.get("sources") or {})
prior_sources["openr1_verified_reasoning"] = SOURCES["openr1_reasoning_preference"]

prov = {
    "schema_version": "staticai.preference-dpo.v3",
    "dataset_id": "STATICAI/STATIC-JUDGMENT-PREFERENCE-V3",
    "recipe": "preference-dpo-v1",
    "license": "mixed-cc-by-4.0-apache-2.0",
    "language_policy": (
        "English preference pairs only. The mixture combines HelpSteer reviewer preferences, "
        "single-turn non-tie LMArena human preferences, and OpenR1 math pairs where the chosen "
        "reasoning trace is machine-verified correct and the rejected trace is verified incorrect."
    ),
    "train_rows": train_n,
    "validation_rows": val_n,
    "train_sha256": train_sha,
    "validation_sha256": val_sha,
    "mixture": prior_mixture,
    "sources": prior_sources,
    "prior_evidence": {
        "arena_evidence": base_prov.get("arena_evidence"),
    },
    "openr1_evidence": {
        "scanned": scanned,
        "rejected_no_pair": rejected_no_pair,
        "rejected_context": rejected_context,
        "rejected_language": rejected_language,
    },
}
saved = save_provenance("preference_v1", prov)
print(json.dumps({"status": "PREFERENCE_V3_READY", "provenance": str(saved), **prov}), flush=True)
