from __future__ import annotations

import json

from scripts.diff_utils import invert_unified_diff
from scripts.foundation_data_common import (
    CONTEXT,
    SOFTWARE_ROOT,
    SOURCES,
    clean_text,
    conversation_length,
    is_english,
    split_for,
    stable_hash,
    stream,
    tokenizer,
    write_jsonl,
    save_provenance,
)

TARGETS = {
    "swe_smith": (20_000, 500),
    "swe_gym": (800, 50),
}

sp = tokenizer()
all_train: list[dict] = []
all_val: list[dict] = []
evidence: dict[str, dict] = {}

for source, (train_target, val_target) in TARGETS.items():
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    seen: set[str] = set()
    scanned = 0
    rejected_context = 0
    rejected_language = 0

    for row in stream(source):
        scanned += 1
        problem = row.get("problem_statement")
        patch = row.get("patch")
        instance_id = str(row.get("instance_id") or scanned)
        repo = str(row.get("repo") or "")
        if not isinstance(problem, str) or not isinstance(patch, str):
            continue
        problem = clean_text(problem)
        patch = clean_text(patch)
        if not problem or not patch or not is_english(problem):
            rejected_language += 1
            continue
        if source == "swe_smith":
            try:
                repair_patch = clean_text(invert_unified_diff(patch))
            except ValueError:
                continue
            target_semantics = "inverse_of_bug_injection_patch"
        else:
            repair_patch = patch
            target_semantics = "forward_reference_fix_patch"

        prompt = (
            "You are repairing a real software repository. Analyze the issue and produce the "
            "minimal code patch that fixes the failing behavior while preserving existing behavior.\n\n"
            f"Repository: {repo}\nIssue:\n{problem}"
        )
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": repair_patch},
        ]
        if conversation_length(messages, sp) > CONTEXT + 1:
            rejected_context += 1
            continue

        identity = stable_hash(source + "|" + instance_id + "|" + repair_patch)
        if identity in seen:
            continue
        seen.add(identity)
        validation_percent = 5 if source == "swe_gym" else 2
        split = split_for(source + "|" + instance_id, validation_percent=validation_percent)
        target = val_rows if split == "validation" else train_rows
        limit = val_target if split == "validation" else train_target
        if len(target) >= limit:
            continue
        target.append({
            "messages": messages,
            "source": SOURCES[source]["repo"],
            "source_subset": source,
            "domain": "software_engineering",
            "repository": repo,
            "instance_id": instance_id,
            "language": "en+python",
            "target_semantics": target_semantics,
            "pair_sha256": identity,
        })

        if len(train_rows) >= train_target and len(val_rows) >= val_target:
            break
        if scanned % 5000 == 0:
            print(json.dumps({
                "stage": "SOFTWARE_PROGRESS",
                "source": source,
                "scanned": scanned,
                "train": len(train_rows),
                "validation": len(val_rows),
                "rejected_context": rejected_context,
            }), flush=True)

    if len(train_rows) < train_target or len(val_rows) < val_target:
        raise RuntimeError(
            f"{source} incomplete: train={len(train_rows)}/{train_target}, "
            f"validation={len(val_rows)}/{val_target}, rejected_context={rejected_context}"
        )
    all_train.extend(train_rows)
    all_val.extend(val_rows)
    evidence[source] = {
        "source": SOURCES[source],
        "scanned": scanned,
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "rejected_context": rejected_context,
        "rejected_language": rejected_language,
        "target_semantics": (
            "inverse_of_bug_injection_patch"
            if source == "swe_smith"
            else "forward_reference_fix_patch"
        ),
    }
    print(json.dumps({
        "stage": "SOFTWARE_SOURCE_READY",
        "source": source,
        "train": len(train_rows),
        "validation": len(val_rows),
    }), flush=True)

train_ids = {row["pair_sha256"] for row in all_train}
val_ids = {row["pair_sha256"] for row in all_val}
if len(train_ids) != len(all_train) or len(val_ids) != len(all_val):
    raise RuntimeError("duplicate software-engineering rows detected")
if train_ids & val_ids:
    raise RuntimeError("software-engineering train/validation leakage detected")

all_train.sort(key=lambda row: stable_hash("software-train|" + row["pair_sha256"]))
all_val.sort(key=lambda row: stable_hash("software-val|" + row["pair_sha256"]))
train_n, train_sha = write_jsonl(SOFTWARE_ROOT / "train.jsonl", all_train)
val_n, val_sha = write_jsonl(SOFTWARE_ROOT / "validation.jsonl", all_val)
prov = {
    "schema_version": "staticai.software-engineering-sft.v2",
    "dataset_id": "STATICAI/STATIC-SOFTWARE-ENGINEERING-SFT-V2",
    "recipe": "software-engineering-sft-v1",
    "language_policy": "English issue descriptions and Python repository repair patches only.",
    "repair_semantics": {
        "swe_smith": "The upstream patch injects the synthetic defect; BLACKHOLE trains on its deterministic inverse as the repair target.",
        "swe_gym": "The upstream patch is the forward reference solution and is retained as the repair target.",
    },
    "train_rows": train_n,
    "validation_rows": val_n,
    "train_sha256": train_sha,
    "validation_sha256": val_sha,
    "mixture": {key: {"train": v[0], "validation": v[1]} for key, v in TARGETS.items()},
    "evidence": evidence,
}
prov_path = save_provenance("software_v1", prov)
print(json.dumps({"status": "SOFTWARE_READY", "provenance": str(prov_path), **prov}), flush=True)
