from __future__ import annotations

import json

from scripts.foundation_data_common import (
    CONTEXT,
    JUDGMENT_ROOT,
    SOURCES,
    clean_text,
    is_english,
    last_user_text,
    split_for,
    stable_hash,
    stream,
    tokenizer,
    write_jsonl,
    save_provenance,
)

TRAIN_TARGET = 16_000
VAL_TARGET = 500
sp = tokenizer()
train_rows: list[dict] = []
val_rows: list[dict] = []
seen: set[str] = set()
scanned = 0
rejected_language = 0
rejected_context = 0


def encode_len(prompt: str, response: str) -> int:
    user_id = sp.piece_to_id("<|user|>")
    assistant_id = sp.piece_to_id("<|assistant|>")
    eos_id = sp.piece_to_id("<|eos|>")
    return (
        3
        + len(sp.encode("\n" + prompt.strip() + "\n", out_type=int))
        + len(sp.encode("\n" + response.strip() + "\n", out_type=int))
    )


for row in stream("helpsteer_preference"):
    scanned += 1
    domain = str(row.get("domain") or "general")
    if domain.casefold() == "multilingual":
        rejected_language += 1
        continue
    context = row.get("context")
    task = last_user_text(context)
    response1 = row.get("response1")
    response2 = row.get("response2")
    preference = row.get("overall_preference")
    if (
        not task
        or not is_english(task)
        or not isinstance(response1, str)
        or not isinstance(response2, str)
        or not isinstance(preference, int)
        or preference == 0
    ):
        rejected_language += 1
        continue

    prompt = clean_text(task)
    first = clean_text(response1)
    second = clean_text(response2)
    chosen, rejected = (first, second) if preference < 0 else (second, first)
    if not chosen or not rejected:
        continue
    if max(encode_len(prompt, chosen), encode_len(prompt, rejected)) > CONTEXT + 1:
        rejected_context += 1
        continue

    identity = stable_hash(prompt + "\0" + chosen + "\0" + rejected)
    if identity in seen:
        continue
    seen.add(identity)
    split = split_for("preference|" + identity, validation_percent=3)
    target = val_rows if split == "validation" else train_rows
    limit = VAL_TARGET if split == "validation" else TRAIN_TARGET
    if len(target) >= limit:
        continue
    target.append({
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "source": SOURCES["helpsteer_preference"]["repo"],
        "domain": domain,
        "language": "en",
        "pair_sha256": identity,
    })

    if len(train_rows) >= TRAIN_TARGET and len(val_rows) >= VAL_TARGET:
        break
    if scanned % 5000 == 0:
        print(json.dumps({
            "stage": "PREFERENCE_PROGRESS",
            "scanned": scanned,
            "train": len(train_rows),
            "validation": len(val_rows),
            "rejected_context": rejected_context,
            "rejected_language": rejected_language,
        }), flush=True)

if len(train_rows) < TRAIN_TARGET or len(val_rows) < VAL_TARGET:
    raise RuntimeError(
        f"preference target incomplete: train={len(train_rows)}/{TRAIN_TARGET}, "
        f"validation={len(val_rows)}/{VAL_TARGET}"
    )

train_ids = {row["pair_sha256"] for row in train_rows}
val_ids = {row["pair_sha256"] for row in val_rows}
if train_ids & val_ids:
    raise RuntimeError("preference train/validation leakage detected")

train_rows.sort(key=lambda row: stable_hash("preference-train|" + row["pair_sha256"]))
val_rows.sort(key=lambda row: stable_hash("preference-val|" + row["pair_sha256"]))
root = JUDGMENT_ROOT / "preference"
train_n, train_sha = write_jsonl(root / "train.jsonl", train_rows)
val_n, val_sha = write_jsonl(root / "validation.jsonl", val_rows)
prov = {
    "schema_version": "staticai.preference-dpo.v1",
    "dataset_id": "STATICAI/STATIC-JUDGMENT-PREFERENCE-V1",
    "recipe": "preference-dpo-v1",
    "source": SOURCES["helpsteer_preference"],
    "language_policy": "English natural-language preference pairs only; multilingual domain excluded.",
    "train_rows": train_n,
    "validation_rows": val_n,
    "train_sha256": train_sha,
    "validation_sha256": val_sha,
    "scanned": scanned,
    "rejected_context": rejected_context,
    "rejected_language": rejected_language,
}
prov_path = save_provenance("preference_v1", prov)
print(json.dumps({"status": "PREFERENCE_READY", "provenance": str(prov_path), **prov}), flush=True)
