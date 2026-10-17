from __future__ import annotations

import json

from scripts.foundation_data_common import (
    CONTEXT,
    INSTRUCTION_ROOT,
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

ULTRA_TRAIN, ULTRA_VAL = 30_000, 750
CODE_TRAIN, CODE_VAL = 10_000, 250
sp = tokenizer()


def collect_ultrachat():
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    seen: set[str] = set()
    scanned = rejected_context = rejected_language = 0
    for row in stream("ultrachat"):
        scanned += 1
        raw = row.get("messages")
        prompt_id = str(row.get("prompt_id") or scanned)
        if not isinstance(raw, list):
            continue
        pairs: list[list[dict[str, str]]] = []
        for left, right in zip(raw, raw[1:]):
            if not isinstance(left, dict) or not isinstance(right, dict):
                continue
            if left.get("role") != "user" or right.get("role") != "assistant":
                continue
            prompt = clean_text(str(left.get("content") or ""))
            answer = clean_text(str(right.get("content") or ""))
            if not prompt or not answer or not is_english(prompt):
                rejected_language += 1
                continue
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]
            if conversation_length(messages, sp) > CONTEXT + 1:
                rejected_context += 1
                continue
            pairs.append(messages)

        split = split_for("ultrachat|" + prompt_id)
        target = val_rows if split == "validation" else train_rows
        limit = ULTRA_VAL if split == "validation" else ULTRA_TRAIN
        for messages in pairs:
            identity = stable_hash(messages[0]["content"] + "\0" + messages[1]["content"])
            if identity in seen or len(target) >= limit:
                continue
            seen.add(identity)
            target.append({
                "messages": messages,
                "source": SOURCES["ultrachat"]["repo"],
                "domain": "general_instruction",
                "language": "en",
                "pair_sha256": identity,
            })
        if len(train_rows) >= ULTRA_TRAIN and len(val_rows) >= ULTRA_VAL:
            break
        if scanned % 5_000 == 0:
            print(json.dumps({
                "stage": "ULTRACHAT_PROGRESS",
                "scanned": scanned,
                "train": len(train_rows),
                "validation": len(val_rows),
            }), flush=True)

    if len(train_rows) < ULTRA_TRAIN or len(val_rows) < ULTRA_VAL:
        raise RuntimeError(
            f"UltraChat target incomplete: {len(train_rows)}/{ULTRA_TRAIN}, "
            f"{len(val_rows)}/{ULTRA_VAL}"
        )
    return train_rows, val_rows, {
        "source": SOURCES["ultrachat"],
        "scanned": scanned,
        "rejected_context": rejected_context,
        "rejected_language": rejected_language,
    }


def collect_code_alpaca():
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    seen: set[str] = set()
    scanned = rejected_context = rejected_language = 0
    for row in stream("code_alpaca"):
        scanned += 1
        instruction = clean_text(str(row.get("instruction") or ""))
        extra = clean_text(str(row.get("input") or ""))
        answer = clean_text(str(row.get("output") or ""))
        if not instruction or not answer or not is_english(instruction):
            rejected_language += 1
            continue
        prompt = instruction if not extra else f"{instruction}\n\nInput:\n{extra}"
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
        if conversation_length(messages, sp) > CONTEXT + 1:
            rejected_context += 1
            continue
        identity = stable_hash(prompt + "\0" + answer)
        if identity in seen:
            continue
        seen.add(identity)
        split = split_for("code-alpaca|" + identity, validation_percent=3)
        target = val_rows if split == "validation" else train_rows
        limit = CODE_VAL if split == "validation" else CODE_TRAIN
        if len(target) >= limit:
            continue
        target.append({
            "messages": messages,
            "source": SOURCES["code_alpaca"]["repo"],
            "domain": "code_instruction",
            "language": "en+code",
            "pair_sha256": identity,
        })
        if len(train_rows) >= CODE_TRAIN and len(val_rows) >= CODE_VAL:
            break

    if len(train_rows) < CODE_TRAIN or len(val_rows) < CODE_VAL:
        raise RuntimeError(
            f"CodeAlpaca target incomplete: {len(train_rows)}/{CODE_TRAIN}, "
            f"{len(val_rows)}/{CODE_VAL}, scanned={scanned}"
        )
    return train_rows, val_rows, {
        "source": SOURCES["code_alpaca"],
        "scanned": scanned,
        "rejected_context": rejected_context,
        "rejected_language": rejected_language,
    }


ultra_train, ultra_val, ultra_evidence = collect_ultrachat()
code_train, code_val, code_evidence = collect_code_alpaca()
train_rows = ultra_train + code_train
val_rows = ultra_val + code_val

train_ids = {row["pair_sha256"] for row in train_rows}
val_ids = {row["pair_sha256"] for row in val_rows}
if len(train_ids) != len(train_rows) or len(val_ids) != len(val_rows):
    raise RuntimeError("duplicate instruction rows detected")
if train_ids & val_ids:
    raise RuntimeError("instruction train/validation leakage detected")

train_rows.sort(key=lambda row: stable_hash("sft-train|" + row["pair_sha256"]))
val_rows.sort(key=lambda row: stable_hash("sft-val|" + row["pair_sha256"]))
train_n, train_sha = write_jsonl(INSTRUCTION_ROOT / "train.jsonl", train_rows)
val_n, val_sha = write_jsonl(INSTRUCTION_ROOT / "validation.jsonl", val_rows)
prov = {
    "schema_version": "staticai.instruction-sft.v2",
    "dataset_id": "STATICAI/STATIC-INSTRUCTION-SFT-V2",
    "recipe": "instruction-sft-v1",
    "language_policy": "English UltraChat plus English CodeAlpaca instructions; source code is retained as code.",
    "train_rows": train_n,
    "validation_rows": val_n,
    "train_sha256": train_sha,
    "validation_sha256": val_sha,
    "mixture": {
        "ultrachat": {"train": len(ultra_train), "validation": len(ultra_val)},
        "code_alpaca": {"train": len(code_train), "validation": len(code_val)},
    },
    "evidence": {"ultrachat": ultra_evidence, "code_alpaca": code_evidence},
}
path = save_provenance("instruction_v1", prov)
print(json.dumps({"status": "INSTRUCTION_READY", "provenance": str(path), **prov}), flush=True)
