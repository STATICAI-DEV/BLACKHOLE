from __future__ import annotations

import json

from scripts.foundation_data_common import (
    CONTEXT,
    JUDGMENT_ROOT,
    SOURCES,
    clean_text,
    conversation_length,
    is_english,
    last_user_text,
    split_for,
    stable_hash,
    stream,
    tokenizer,
    write_jsonl,
    save_provenance,
)

TARGETS = {
    "helpsteer_preference": (20_000, 500),
    "helpsteer_edit": (6_000, 200),
    "helpsteer_feedback": (8_000, 200),
    "helpsteer_principle": (4_000, 100),
}

sp = tokenizer()
all_train: list[dict] = []
all_val: list[dict] = []
evidence: dict[str, dict] = {}


def add_candidate(
    *,
    source: str,
    key: str,
    prompt: str,
    answer: str,
    domain: str,
    train_rows: list[dict],
    val_rows: list[dict],
    train_target: int,
    val_target: int,
    seen: set[str],
) -> tuple[bool, str | None]:
    prompt = clean_text(prompt)
    answer = clean_text(answer)
    if not prompt or not answer or not is_english(prompt):
        return False, "language"
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]
    if conversation_length(messages, sp) > CONTEXT + 1:
        return False, "context"
    identity = stable_hash(source + "|" + key + "|" + prompt + "\0" + answer)
    if identity in seen:
        return False, "duplicate"
    seen.add(identity)
    split = split_for(source + "|" + key, validation_percent=3)
    target = val_rows if split == "validation" else train_rows
    limit = val_target if split == "validation" else train_target
    if len(target) >= limit:
        return False, "full"
    target.append({
        "messages": messages,
        "source": SOURCES[source]["repo"],
        "source_subset": source,
        "domain": domain,
        "language": "en",
        "pair_sha256": identity,
    })
    return True, None


for source, (train_target, val_target) in TARGETS.items():
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    seen: set[str] = set()
    scanned = 0
    rejected_context = 0
    rejected_language = 0

    for row in stream(source):
        scanned += 1
        domain = str(row.get("domain") or "general")
        if domain.casefold() == "multilingual":
            rejected_language += 1
            continue
        context = row.get("context")
        task = last_user_text(context)
        if not task or not is_english(task):
            rejected_language += 1
            continue

        candidates: list[tuple[str, str, str]] = []

        if source == "helpsteer_preference":
            response1 = row.get("response1")
            response2 = row.get("response2")
            preference = row.get("overall_preference")
            prefs = row.get("individual_preference")
            if (
                isinstance(response1, str)
                and isinstance(response2, str)
                and isinstance(preference, int)
                and preference != 0
            ):
                reasoning = ""
                if isinstance(prefs, list):
                    reasons = [
                        clean_text(str(item.get("reasoning") or ""))
                        for item in prefs
                        if isinstance(item, dict) and item.get("reasoning")
                    ]
                    reasoning = next((item for item in reasons if item and is_english(item)), "")
                winner = "A" if preference < 0 else "B"
                if not reasoning:
                    reasoning = (
                        "The preferred response better satisfies the user's request and should be selected "
                        "over the alternative."
                    )
                prompt = (
                    f"Task:\n{task}\n\nResponse A:\n{clean_text(response1)}\n\n"
                    f"Response B:\n{clean_text(response2)}\n\n"
                    "Judge which response is better. Consider correctness, completeness, relevance, "
                    "clarity, and whether the response actually solves the user's problem. "
                    "Choose A or B and justify the judgment."
                )
                answer = f"Reasoning: {reasoning}\nJudgment: Response {winner} is better."
                candidates.append(("preference", prompt, answer))

        elif source == "helpsteer_edit":
            original = row.get("original_response")
            edited = row.get("edited_response")
            feedback = row.get("feedback")
            if isinstance(original, str) and isinstance(edited, str):
                feedback_items = [
                    clean_text(str(item))
                    for item in (feedback if isinstance(feedback, list) else [])
                    if isinstance(item, str) and item.strip()
                ][:3]
                prompt = (
                    f"Task:\n{task}\n\nOriginal response:\n{clean_text(original)}\n\n"
                    "Reviewer feedback:\n- " + "\n- ".join(feedback_items or ["Improve correctness and usefulness."]) +
                    "\n\nRewrite the response so it resolves the substantive problems identified by the reviewers."
                )
                candidates.append(("edit", prompt, clean_text(edited)))

        elif source == "helpsteer_feedback":
            for index in (1, 2):
                response = row.get(f"response{index}")
                feedback = row.get(f"feedback{index}")
                if not isinstance(response, str) or not isinstance(feedback, list):
                    continue
                feedback_items = [
                    clean_text(str(item))
                    for item in feedback
                    if isinstance(item, str) and item.strip()
                ][:3]
                if not feedback_items:
                    continue
                prompt = (
                    f"Task:\n{task}\n\nCandidate response:\n{clean_text(response)}\n\n"
                    "Act as a rigorous reviewer. Identify the important strengths, errors, omissions, "
                    "and reasoning failures in the candidate response."
                )
                answer = "Review:\n- " + "\n- ".join(feedback_items)
                candidates.append((f"feedback{index}", prompt, answer))

        elif source == "helpsteer_principle":
            response = row.get("response")
            principle = row.get("principle")
            fulfilment = row.get("fulfilment")
            if all(isinstance(item, str) and item.strip() for item in (response, principle, fulfilment)):
                prompt = (
                    f"Task:\n{task}\n\nCandidate response:\n{clean_text(response)}\n\n"
                    f"Judgment criterion: {clean_text(principle)}\n\n"
                    "Decide whether the response satisfies this criterion. Give the judgment first."
                )
                answer = f"Judgment: {clean_text(fulfilment)}."
                candidates.append(("principle", prompt, answer))

        for suffix, prompt, answer in candidates:
            added, reason = add_candidate(
                source=source,
                key=f"{scanned}:{suffix}",
                prompt=prompt,
                answer=answer,
                domain=domain,
                train_rows=train_rows,
                val_rows=val_rows,
                train_target=train_target,
                val_target=val_target,
                seen=seen,
            )
            if not added and reason == "context":
                rejected_context += 1
            elif not added and reason == "language":
                rejected_language += 1
            if len(train_rows) >= train_target and len(val_rows) >= val_target:
                break

        if len(train_rows) >= train_target and len(val_rows) >= val_target:
            break
        if scanned % 5000 == 0:
            print(json.dumps({
                "stage": "JUDGMENT_PROGRESS",
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
    }
    print(json.dumps({
        "stage": "JUDGMENT_SOURCE_READY",
        "source": source,
        "train": len(train_rows),
        "validation": len(val_rows),
    }), flush=True)

train_ids = {row["pair_sha256"] for row in all_train}
val_ids = {row["pair_sha256"] for row in all_val}
if len(train_ids) != len(all_train) or len(val_ids) != len(all_val):
    raise RuntimeError("duplicate judgment rows detected")
if train_ids & val_ids:
    raise RuntimeError("judgment train/validation leakage detected")

all_train.sort(key=lambda row: stable_hash("judgment-train|" + row["pair_sha256"]))
all_val.sort(key=lambda row: stable_hash("judgment-val|" + row["pair_sha256"]))
train_n, train_sha = write_jsonl(JUDGMENT_ROOT / "train.jsonl", all_train)
val_n, val_sha = write_jsonl(JUDGMENT_ROOT / "validation.jsonl", all_val)
prov = {
    "schema_version": "staticai.judgment-sft.v1",
    "dataset_id": "STATICAI/STATIC-JUDGMENT-SFT-V1",
    "recipe": "judgment-sft-v1",
    "language_policy": (
        "English natural-language prompts and reviewer reasoning only. Code-domain examples are "
        "retained only when the task and reviewer rationale pass English detection."
    ),
    "train_rows": train_n,
    "validation_rows": val_n,
    "train_sha256": train_sha,
    "validation_sha256": val_sha,
    "mixture": {key: {"train": v[0], "validation": v[1]} for key, v in TARGETS.items()},
    "evidence": evidence,
    "judgment_dimensions": [
        "correctness",
        "completeness",
        "relevance",
        "clarity",
        "critique",
        "error repair",
        "principle adherence",
    ],
}
prov_path = save_provenance("judgment_v1", prov)
print(json.dumps({"status": "JUDGMENT_READY", "provenance": str(prov_path), **prov}), flush=True)
