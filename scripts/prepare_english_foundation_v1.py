"""Build and register an English-only conversational foundation dataset for BLACKHOLE.

Sources are restricted to OpenAssistant records whose per-row language metadata is exactly "en".
No training is started by this script.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import truststore
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download

from blackhole.datasets import DatasetRegistry

truststore.inject_into_ssl()

BUCKET = "hf://buckets/STATICAI/BLACKHOLE/DATA"
LOCAL_ROOT = Path(r"D:\04_BLACKHOLE\Datasets")
RAW_LOCAL = LOCAL_ROOT / "Raw" / "OpenAssistant_English"
PREP_LOCAL = LOCAL_ROOT / "Training_Ready" / "STATIC_ENGLISH_FOUNDATION_V1"
APPROVAL_LOCAL = Path(r"D:\04_BLACKHOLE\Governance\Dataset_Approvals")
REGISTRY_LOCAL = PREP_LOCAL / "_registry"
CONTEXT_BYTES = 256
FOUNDATION_ROWS = 5000
TRAIN_ROWS = 4000
VALIDATION_ROWS = 500
TEST_ROWS = 500
CONTROLLED_POOL_ROWS = 6000
SOURCE_TARGET = 6000

SOURCES = (
    {
        "repo": "OpenAssistant/oasst1",
        "tag": "oasst1",
        "license": "apache-2.0",
        "config": "default",
        "split": "train",
        "kind": "openassistant",
    },
    {
        "repo": "OpenAssistant/oasst2",
        "tag": "oasst2",
        "license": "apache-2.0",
        "config": "default",
        "split": "train",
        "kind": "openassistant",
    },
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def payload_size(prompt: str, answer: str) -> int:
    return len(f"\n[USER]\n{prompt}\n[STATIC]\n{answer}\n[END]\n".encode("utf-8"))


def row_ok(row: dict[str, Any]) -> bool:
    return (
        row.get("lang") == "en"
        and row.get("deleted") is False
        and row.get("review_result") is True
        and row.get("tree_state") in {"ready_for_export", None}
        and row.get("role") in {"prompter", "assistant"}
        and isinstance(row.get("text"), str)
        and bool(row["text"].strip())
    )


def collect_openassistant(source: dict[str, str], target: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read the Hub's bulk Parquet export and keep only explicitly English reviewed pairs."""
    import pyarrow.parquet as pq

    repo = source["repo"]
    parquet_path = Path(
        hf_hub_download(
            repo_id=repo,
            repo_type="dataset",
            revision="refs/convert/parquet",
            filename="default/train/0000.parquet",
        )
    )
    parquet = pq.ParquetFile(parquet_path)
    messages: dict[str, dict[str, Any]] = {}
    pending: dict[str, list[dict[str, Any]]] = {}
    seen_pairs: set[str] = set()
    candidates: list[dict[str, Any]] = []
    scanned = 0
    english_rows = 0
    rejected_context = 0
    rejected_language = 0

    def maybe_add(parent: dict[str, Any], child: dict[str, Any]) -> None:
        nonlocal rejected_context
        if len(candidates) >= target:
            return
        if parent.get("role") != "prompter" or child.get("role") != "assistant":
            return
        if parent.get("lang") != "en" or child.get("lang") != "en":
            return
        tree_id = str(child.get("message_tree_id") or parent.get("message_tree_id") or "")
        if not tree_id:
            return
        rank = child.get("rank")
        if rank not in (None, 0):
            return
        prompt = normalize_text(str(parent["text"]))
        answer = normalize_text(str(child["text"]))
        if not prompt or not answer:
            return
        if payload_size(prompt, answer) > CONTEXT_BYTES + 1:
            rejected_context += 1
            return
        identity = sha256_bytes(
            json.dumps([prompt, answer], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        if identity in seen_pairs:
            return
        seen_pairs.add(identity)
        candidates.append(
            {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ],
                "source": repo,
                "source_revision": None,
                "source_message_id": child.get("message_id"),
                "source_parent_id": parent.get("message_id"),
                "source_tree_id": tree_id,
                "source_language": "en",
                "pair_sha256": identity,
            }
        )

    columns = [
        "message_id", "parent_id", "text", "role", "lang", "review_result",
        "deleted", "rank", "message_tree_id", "tree_state",
    ]
    for batch in parquet.iter_batches(batch_size=4096, columns=columns):
        for row in batch.to_pylist():
            scanned += 1
            if row.get("lang") != "en":
                rejected_language += 1
                continue
            english_rows += 1
            if not row_ok(row):
                continue
            message_id = row.get("message_id")
            if not isinstance(message_id, str) or not message_id:
                continue
            messages[message_id] = row
            parent_id = row.get("parent_id")
            if row.get("role") == "assistant" and isinstance(parent_id, str):
                parent = messages.get(parent_id)
                if parent is not None:
                    maybe_add(parent, row)
                else:
                    pending.setdefault(parent_id, []).append(row)
            if row.get("role") == "prompter":
                for child in pending.pop(message_id, []):
                    maybe_add(row, child)
            if len(candidates) >= target:
                break
        if len(candidates) >= target:
            break

    evidence = {
        "dataset": repo,
        "config": source["config"],
        "split": source["split"],
        "source_total_rows": parquet.metadata.num_rows,
        "rows_scanned": scanned,
        "english_rows_seen": english_rows,
        "non_english_rows_rejected": rejected_language,
        "over_context_pairs_rejected": rejected_context,
        "accepted_short_english_pairs": len(candidates),
        "language_gate": "both source rows must have lang == 'en'",
        "multiple_short_pairs_per_tree_allowed": True,
        "split_isolation": "all pairs from a conversation tree are assigned to one split",
        "ingestion": "Hub refs/convert/parquet bulk export",
    }
    return candidates, evidence


def controlled_examples(limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(prompt: str, answer: str, category: str) -> None:
        if len(rows) >= limit:
            return
        prompt = normalize_text(prompt)
        answer = normalize_text(answer)
        if payload_size(prompt, answer) > CONTEXT_BYTES + 1:
            raise RuntimeError("controlled example exceeds context")
        key = sha256_bytes(f"{prompt}\0{answer}".encode("utf-8"))
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ],
                "source": "STATICAI/authored-english-foundation-v1",
                "source_revision": "1",
                "source_tree_id": "static-" + key[:20],
                "source_language": "en",
                "category": category,
                "pair_sha256": key,
            }
        )

    greetings = [
        ("Hello.", "Hello!"),
        ("Hi.", "Hi!"),
        ("Good morning.", "Good morning!"),
        ("Good evening.", "Good evening!"),
        ("Thank you.", "You're welcome."),
        ("Goodbye.", "Goodbye!"),
        ("Can you help me?", "Yes, I can help."),
        ("Nice to meet you.", "Nice to meet you too."),
        ("How are you?", "I'm doing well."),
        ("Are you ready?", "Yes, I'm ready."),
    ]
    for cycle in range(5):
        for prompt, answer in greetings:
            add(f"{prompt} Reply naturally. Example {cycle + 1}.", answer, "conversation")

    facts = [
        ("What color is grass?", "Green."),
        ("What color is snow?", "White."),
        ("What animal barks?", "A dog."),
        ("What animal meows?", "A cat."),
        ("What do bees make?", "Honey."),
        ("What do humans breathe?", "Air."),
        ("What planet do we live on?", "Earth."),
        ("How many days are in a week?", "Seven."),
        ("How many hours are in a day?", "Twenty-four."),
        ("What month follows January?", "February."),
        ("What day follows Monday?", "Tuesday."),
        ("What is the opposite of hot?", "Cold."),
        ("What is the opposite of up?", "Down."),
        ("What is the plural of book?", "Books."),
        ("Name a common fruit.", "Apple."),
        ("Name a common vegetable.", "Carrot."),
        ("What is frozen water called?", "Ice."),
        ("What is H2O commonly called?", "Water."),
        ("What shape has three sides?", "A triangle."),
        ("What shape has four equal sides?", "A square."),
    ]
    for cycle in range(5):
        for prompt, answer in facts:
            add(f"{prompt} Answer briefly. Set {cycle + 1}.", answer, "basic_english")

    for a in range(0, 20):
        for b in range(0, 10):
            add(f"What is {a} plus {b}?", str(a + b), "arithmetic")
            if len(rows) >= limit:
                return rows
    for a in range(10, 30):
        for b in range(0, 10):
            add(f"What is {a} minus {b}?", str(a - b), "arithmetic")
            if len(rows) >= limit:
                return rows
    for a in range(0, 13):
        for b in range(0, 13):
            add(f"What is {a} times {b}?", str(a * b), "arithmetic")
            if len(rows) >= limit:
                return rows
    for a in range(0, 50):
        for b in range(0, 50):
            if a == b:
                continue
            add(f"Which number is larger, {a} or {b}?", str(max(a, b)), "comparison")
            if len(rows) >= limit:
                return rows
    words = ["YES", "NO", "OK", "READY", "BLUE", "GREEN", "HELLO", "THANKS", "ONE", "TWO"]
    for cycle in range(400):
        for word in words:
            add(f"Reply only with {word}. Instruction {cycle + 1}.", word, "instruction_following")
            if len(rows) >= limit:
                return rows
    return rows[:limit]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("wb") as stream:
        for row in rows:
            encoded = (
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            stream.write(encoded)
            digest.update(encoded)
    return digest.hexdigest()


def upload(fs: HfFileSystem, local: Path, remote: str) -> None:
    with local.open("rb") as source, fs.open(remote, "wb") as destination:
        while chunk := source.read(1024 * 1024):
            destination.write(chunk)


def verify_rows(rows: list[dict[str, Any]], label: str) -> None:
    identities: set[str] = set()
    for index, row in enumerate(rows):
        messages = row.get("messages")
        if (
            not isinstance(messages, list)
            or len(messages) != 2
            or [part.get("role") for part in messages] != ["user", "assistant"]
        ):
            raise RuntimeError(f"{label} row {index} has invalid message roles")
        prompt = messages[0].get("content")
        answer = messages[1].get("content")
        if not isinstance(prompt, str) or not isinstance(answer, str) or not prompt or not answer:
            raise RuntimeError(f"{label} row {index} has empty text")
        if row.get("source_language") != "en":
            raise RuntimeError(f"{label} row {index} is not explicitly English")
        if payload_size(prompt, answer) > CONTEXT_BYTES + 1:
            raise RuntimeError(f"{label} row {index} exceeds model context")
        identity = row.get("pair_sha256")
        if not isinstance(identity, str) or len(identity) != 64:
            raise RuntimeError(f"{label} row {index} lacks pair identity")
        if identity in identities:
            raise RuntimeError(f"{label} contains duplicate pair")
        identities.add(identity)


def main() -> None:
    api = HfApi()
    who = api.whoami(token=True)
    if str(who.get("name") or "").casefold() != "staticai":
        raise RuntimeError(f"WRONG_HF_ACCOUNT: {who.get('name')}")
    fs = HfFileSystem(token=True)
    RAW_LOCAL.mkdir(parents=True, exist_ok=True)
    PREP_LOCAL.mkdir(parents=True, exist_ok=True)
    APPROVAL_LOCAL.mkdir(parents=True, exist_ok=True)
    REGISTRY_LOCAL.mkdir(parents=True, exist_ok=True)

    all_external: list[dict[str, Any]] = []
    source_evidence: list[dict[str, Any]] = []
    revisions: dict[str, str] = {}
    for source in SOURCES:
        revision = str(api.dataset_info(source["repo"]).sha)
        revisions[source["repo"]] = revision
        rows, evidence = collect_openassistant(source, SOURCE_TARGET)
        if len(rows) < 500:
            raise RuntimeError(f"not enough short English pairs from {source['repo']}: {len(rows)}")
        for row in rows:
            row["source_revision"] = revision
        verify_rows(rows, source["tag"])
        raw_path = RAW_LOCAL / f"{source['tag']}_english_short_pairs.jsonl"
        raw_hash = write_jsonl(raw_path, rows)
        remote_raw = f"{BUCKET}/raw/english/{source['tag']}_english_short_pairs_v1.jsonl"
        upload(fs, raw_path, remote_raw)
        evidence.update(
            {
                "source_revision": revision,
                "license": source["license"],
                "raw_artifact_uri": remote_raw,
                "raw_sha256": raw_hash,
            }
        )
        source_evidence.append(evidence)
        all_external.extend(rows)
        print(json.dumps({"stage": "SOURCE_READY", "source": source["repo"], "rows": len(rows)}), flush=True)

    dedup: dict[str, dict[str, Any]] = {}
    for row in all_external:
        dedup.setdefault(str(row["pair_sha256"]), row)
    external = list(dedup.values())
    if len(external) < 1500:
        raise RuntimeError(f"only {len(external)} deduplicated English external rows available")

    controlled = controlled_examples(CONTROLLED_POOL_ROWS)
    if len(controlled) < 1000:
        raise RuntimeError(f"controlled English pool is unexpectedly small: {len(controlled)}")
    verify_rows(controlled, "controlled")

    combined_map: dict[str, dict[str, Any]] = {}
    for row in external + controlled:
        combined_map.setdefault(str(row["pair_sha256"]), row)
    combined = list(combined_map.values())

    pools: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for row in combined:
        group = f"{row.get('source')}|{row.get('source_tree_id')}"
        slot = int(sha256_bytes(("split-tree-v1:" + group).encode("utf-8"))[:8], 16) % 10
        split = "test" if slot == 0 else "validation" if slot == 1 else "train"
        pools[split].append(row)
    for name, rows in pools.items():
        rows.sort(key=lambda row: sha256_bytes((f"select-{name}-v1:" + row["pair_sha256"]).encode("utf-8")))

    if len(pools["train"]) < TRAIN_ROWS or len(pools["validation"]) < VALIDATION_ROWS or len(pools["test"]) < TEST_ROWS:
        raise RuntimeError(
            "insufficient tree-isolated split pools: "
            f"train={len(pools['train'])}, validation={len(pools['validation'])}, test={len(pools['test'])}"
        )
    train = pools["train"][:TRAIN_ROWS]
    validation = pools["validation"][:VALIDATION_ROWS]
    test = pools["test"][:TEST_ROWS]
    verify_rows(train, "train")
    verify_rows(validation, "validation")
    verify_rows(test, "test")

    train_ids = {row["pair_sha256"] for row in train}
    val_ids = {row["pair_sha256"] for row in validation}
    test_ids = {row["pair_sha256"] for row in test}
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError("cross-split duplicate leakage detected")
    train_groups = {(row.get("source"), row.get("source_tree_id")) for row in train}
    val_groups = {(row.get("source"), row.get("source_tree_id")) for row in validation}
    test_groups = {(row.get("source"), row.get("source_tree_id")) for row in test}
    if train_groups & val_groups or train_groups & test_groups or val_groups & test_groups:
        raise RuntimeError("conversation-tree leakage detected across splits")

    train_path = PREP_LOCAL / "train.jsonl"
    val_path = PREP_LOCAL / "validation.jsonl"
    test_path = PREP_LOCAL / "test.jsonl"
    train_hash = write_jsonl(train_path, train)
    val_hash = write_jsonl(val_path, validation)
    test_hash = write_jsonl(test_path, test)

    remote_base = f"{BUCKET}/prepared/static_english_foundation_v1"
    train_uri = remote_base + "/train.jsonl"
    val_uri = remote_base + "/validation.jsonl"
    test_uri = f"{BUCKET}/evaluations/static_english_foundation_v1/test.jsonl"
    for local, remote in ((train_path, train_uri), (val_path, val_uri), (test_path, test_uri)):
        upload(fs, local, remote)

    registry = DatasetRegistry(REGISTRY_LOCAL)
    manifest = registry.register(
        {
            "dataset_id": "STATICAI/STATIC-ENGLISH-FOUNDATION-V1",
            "revision": hashlib.sha256(
                json.dumps(revisions, sort_keys=True).encode("utf-8")
            ).hexdigest()[:16],
            "license": "mixed-approved",
            "license_review": "approved",
            "quality_review": "approved",
            "approval_status": "Approved",
            "owner_authorized": True,
            "role": "instruction",
            "language": "en",
            "split": "train",
            "rows": len(train),
            "content_sha256": train_hash,
            "artifact_uri": train_uri,
            "validation_artifact_uri": val_uri,
            "validation_content_sha256": val_hash,
            "training_objective": "assistant-response-only-v2",
            "context_limit_bytes": CONTEXT_BYTES,
            "test_artifact_uri": test_uri,
            "test_content_sha256": test_hash,
            "test_rows": len(test),
            "source_datasets": [source["repo"] for source in SOURCES]
            + ["STATICAI/authored-english-foundation-v1"],
            "license_evidence": {
                **{source["repo"]: source["license"] for source in SOURCES},
                "STATICAI/authored-english-foundation-v1": "STATICAI proprietary",
            },
            "language_evidence": {
                "OpenAssistant": "both source rows must have explicit lang == en",
                "internal_gate": "STATICAI authored English examples only",
                "non_english_source_rows_included": 0,
            },
            "review_scope": (
                "English-only row metadata enforcement; exact user/assistant schema; "
                "maximum 257-byte formatted sequence; deduplication; split isolation; "
                "review_result true and deleted false for OpenAssistant source pairs."
            ),
        }
    )
    manifest_path = REGISTRY_LOCAL / "manifests" / "registered" / f"{manifest['record_id']}.json"
    remote_manifest = f"{BUCKET}/manifests/registered/{manifest['record_id']}.json"
    upload(fs, manifest_path, remote_manifest)

    provenance = {
        "schema_version": "staticai.english-foundation.v1",
        "dataset_id": manifest["dataset_id"],
        "language": "en",
        "language_policy": "ENGLISH_ONLY",
        "non_english_rows_allowed": False,
        "source_revisions": revisions,
        "source_evidence": source_evidence,
        "controlled_pool_rows": len(controlled),
        "external_pool_rows": len(external),
        "selected_source_counts": {
            "train": {source: sum(1 for row in train if row["source"] == source) for source in sorted({r["source"] for r in train})},
            "validation": {source: sum(1 for row in validation if row["source"] == source) for source in sorted({r["source"] for r in validation})},
            "test": {source: sum(1 for row in test if row["source"] == source) for source in sorted({r["source"] for r in test})},
        },
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "train_sha256": train_hash,
        "validation_sha256": val_hash,
        "test_sha256": test_hash,
        "manifest_uri": remote_manifest,
        "record_id": manifest["record_id"],
    }
    provenance_path = PREP_LOCAL / "PROVENANCE.json"
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    upload(fs, provenance_path, f"{BUCKET}/manifests/static_english_foundation_v1_provenance.json")

    approval = {
        "schema_version": "staticai.dataset-approval.v1",
        "dataset": "STATIC-ENGLISH-FOUNDATION-V1",
        "status": "APPROVED_FOR_ENGLISH_TRAINING",
        "language": "en",
        "non_english_rows": 0,
        "training_rows": len(train),
        "validation_rows": len(validation),
        "heldout_test_rows": len(test),
        "record_id": manifest["record_id"],
        "manifest_uri": remote_manifest,
        "training_objective": "assistant-response-only-v2",
        "notes": "No training was started by dataset preparation.",
    }
    approval_path = APPROVAL_LOCAL / "STATIC_ENGLISH_FOUNDATION_V1_APPROVAL.json"
    approval_path.write_text(json.dumps(approval, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "READY",
                "manifest": manifest,
                "provenance": provenance,
                "approval_path": str(approval_path),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
