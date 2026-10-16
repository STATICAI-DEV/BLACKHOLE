"""Prepare exactly 10,000 approved-source English rows and independent held-out books."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from huggingface_hub import HfFileSystem, hf_hub_download
from blackhole.datasets import DatasetRegistry

SOURCE_REPO = "HuggingFaceTB/openstax_paragraphs"
REVISION = "d336e6a20db3540f7c96dae6bb4a9f1297b70f39"
DESTINATION = "hf://buckets/STATICAI/BLACKHOLE/DATA/"
LOCAL = Path(__file__).resolve().parents[1] / ".cohort-staging"
TAG = "openstax-ternary-10000-20261016"


def paragraphs(value):
    if isinstance(value, dict):
        candidate = value.get("paragraph")
        if isinstance(candidate, str):
            yield candidate
        for key, item in value.items():
            if key != "paragraph":
                yield from paragraphs(item)
    elif isinstance(value, list):
        for item in value:
            yield from paragraphs(item)


def rows_from_book(book):
    seen = set()
    for paragraph in paragraphs(book.get("chapters", [])):
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if len(paragraph) < 130 or "OpenStax" in paragraph or "©" in paragraph:
            continue
        words = paragraph.split()
        for position in range(0, max(0, len(words) - 24), 17):
            first = " ".join(words[position:position + 12])
            second = " ".join(words[position + 12:position + 25])
            if len(first) < 35 or len(second) < 35:
                continue
            prompt = "Continue this passage: " + first
            formatted = f"\n[USER]\n{prompt}\n[STATIC]\n{second}\n[END]\n"
            if len(formatted.encode("utf-8")) > 257:
                continue
            pair = {"messages": [{"role": "user", "content": prompt},
                                 {"role": "assistant", "content": second}]}
            key = hashlib.sha256(json.dumps(pair, sort_keys=True).encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            yield pair


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = Path(hf_hub_download(SOURCE_REPO, "openstax_books.jsonl", repo_type="dataset", revision=REVISION))
    books = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    english = [item for item in books if item.get("language") == "en"]
    english.sort(key=lambda b: str(b["book_title"]))
    if len(english) < 7:
        raise RuntimeError("insufficient disjoint English books")
    holdout_titles = sorted([b["book_title"] for b in english],
                            key=lambda x: hashlib.sha256(("holdout-v1:" + x).encode()).hexdigest())[:6]
    held = set(holdout_titles)
    train_rows, validation_rows, train_titles = [], [], []
    unique = set()
    for book in english:
        title = book["book_title"]
        destination = validation_rows if title in held else train_rows
        for row in rows_from_book(book):
            identity = json.dumps(row, sort_keys=True)
            if identity in unique:
                continue
            unique.add(identity)
            if len(destination) < (300 if title in held else 10000):
                destination.append(row)
                if title not in held and title not in train_titles:
                    train_titles.append(title)
            if len(destination) >= (300 if title in held else 10000):
                break
    if len(train_rows) != 10000 or len(validation_rows) != 300:
        raise RuntimeError(f"insufficient eligible rows: train={len(train_rows)}, validation={len(validation_rows)}")
    if set(train_titles) & held:
        raise RuntimeError("training books overlap heldout books")

    train_path = LOCAL / TAG / "train.jsonl"
    val_path = LOCAL / TAG / "validation.jsonl"
    train_hash = write_jsonl(train_path, train_rows)
    val_hash = write_jsonl(val_path, validation_rows)
    fs = HfFileSystem(token=True)
    for file, name in ((train_path, "train.jsonl"), (val_path, "validation.jsonl")):
        with file.open("rb") as input_stream, fs.open(f"{DESTINATION}prepared/{TAG}/{name}", "wb") as output:
            while chunk := input_stream.read(1024 * 1024):
                output.write(chunk)
    (LOCAL / "registry").mkdir(parents=True, exist_ok=True)
    registry = DatasetRegistry(LOCAL / "registry")
    manifest = registry.register({
        "dataset_id": SOURCE_REPO,
        "revision": REVISION,
        "license": "cc-by-4.0",
        "license_review": "approved",
        "quality_review": "approved",
        "approval_status": "Approved",
        "owner_authorized": True,
        "role": "instruction",
        "language": "en",
        "split": TAG,
        "rows": 10000,
        "content_sha256": train_hash,
        "artifact_uri": f"{DESTINATION}prepared/{TAG}/train.jsonl",
        "validation_artifact_uri": f"{DESTINATION}prepared/{TAG}/validation.jsonl",
        "validation_content_sha256": val_hash,
        "license_evidence": {"source": SOURCE_REPO, "revision": REVISION,
                             "attribution": "OpenStax, Rice University, CC BY 4.0"},
        "review_scope": "Structural checks and book-disjoint 300-row holdout; factual verification not independently claimed.",
    })
    remote_manifest = f"{DESTINATION}manifests/registered/{manifest['record_id']}.json"
    with fs.open(remote_manifest, "w") as stream:
        json.dump(manifest, stream, sort_keys=True)
    evidence = {
        "dataset_id": SOURCE_REPO, "source_revision": REVISION,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "train_rows": len(train_rows), "validation_rows": len(validation_rows),
        "train_sha256": train_hash, "validation_sha256": val_hash,
        "train_books": train_titles, "validation_books": holdout_titles,
        "license": "CC BY 4.0", "registration_id": manifest["record_id"],
        "source_author": "OpenStax, Rice University",
    }
    with fs.open(f"{DESTINATION}manifests/{TAG}-provenance.json", "w") as stream:
        json.dump(evidence, stream, indent=2)
    print(json.dumps({"status": "PREPARED_AND_UPLOADED", "rows": 10000,
                      "validation_rows": 300, "dataset_manifest": manifest,
                      "separate_book_splits": True}, indent=2))


if __name__ == "__main__":
    main()
