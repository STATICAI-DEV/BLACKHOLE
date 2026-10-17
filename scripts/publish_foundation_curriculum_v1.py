from __future__ import annotations
import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, HfFileSystem
import truststore

from blackhole.datasets import DatasetRegistry

truststore.inject_into_ssl()

ROOT = Path(r"D:\04_BLACKHOLE\Datasets\Foundation_V1")
REPO = Path(r"D:\05_NETWORK\Infrastructure\Local\BLACKHOLE")
REGISTRY_ROOT = ROOT / "registry"
REGISTRY_ROOT.mkdir(parents=True, exist_ok=True)
TOKENIZER = ROOT / "tokenizer" / "static_foundation_tokenizer_v1.model"
ASSET = REPO / "blackhole" / "adapters" / "assets" / "static_foundation_tokenizer_v1.model"
BUCKET = "hf://buckets/STATICAI/BLACKHOLE/DATA"

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

def upload(fs: HfFileSystem, local: Path, remote: str) -> None:
    with local.open("rb") as src, fs.open(remote, "wb") as dst:
        while chunk := src.read(1024 * 1024):
            dst.write(chunk)

api = HfApi()
who = api.whoami(token=True)
if str(who.get("name") or "").casefold() != "staticai":
    raise RuntimeError(f"WRONG_HF_ACCOUNT: {who.get('name')}")
fs = HfFileSystem(token=True)

required = {
    "pretrain": ROOT / "pretrain" / "train.jsonl",
    "pretrain_val": ROOT / "pretrain" / "validation.jsonl",
    "reasoning": ROOT / "reasoning" / "train.jsonl",
    "reasoning_val": ROOT / "reasoning" / "validation.jsonl",
    "software": ROOT / "software_engineering" / "train.jsonl",
    "software_val": ROOT / "software_engineering" / "validation.jsonl",
    "instruction": ROOT / "instruction" / "train.jsonl",
    "instruction_val": ROOT / "instruction" / "validation.jsonl",
    "judgment": ROOT / "judgment" / "train.jsonl",
    "judgment_val": ROOT / "judgment" / "validation.jsonl",
    "preference": ROOT / "judgment" / "preference" / "train.jsonl",
    "preference_val": ROOT / "judgment" / "preference" / "validation.jsonl",
    "pretrain_prov": ROOT / "provenance" / "pretrain_v1.json",
    "reasoning_prov": ROOT / "provenance" / "reasoning_v1.json",
    "software_prov": ROOT / "provenance" / "software_v1.json",
    "instruction_prov": ROOT / "provenance" / "instruction_v1.json",
    "judgment_prov": ROOT / "provenance" / "judgment_v1.json",
    "preference_prov": ROOT / "provenance" / "preference_v1.json",
    "judgment_benchmark": ROOT / "evaluations" / "judgment_v1" / "rewardbench2.jsonl",
    "judgment_benchmark_prov": ROOT / "provenance" / "judgment_benchmark_v1.json",
    "software_benchmark_tasks": ROOT / "evaluations" / "software_v1" / "tasks.jsonl",
    "software_benchmark_answers": ROOT / "evaluations" / "software_v1" / "answer_key.jsonl",
    "software_benchmark_prov": ROOT / "provenance" / "software_benchmark_v1.json",
    "tokenizer": TOKENIZER,
}
missing = [str(path) for path in required.values() if not path.is_file()]
if missing:
    raise RuntimeError("MISSING_PREPARED_ARTIFACTS: " + ", ".join(missing))

ASSET.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(TOKENIZER, ASSET)
tokenizer_sha = sha256(TOKENIZER)
tokenizer_uri = f"{BUCKET}/tokenizers/static_foundation_bpe_v1/tokenizer.model"
upload(fs, TOKENIZER, tokenizer_uri)

stage_specs = [
    {
        "name": "pretrain",
        "dataset_id": "STATICAI/STATIC-FOUNDATION-PRETRAIN-V1",
        "role": "pretraining",
        "recipe": "causal-pretrain-v1",
        "license": "mixed-odc-by-apache-2.0",
        "train": required["pretrain"],
        "validation": required["pretrain_val"],
        "provenance": required["pretrain_prov"],
        "remote": "static_foundation_pretrain_v1",
        "sources": [
            "HuggingFaceFW/fineweb-edu",
            "codeparrot/github-jupyter-code-to-text",
            "HuggingFaceTB/finemath",
        ],
    },
    {
        "name": "reasoning",
        "dataset_id": "STATICAI/STATIC-REASONING-MID-V2",
        "role": "reasoning",
        "recipe": "reasoning-mid-v1",
        "license": "apache-2.0",
        "train": required["reasoning"],
        "validation": required["reasoning_val"],
        "provenance": required["reasoning_prov"],
        "remote": "static_reasoning_mid_v2",
        "sources": ["open-thoughts/OpenThoughts-114k"],
    },
    {
        "name": "software",
        "dataset_id": "STATICAI/STATIC-SOFTWARE-ENGINEERING-SFT-V2",
        "role": "software_engineering",
        "recipe": "software-engineering-sft-v1",
        "license": "mit",
        "train": required["software"],
        "validation": required["software_val"],
        "provenance": required["software_prov"],
        "remote": "static_software_engineering_sft_v2",
        "sources": [
            "SWE-bench/SWE-smith-py",
            "SWE-Gym/SWE-Gym",
        ],
    },
    {
        "name": "instruction",
        "dataset_id": "STATICAI/STATIC-INSTRUCTION-SFT-V2",
        "role": "instruction",
        "recipe": "instruction-sft-v1",
        "license": "mixed-mit-cc-by-4.0",
        "train": required["instruction"],
        "validation": required["instruction_val"],
        "provenance": required["instruction_prov"],
        "remote": "static_instruction_sft_v2",
        "sources": [
            "HuggingFaceH4/ultrachat_200k",
            "sahil2801/CodeAlpaca-20k",
        ],
    },
    {
        "name": "judgment",
        "dataset_id": "STATICAI/STATIC-JUDGMENT-SFT-V1",
        "role": "judgment",
        "recipe": "judgment-sft-v1",
        "license": "cc-by-4.0",
        "train": required["judgment"],
        "validation": required["judgment_val"],
        "provenance": required["judgment_prov"],
        "remote": "static_judgment_sft_v1",
        "sources": ["nvidia/HelpSteer3"],
    },
    {
        "name": "preference",
        "dataset_id": "STATICAI/STATIC-JUDGMENT-PREFERENCE-V3",
        "role": "preference",
        "recipe": "preference-dpo-v1",
        "license": "mixed-cc-by-4.0-apache-2.0",
        "train": required["preference"],
        "validation": required["preference_val"],
        "provenance": required["preference_prov"],
        "remote": "static_judgment_preference_v3",
        "sources": [
            "nvidia/HelpSteer3",
            "lmarena-ai/arena-human-preference-55k",
            "open-r1/OpenR1-Math-220k",
        ],
    },
]

registry = DatasetRegistry(REGISTRY_ROOT)
published = {}
for spec in stage_specs:
    provenance = json.loads(spec["provenance"].read_text(encoding="utf-8"))
    if provenance.get("dataset_id") != spec["dataset_id"]:
        raise RuntimeError(
            f"{spec['name']} provenance dataset identity mismatch: "
            f"{provenance.get('dataset_id')} != {spec['dataset_id']}"
        )
    if provenance.get("recipe") != spec["recipe"]:
        raise RuntimeError(
            f"{spec['name']} provenance recipe mismatch: "
            f"{provenance.get('recipe')} != {spec['recipe']}"
        )
    train_rows = int(provenance["train_rows"])
    validation_rows = int(provenance["validation_rows"])
    if train_rows < 1 or validation_rows < 1:
        raise RuntimeError(f"{spec['name']} has an empty train or validation split")
    train_sha = sha256(spec["train"])
    val_sha = sha256(spec["validation"])
    if train_sha != provenance["train_sha256"] or val_sha != provenance["validation_sha256"]:
        raise RuntimeError(f"{spec['name']} provenance hash mismatch")

    remote_base = f"{BUCKET}/prepared/{spec['remote']}"
    train_uri = remote_base + "/train.jsonl"
    val_uri = remote_base + "/validation.jsonl"
    prov_uri = f"{BUCKET}/manifests/{spec['remote']}_provenance.json"
    upload(fs, spec["train"], train_uri)
    upload(fs, spec["validation"], val_uri)
    upload(fs, spec["provenance"], prov_uri)

    revision = train_sha[:16]
    manifest = registry.register({
        "dataset_id": spec["dataset_id"],
        "revision": revision,
        "license": spec["license"],
        "license_review": "approved",
        "quality_review": "approved",
        "approval_status": "Approved",
        "owner_authorized": True,
        "role": spec["role"],
        "language": "en",
        "split": "train",
        "rows": train_rows,
        "content_sha256": train_sha,
        "artifact_uri": train_uri,
        "validation_artifact_uri": val_uri,
        "validation_content_sha256": val_sha,
        "validation_rows": validation_rows,
        "training_recipe_version": spec["recipe"],
        "tokenizer_name": "STATIC-FOUNDATION-BPE-v1",
        "tokenizer_uri": tokenizer_uri,
        "tokenizer_sha256": tokenizer_sha,
        "context_tokens": 2048,
        "source_datasets": spec["sources"],
        "provenance_uri": prov_uri,
        "language_policy": (
            "English natural-language training only. Source code is retained as code. "
            "Multilingual chat/instruction subsets are excluded."
        ),
    })
    manifest_path = REGISTRY_ROOT / "manifests" / "registered" / f"{manifest['record_id']}.json"
    manifest_uri = f"{BUCKET}/manifests/registered/{manifest['record_id']}.json"
    upload(fs, manifest_path, manifest_uri)
    published[spec["name"]] = {
        "manifest": manifest,
        "manifest_uri": manifest_uri,
        "train_uri": train_uri,
        "validation_uri": val_uri,
    }
    print(json.dumps({
        "stage": "PUBLISHED",
        "name": spec["name"],
        "rows": train_rows,
        "validation_rows": validation_rows,
        "manifest_uri": manifest_uri,
    }), flush=True)

benchmark_uri = f"{BUCKET}/evaluations/rewardbench2_heldout_v1/rewardbench2.jsonl"
benchmark_provenance_uri = f"{BUCKET}/manifests/rewardbench2_heldout_v1_provenance.json"
upload(fs, required["judgment_benchmark"], benchmark_uri)
upload(fs, required["judgment_benchmark_prov"], benchmark_provenance_uri)
benchmark_provenance = json.loads(required["judgment_benchmark_prov"].read_text(encoding="utf-8"))
if benchmark_provenance.get("training_eligible") is not False:
    raise RuntimeError("JUDGMENT_BENCHMARK_MUST_BE_HELD_OUT")
if sha256(required["judgment_benchmark"]) != benchmark_provenance.get("sha256"):
    raise RuntimeError("JUDGMENT_BENCHMARK_HASH_MISMATCH")

software_tasks_uri = f"{BUCKET}/evaluations/swebench_verified_heldout_v1/tasks.jsonl"
software_answers_uri = f"{BUCKET}/evaluations/swebench_verified_heldout_v1/answer_key.jsonl"
software_provenance_uri = f"{BUCKET}/manifests/swebench_verified_heldout_v1_provenance.json"
upload(fs, required["software_benchmark_tasks"], software_tasks_uri)
upload(fs, required["software_benchmark_answers"], software_answers_uri)
upload(fs, required["software_benchmark_prov"], software_provenance_uri)
software_provenance = json.loads(required["software_benchmark_prov"].read_text(encoding="utf-8"))
if software_provenance.get("training_eligible") is not False:
    raise RuntimeError("SOFTWARE_BENCHMARK_MUST_BE_HELD_OUT")
if sha256(required["software_benchmark_tasks"]) != software_provenance.get("tasks_sha256"):
    raise RuntimeError("SOFTWARE_BENCHMARK_TASK_HASH_MISMATCH")
if sha256(required["software_benchmark_answers"]) != software_provenance.get("answer_key_sha256"):
    raise RuntimeError("SOFTWARE_BENCHMARK_ANSWER_HASH_MISMATCH")

summary = {
    "schema_version": "staticai.foundation-curriculum.v2",
    "tokenizer_uri": tokenizer_uri,
    "tokenizer_sha256": tokenizer_sha,
    "stages": published,
    "benchmarks": {
        "judgment": {
            "dataset_id": benchmark_provenance["dataset_id"],
            "artifact_uri": benchmark_uri,
            "provenance_uri": benchmark_provenance_uri,
            "rows": benchmark_provenance["rows"],
            "sha256": benchmark_provenance["sha256"],
            "training_eligible": False,
        },
        "software_engineering": {
            "dataset_id": software_provenance["dataset_id"],
            "tasks_uri": software_tasks_uri,
            "answer_key_uri": software_answers_uri,
            "provenance_uri": software_provenance_uri,
            "rows": software_provenance["rows"],
            "tasks_sha256": software_provenance["tasks_sha256"],
            "answer_key_sha256": software_provenance["answer_key_sha256"],
            "training_eligible": False,
        }
    },
}
summary_path = ROOT / "CURRICULUM_MANIFEST.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
upload(fs, summary_path, f"{BUCKET}/manifests/static_foundation_curriculum_v2.json")
print(json.dumps({"status": "CURRICULUM_PUBLISHED", "summary": summary}, default=str), flush=True)
