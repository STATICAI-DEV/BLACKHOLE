"""Train fresh STATIC-X/T/K through the approved foundation curriculum.

Stages per model:
  1. causal pretraining from random weights
  2. reasoning mid-training from stage-1 weights with a fresh optimizer
  3. software-engineering SFT from stage-2 weights with a fresh optimizer
  4. general instruction SFT from stage-3 weights with a fresh optimizer
  5. business/commerce SFT from stage-4 weights with a fresh optimizer
  6. corporate-finance SFT from stage-5 weights with a fresh optimizer
  7. normal relationship/persona + context-gated adult social SFT from stage-6 weights
  8. browser-agent/research SFT from stage-7 weights with a fresh optimizer
  9. ENTJ-style executive persona SFT from stage-8 weights with a fresh optimizer
 10. ethics/business-integrity SFT from stage-9 weights with a fresh optimizer
 11. judgment/critique SFT from stage-10 weights with a fresh optimizer
 12. direct preference optimization from stage-11 weights against a frozen stage-start reference

All compute runs on the authenticated local CUDA worker. This script is intentionally
sequential because three 147M training states do not belong in an 8 GB laptop GPU at once.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import truststore
from gradio_client import Client
from huggingface_hub import HfApi, HfFileSystem, get_token

truststore.inject_into_ssl()

OWNER = "STATICAI"
SPACE = "STATICAI/BLACKHOLE"
CURRICULUM_URI = "hf://buckets/STATICAI/BLACKHOLE/DATA/manifests/static_foundation_curriculum_v5.json"
WORKER = Path(r"D:\05_NETWORK\Infrastructure\Local\BLACKHOLE\scripts\local_gpu_worker.py")
OUT = Path(r"D:\04_BLACKHOLE\Training")
PYTHON = Path(sys.executable)

MODELS = (
    ("STATIC-T", 1234),
    ("STATIC-X", 2401),
    ("STATIC-K", 2402),
)

STAGE_CONFIG = {
    "pretrain": {
        "recipe": "causal-pretrain-v1",
        "learning_rate": 3e-4,
        "weight_decay": 0.1,
        "warmup_steps": 500,
        "min_lr_ratio": 0.1,
        "checkpoint_interval": 5000,
        "evaluation_interval": 5000,
    },
    "reasoning": {
        "recipe": "reasoning-mid-v1",
        "learning_rate": 1e-4,
        "weight_decay": 0.1,
        "warmup_steps": 200,
        "min_lr_ratio": 0.1,
        "checkpoint_interval": 2500,
        "evaluation_interval": 1000,
    },
    "software": {
        "recipe": "software-engineering-sft-v1",
        "learning_rate": 7e-5,
        "weight_decay": 0.05,
        "warmup_steps": 150,
        "min_lr_ratio": 0.1,
        "checkpoint_interval": 2000,
        "evaluation_interval": 1000,
    },
    "instruction": {
        "recipe": "instruction-sft-v1",
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "warmup_steps": 100,
        "min_lr_ratio": 0.1,
        "checkpoint_interval": 2500,
        "evaluation_interval": 1000,
    },
    "business": {
        "recipe": "business-sft-v1",
        "learning_rate": 3e-5,
        "weight_decay": 0.01,
        "warmup_steps": 100,
        "min_lr_ratio": 0.1,
        "checkpoint_interval": 1500,
        "evaluation_interval": 750,
    },
    "finance": {
        "recipe": "finance-sft-v1",
        "learning_rate": 3e-5,
        "weight_decay": 0.01,
        "warmup_steps": 100,
        "min_lr_ratio": 0.1,
        "checkpoint_interval": 1500,
        "evaluation_interval": 750,
    },
    "relationship": {
        "recipe": "relationship-sft-v1",
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_steps": 100,
        "min_lr_ratio": 0.1,
        "checkpoint_interval": 1500,
        "evaluation_interval": 750,
    },
    "browser": {
        "recipe": "browser-agent-sft-v1",
        "learning_rate": 3e-5,
        "weight_decay": 0.01,
        "warmup_steps": 75,
        "min_lr_ratio": 0.1,
        "checkpoint_interval": 750,
        "evaluation_interval": 375,
    },
    "executive": {
        "recipe": "executive-persona-sft-v1",
        "learning_rate": 1.5e-5,
        "weight_decay": 0.01,
        "warmup_steps": 50,
        "min_lr_ratio": 0.1,
        "checkpoint_interval": 500,
        "evaluation_interval": 250,
    },
    "ethics": {
        "recipe": "ethics-sft-v1",
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_steps": 100,
        "min_lr_ratio": 0.1,
        "checkpoint_interval": 1500,
        "evaluation_interval": 750,
    },
    "judgment": {
        "recipe": "judgment-sft-v1",
        "learning_rate": 3e-5,
        "weight_decay": 0.01,
        "warmup_steps": 100,
        "min_lr_ratio": 0.1,
        "checkpoint_interval": 2000,
        "evaluation_interval": 1000,
    },
    "preference": {
        "recipe": "preference-dpo-v1",
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "warmup_steps": 100,
        "min_lr_ratio": 0.2,
        "checkpoint_interval": 1000,
        "evaluation_interval": 500,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "dpo_beta": 0.1,
    },
}

MICRO_BATCH = 2
ACCUMULATION = 4
EFFECTIVE_BATCH = MICRO_BATCH * ACCUMULATION
CONTEXT = 2048
VALIDATION_ROWS = 128
DATA_ORDER_SEED = 20_261_017
STAGE_ORDER = ("pretrain", "reasoning", "software", "instruction", "business", "finance", "relationship", "browser", "executive", "ethics", "judgment", "preference")


def save(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(fs: HfFileSystem, uri: str) -> dict[str, Any]:
    with fs.open(uri, "r") as stream:
        return json.load(stream)


def execute_worker(job_id: str) -> None:
    subprocess.run(
        [
            str(PYTHON),
            str(WORKER),
            "--execute-training",
            "--once",
            "--job-id",
            job_id,
        ],
        check=True,
    )


def main() -> None:
    api = HfApi()
    who = api.whoami(token=True)
    if str(who.get("name") or "").casefold() != OWNER.casefold():
        raise RuntimeError(f"WRONG_HF_ACCOUNT: {who.get('name')}")
    if not WORKER.is_file():
        raise RuntimeError("LOCAL_WORKER_MISSING")

    fs = HfFileSystem(token=True)
    curriculum = read_json(fs, CURRICULUM_URI)
    stages = curriculum.get("stages")
    expected_stages = set(STAGE_ORDER)
    if not isinstance(stages, dict) or set(stages) != expected_stages:
        raise RuntimeError("INVALID_CURRICULUM_MANIFEST")

    manifests: dict[str, dict[str, Any]] = {}
    for stage_name in STAGE_ORDER:
        stage_info = stages[stage_name]
        manifest_uri = stage_info.get("manifest_uri")
        if not isinstance(manifest_uri, str):
            raise RuntimeError(f"{stage_name} manifest URI missing")
        manifest = read_json(fs, manifest_uri)
        expected_recipe = STAGE_CONFIG[stage_name]["recipe"]
        if (
            manifest.get("approval_status") != "Approved"
            or manifest.get("eligible_for_training") is not True
            or manifest.get("language") != "en"
            or manifest.get("training_recipe_version") != expected_recipe
            or manifest.get("context_tokens") != CONTEXT
        ):
            raise RuntimeError(f"INVALID_{stage_name.upper()}_MANIFEST")
        manifests[stage_name] = manifest

    receipt_path = OUT / (
        datetime.now().strftime("%Y-%m-%d_%H%M")
        + "_STATIC_FOUNDATION_GENERATION_V1_RECEIPT.json"
    )
    if receipt_path.exists():
        raise RuntimeError(f"RECEIPT_ALREADY_EXISTS: {receipt_path}")

    receipt: dict[str, Any] = {
        "schema_version": "staticai.foundation-generation.v3",
        "owner": OWNER,
        "space": SPACE,
        "architecture": "STATIC-FOUNDATION-TERNARY-v1",
        "parameters_per_model": 147_284_736,
        "context_tokens": CONTEXT,
        "micro_batch": MICRO_BATCH,
        "gradient_accumulation": ACCUMULATION,
        "effective_batch": EFFECTIVE_BATCH,
        "controlled_comparison": {
            "same_architecture": True,
            "same_dataset_hashes": True,
            "same_stage_order": list(STAGE_ORDER),
            "same_data_order_seed": DATA_ORDER_SEED,
            "same_training_environment": True,
            "intentional_difference": "model initialization seed only",
        },
        "models": {},
        "created_at": datetime.now().astimezone().isoformat(),
        "state": "STARTING",
    }
    save(receipt_path, receipt)

    subprocess.run([str(PYTHON), str(WORKER), "--check-only"], check=True)
    client = Client(SPACE, token=get_token(), verbose=False, download_files=False)

    for model, seed in MODELS:
        model_receipt = {"seed": seed, "stages": {}}
        receipt["models"][model] = model_receipt
        previous_checkpoint = None
        global_step = 0

        for stage_name in STAGE_ORDER:
            manifest = manifests[stage_name]
            stage_cfg = STAGE_CONFIG[stage_name]
            rows = int(manifest["rows"])
            stage_micro_batch = int(stage_cfg.get("batch_size", MICRO_BATCH))
            stage_accumulation = int(stage_cfg.get("gradient_accumulation_steps", ACCUMULATION))
            stage_effective_batch = stage_micro_batch * stage_accumulation
            if rows % stage_effective_batch:
                raise RuntimeError(
                    f"{stage_name} rows {rows} are not divisible by effective batch {stage_effective_batch}"
                )
            steps = rows // stage_effective_batch
            stamp = datetime.now().strftime("%Y%m%d%H%M%S")
            run_id = f"{model}-FOUNDATION-V1-{stage_name.upper()}-{stamp}"
            config: dict[str, Any] = {
                "execution_target": "LOCAL",
                "job_type": "TRAIN",
                "run_id": run_id,
                "model_adapter": model,
                "seed": seed,
                "train_steps": steps,
                "batch_size": stage_micro_batch,
                "gradient_accumulation_steps": stage_accumulation,
                "context_length": CONTEXT,
                "learning_rate": stage_cfg["learning_rate"],
                "weight_decay": stage_cfg["weight_decay"],
                "warmup_steps": min(stage_cfg["warmup_steps"], max(steps - 1, 0)),
                "min_lr_ratio": stage_cfg["min_lr_ratio"],
                "checkpoint_interval": min(stage_cfg["checkpoint_interval"], steps),
                "evaluation_interval": min(stage_cfg["evaluation_interval"], steps),
                "validation_max_rows": VALIDATION_ROWS,
                "dataset_position": 0,
                "dataset_epoch": 0,
                "shuffle_seed": DATA_ORDER_SEED,
                "training_recipe_version": stage_cfg["recipe"],
                "resume_step": global_step,
                "reset_optimizer_on_resume": previous_checkpoint is not None,
            }
            if "dpo_beta" in stage_cfg:
                config["dpo_beta"] = float(stage_cfg["dpo_beta"])
            if previous_checkpoint is not None:
                config["resume_checkpoint_uri"] = previous_checkpoint

            queued = client.predict(
                {
                    "config": config,
                    "dataset_manifest": manifest,
                    "owner_authorized": True,
                },
                api_name="/local_job_enqueue",
            )
            if queued.get("state") != "QUEUED":
                raise RuntimeError(f"{model} {stage_name} queue rejected: {queued!r}")
            job_id = str(queued["job_id"])
            model_receipt["stages"][stage_name] = {
                "state": "QUEUED",
                "job_id": job_id,
                "run_id": run_id,
                "rows": rows,
                "steps": steps,
                "recipe": stage_cfg["recipe"],
                "micro_batch": stage_micro_batch,
                "gradient_accumulation": stage_accumulation,
                "effective_batch": stage_effective_batch,
            }
            save(receipt_path, receipt)
            print(json.dumps({
                "stage": "QUEUED",
                "model": model,
                "curriculum_stage": stage_name,
                "job_id": job_id,
                "run_id": run_id,
                "steps": steps,
                "rows": rows,
            }), flush=True)

            execute_worker(job_id)
            remote_run = f"hf://buckets/STATICAI/BLACKHOLE/LAB/runs/{run_id}"
            checkpoint = remote_run + "/checkpoint-final"
            checkpoint_manifest = read_json(fs, checkpoint + "/CHECKPOINT_MANIFEST.json")
            run_manifest = read_json(fs, remote_run + "/RUN_MANIFEST.json")
            global_step += steps
            if (
                checkpoint_manifest.get("trained") is not True
                or checkpoint_manifest.get("model_adapter") != model
                or checkpoint_manifest.get("step") != global_step
                or checkpoint_manifest.get("dataset_position") != rows
                or run_manifest.get("result") != "completed"
            ):
                raise RuntimeError(f"{model} {stage_name} completion evidence is invalid")
            previous_checkpoint = checkpoint
            model_receipt["stages"][stage_name] = {
                **model_receipt["stages"][stage_name],
                "state": "COMPLETED",
                "checkpoint": checkpoint,
                "global_step": global_step,
                "actual_metrics": run_manifest.get("actual_metrics"),
            }
            save(receipt_path, receipt)
            print(json.dumps({
                "stage": "COMPLETED",
                "model": model,
                "curriculum_stage": stage_name,
                "global_step": global_step,
                "checkpoint": checkpoint,
                "metrics": run_manifest.get("actual_metrics"),
            }), flush=True)

        model_receipt["state"] = "COMPLETED"
        model_receipt["final_checkpoint"] = previous_checkpoint
        save(receipt_path, receipt)

    receipt["state"] = "COMPLETED"
    receipt["completed_at"] = datetime.now().astimezone().isoformat()
    save(receipt_path, receipt)
    print(json.dumps({
        "status": "GENERATION_COMPLETED",
        "receipt": str(receipt_path),
        "models": {
            name: payload.get("final_checkpoint")
            for name, payload in receipt["models"].items()
        },
    }), flush=True)


if __name__ == "__main__":
    main()
