"""Shared helpers for STATIC foundation curriculum preparation."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import sentencepiece as spm
import truststore
from datasets import load_dataset
from langdetect import DetectorFactory, LangDetectException, detect

truststore.inject_into_ssl()
DetectorFactory.seed = 0

ROOT = Path(r"D:\04_BLACKHOLE\Datasets\Foundation_V1")
TOKENIZER_ROOT = ROOT / "tokenizer"
PRETRAIN_ROOT = ROOT / "pretrain"
REASONING_ROOT = ROOT / "reasoning"
SOFTWARE_ROOT = ROOT / "software_engineering"
INSTRUCTION_ROOT = ROOT / "instruction"
BUSINESS_ROOT = ROOT / "business"
RELATIONSHIP_ROOT = ROOT / "relationship"
BROWSER_ROOT = ROOT / "browser_agent"
JUDGMENT_ROOT = ROOT / "judgment"
PROVENANCE_ROOT = ROOT / "provenance"
VOCAB_SIZE = 32_000
CONTEXT = 2_048
SEED = 20_261_017

SOURCES = {
    "fineweb": {
        "repo": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT", "split": "train",
        "license": "odc-by", "language": "en",
    },
    "code": {
        "repo": "codeparrot/github-jupyter-code-to-text",
        "config": "default", "split": "train",
        "license": "apache-2.0", "language": "en",
    },
    "finemath": {
        "repo": "HuggingFaceTB/finemath",
        "config": "finemath-3plus", "split": "train",
        "license": "odc-by", "language": "en",
    },
    "reasoning": {
        "repo": "open-thoughts/OpenThoughts-114k",
        "config": "default", "split": "train",
        "license": "apache-2.0", "language": "en-filtered",
    },
    "code_alpaca": {
        "repo": "sahil2801/CodeAlpaca-20k",
        "config": "default", "split": "train",
        "license": "cc-by-4.0", "language": "en",
    },
    "ultrachat": {
        "repo": "HuggingFaceH4/ultrachat_200k",
        "config": "default", "split": "train_sft",
        "license": "mit", "language": "en",
    },
    "swe_smith": {
        "repo": "SWE-bench/SWE-smith-py",
        "config": "default", "split": "train",
        "license": "mit", "language": "en",
    },
    "swe_gym": {
        "repo": "SWE-Gym/SWE-Gym",
        "config": "default", "split": "train",
        "license": "mit", "language": "en",
    },
    "helpsteer_preference": {
        "repo": "nvidia/HelpSteer3",
        "config": "preference", "split": "train",
        "license": "cc-by-4.0", "language": "en-filtered",
    },
    "helpsteer_edit": {
        "repo": "nvidia/HelpSteer3",
        "config": "edit", "split": "train",
        "license": "cc-by-4.0", "language": "en-filtered",
    },
    "helpsteer_feedback": {
        "repo": "nvidia/HelpSteer3",
        "config": "feedback", "split": "train",
        "license": "cc-by-4.0", "language": "en-filtered",
    },
    "helpsteer_principle": {
        "repo": "nvidia/HelpSteer3",
        "config": "principle", "split": "train",
        "license": "cc-by-4.0", "language": "en-filtered",
    },
    "rewardbench2": {
        "repo": "allenai/reward-bench-2",
        "config": "default", "split": "test",
        "license": "odc-by", "language": "en",
    },
    "swebench_verified": {
        "repo": "princeton-nlp/SWE-bench_Verified",
        "config": "default", "split": "test",
        "license": "benchmark-source-repository-licenses", "language": "en+python",
    },
    "arena_preference": {
        "repo": "lmarena-ai/arena-human-preference-55k",
        "config": "default", "split": "train",
        "license": "apache-2.0", "language": "en",
    },
    "openr1_reasoning_preference": {
        "repo": "open-r1/OpenR1-Math-220k",
        "config": "default", "split": "train",
        "license": "apache-2.0", "language": "en",
    },
    "finqa_business": {
        "repo": "bevaya/FinQA",
        "config": "default", "split": "train",
        "license": "cc-by-4.0", "language": "en",
    },
    "persona_chat": {
        "repo": "google/Synthetic-Persona-Chat",
        "config": "default", "split": "train",
        "license": "cc-by-4.0", "language": "en",
    },
    "browser_sft_train": {
        "repo": "saital/browser-agent-phase1-sft-reasoning-action",
        "config": "default", "split": "train",
        "license": "mit", "language": "en",
    },
    "browser_sft_val": {
        "repo": "saital/browser-agent-phase1-sft-reasoning-action",
        "config": "default", "split": "val",
        "license": "mit", "language": "en",
    },
    "research_qa": {
        "repo": "jxg25/EntiWeave",
        "config": "default", "split": "train",
        "license": "cc-by-4.0", "language": "en",
    },
}

def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()

def split_for(key: str, validation_percent: int = 2) -> str:
    bucket = int(stable_hash("split-v1:" + key)[:8], 16) % 100
    return "validation" if bucket < validation_percent else "train"

def clean_text(value: str) -> str:
    value = value.replace("\x00", "")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"\n{5,}", "\n\n\n\n", value)
    return value.strip()

def is_english(value: str) -> bool:
    sample = re.sub(r"[\W_]+", " ", value)
    sample = " ".join(sample.split())[:4000]
    if len(sample) < 24:
        return True
    try:
        return detect(sample) == "en"
    except LangDetectException:
        return False

def stream(source: str, *, shuffle: bool = True, buffer_size: int = 1000):
    meta = SOURCES[source]
    ds = load_dataset(meta["repo"], meta["config"], split=meta["split"], streaming=True)
    return ds.shuffle(seed=SEED, buffer_size=buffer_size) if shuffle else ds

def tokenizer() -> spm.SentencePieceProcessor:
    path = TOKENIZER_ROOT / "static_foundation_tokenizer_v1.model"
    if not path.is_file():
        raise RuntimeError("foundation tokenizer has not been prepared")
    sp = spm.SentencePieceProcessor(model_file=str(path))
    if sp.get_piece_size() != VOCAB_SIZE:
        raise RuntimeError("foundation tokenizer vocabulary mismatch")
    return sp


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    with path.open("wb") as handle:
        for row in rows:
            data = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            handle.write(data)
            digest.update(data)
            count += 1
    return count, digest.hexdigest()

def conversation_length(messages: list[dict[str, str]], sp) -> int:
    # A 2,048-token example cannot plausibly contain this much UTF-8 text.
    # Reject pathological long reasoning traces before expensive SentencePiece work.
    if sum(len(message.get("content", "")) for message in messages) > 12_000:
        return CONTEXT + 2
    user_id = sp.piece_to_id("<|user|>")
    assistant_id = sp.piece_to_id("<|assistant|>")
    eos_id = sp.piece_to_id("<|eos|>")
    sequence: list[int] = []
    for message in messages:
        sequence.append(user_id if message["role"] == "user" else assistant_id)
        sequence.extend(sp.encode("\n" + message["content"].strip() + "\n", out_type=int))
    sequence.append(eos_id)
    return len(sequence)

def normalize_open_thought(row: dict[str, Any]) -> list[dict[str, str]] | None:
    raw = row.get("conversations")
    if not isinstance(raw, list):
        return None
    human = None
    answer = None
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = item.get("from")
        value = item.get("value")
        if not isinstance(value, str) or not value.strip():
            continue
        if role in {"human", "user"} and human is None:
            human = clean_text(value)
        elif role in {"gpt", "assistant"} and human is not None:
            answer = clean_text(value)
            break
    if not human or not answer or not is_english(human):
        return None
    return [
        {"role": "user", "content": human},
        {"role": "assistant", "content": answer},
    ]


def last_user_text(context: Any) -> str | None:
    if not isinstance(context, list):
        return None
    for item in reversed(context):
        if not isinstance(item, dict):
            continue
        if item.get("role") == "user":
            content = item.get("content")
            if isinstance(content, str) and content.strip():
                return clean_text(content)
    return None


def normalize_messages(row: dict[str, Any], field: str = "messages") -> list[dict[str, str]] | None:
    raw = row.get(field)
    if not isinstance(raw, list):
        return None
    result: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str) or not content.strip():
            continue
        result.append({"role": role, "content": clean_text(content)})
    if len(result) < 2 or not any(item["role"] == "assistant" for item in result):
        return None
    first_user = next((item["content"] for item in result if item["role"] == "user"), None)
    if not first_user or not is_english(first_user):
        return None
    return result

def token_chunks(text: str, sp, min_tokens: int = 128, max_tokens: int = CONTEXT):
    ids = sp.encode(text, out_type=int)
    if not 128 <= max_tokens <= CONTEXT:
        raise ValueError("max_tokens must be between 128 and model context")
    for start in range(0, len(ids), max_tokens):
        part = ids[start:start + max_tokens]
        if len(part) < min_tokens:
            continue
        decoded = sp.decode(part).strip()
        if len(decoded) >= 250:
            yield decoded

def save_provenance(name: str, payload: dict[str, Any]) -> Path:
    PROVENANCE_ROOT.mkdir(parents=True, exist_ok=True)
    path = PROVENANCE_ROOT / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
