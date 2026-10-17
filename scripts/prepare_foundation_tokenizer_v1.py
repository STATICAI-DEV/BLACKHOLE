from __future__ import annotations
import json
from collections import Counter
import sentencepiece as spm

from scripts.foundation_data_common import (
    TOKENIZER_ROOT, VOCAB_SIZE, SOURCES, clean_text, stream,
)

TOKENIZER_ROOT.mkdir(parents=True, exist_ok=True)
corpus = TOKENIZER_ROOT / "tokenizer_corpus.txt"
targets = {"fineweb": 14000, "code": 6000, "finemath": 6000}
written = Counter()

if not corpus.is_file() or corpus.stat().st_size < 10_000_000:
    with corpus.open("w", encoding="utf-8", newline="\n") as handle:
        for source, target in targets.items():
            for row in stream(source, shuffle=False):
                if source == "fineweb":
                    text = row.get("text")
                    if row.get("language") not in (None, "en"):
                        continue
                elif source == "code":
                    text = row.get("content")
                else:
                    text = row.get("text")
                    if row.get("language") != "en":
                        continue
                    score = row.get("language_score")
                    if isinstance(score, (int, float)) and score < 0.75:
                        continue
                if not isinstance(text, str):
                    continue
                text = clean_text(text)
                if len(text) < 300:
                    continue
                handle.write(text[:16000].replace("\n", " <0x0A> ") + "\n")
                written[source] += 1
                if written[source] >= target:
                    break
            print(json.dumps({"stage": "TOKENIZER_SOURCE", "source": source, "rows": written[source]}), flush=True)

model = TOKENIZER_ROOT / "static_foundation_tokenizer_v1.model"
vocab = TOKENIZER_ROOT / "static_foundation_tokenizer_v1.vocab"
if not model.is_file() or not vocab.is_file():
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(TOKENIZER_ROOT / "static_foundation_tokenizer_v1"),
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=1.0,
        byte_fallback=True,
        pad_id=0,
        unk_id=1,
        bos_id=-1,
        eos_id=-1,
        user_defined_symbols=["<|eos|>", "<|user|>", "<|assistant|>"],
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        split_digits=True,
        input_sentence_size=500000,
        shuffle_input_sentence=True,
        train_extremely_large_corpus=True,
    )

processor = spm.SentencePieceProcessor(model_file=str(model))
if processor.get_piece_size() != VOCAB_SIZE:
    raise RuntimeError(f"tokenizer vocab mismatch: {processor.get_piece_size()}")
print(json.dumps({
    "status": "TOKENIZER_READY",
    "path": str(model),
    "vocab_size": processor.get_piece_size(),
    "sources": SOURCES,
}), flush=True)
