from __future__ import annotations
import json
from pathlib import Path
from scripts.foundation_data_common import ROOT, SOURCES, clean_text, is_english, stable_hash, stream, write_jsonl, save_provenance

out_root = ROOT / "evaluations" / "judgment_v1"
rows = []
seen = set()
for raw in stream("rewardbench2"):
    prompt = raw.get("prompt")
    chosen = raw.get("chosen")
    rejected = raw.get("rejected")
    if not isinstance(prompt, str) or not is_english(prompt):
        continue
    if not isinstance(chosen, list) or not isinstance(rejected, list) or not chosen or not rejected:
        continue
    chosen_clean = [clean_text(str(x)) for x in chosen if isinstance(x, str) and x.strip()]
    rejected_clean = [clean_text(str(x)) for x in rejected if isinstance(x, str) and x.strip()]
    if not chosen_clean or not rejected_clean:
        continue
    identity = str(raw.get("id") or stable_hash(prompt))
    if identity in seen:
        continue
    seen.add(identity)
    rows.append({
        "id": identity,
        "prompt": clean_text(prompt),
        "chosen": chosen_clean,
        "rejected": rejected_clean,
        "subset": raw.get("subset"),
        "source": SOURCES["rewardbench2"]["repo"],
        "language": "en",
        "role": "benchmark",
    })

count, digest = write_jsonl(out_root / "rewardbench2.jsonl", rows)
prov = {
    "schema_version": "staticai.judgment-benchmark.v1",
    "dataset_id": "STATICAI/REWARDBENCH2-HELDOUT-V1",
    "source": SOURCES["rewardbench2"],
    "role": "benchmark",
    "training_eligible": False,
    "rows": count,
    "sha256": digest,
    "notes": "Held out from all training stages. Used only for judgment/reward evaluation.",
}
path = save_provenance("judgment_benchmark_v1", prov)
print(json.dumps({"status":"JUDGMENT_BENCHMARK_READY","provenance":str(path),**prov}),flush=True)
