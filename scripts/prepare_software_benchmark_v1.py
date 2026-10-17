from __future__ import annotations
import json
from scripts.foundation_data_common import ROOT, SOURCES, clean_text, is_english, stable_hash, stream, write_jsonl, save_provenance

out_root = ROOT / "evaluations" / "software_v1"
tasks = []
answers = []
seen = set()

for raw in stream("swebench_verified", shuffle=False):
    instance_id = str(raw.get("instance_id") or "")
    problem = raw.get("problem_statement")
    repo = str(raw.get("repo") or "")
    base_commit = str(raw.get("base_commit") or "")
    patch = raw.get("patch")
    test_patch = raw.get("test_patch")
    if not instance_id or instance_id in seen:
        continue
    if not isinstance(problem, str) or not problem.strip() or not is_english(problem):
        continue
    if not isinstance(patch, str) or not isinstance(test_patch, str):
        continue
    seen.add(instance_id)
    tasks.append({
        "instance_id": instance_id,
        "repo": repo,
        "base_commit": base_commit,
        "problem_statement": clean_text(problem),
        "difficulty": raw.get("difficulty"),
        "language": "en+python",
        "role": "benchmark_task",
    })
    answers.append({
        "instance_id": instance_id,
        "patch": patch,
        "test_patch": test_patch,
        "FAIL_TO_PASS": raw.get("FAIL_TO_PASS"),
        "PASS_TO_PASS": raw.get("PASS_TO_PASS"),
        "environment_setup_commit": raw.get("environment_setup_commit"),
        "role": "benchmark_answer_key",
    })

if len(tasks) != 500 or len(answers) != 500:
    raise RuntimeError(f"SWE-bench Verified expected 500 rows, got tasks={len(tasks)}, answers={len(answers)}")

task_n, task_sha = write_jsonl(out_root / "tasks.jsonl", tasks)
answer_n, answer_sha = write_jsonl(out_root / "answer_key.jsonl", answers)
prov = {
    "schema_version": "staticai.software-benchmark.v1",
    "dataset_id": "STATICAI/SWEBENCH-VERIFIED-HELDOUT-V1",
    "source": SOURCES["swebench_verified"],
    "role": "benchmark",
    "training_eligible": False,
    "rows": task_n,
    "tasks_sha256": task_sha,
    "answer_key_sha256": answer_sha,
    "notes": (
        "SWE-bench Verified is held out from every training stage. Tasks and answer key are stored "
        "separately; evaluation must score generated patches by repository tests, not text similarity."
    ),
}
path = save_provenance("software_benchmark_v1", prov)
print(json.dumps({"status":"SOFTWARE_BENCHMARK_READY","provenance":str(path),**prov}),flush=True)
