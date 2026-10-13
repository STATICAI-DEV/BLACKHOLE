"""Persisted held-out evaluation records and paired comparisons."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from .atomic import atomic_write_json
from .config import require_storage_root

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class EvaluationStore:
    def __init__(self, lab_root: Path | str) -> None:
        self.lab_root = Path(lab_root)
        self.root = Path(lab_root) / "evaluations"

    def _path(self, evaluation_id: str) -> Path:
        if not _ID.fullmatch(evaluation_id):
            raise ValueError("evaluation_id must contain only safe identifier characters")
        return self.root / evaluation_id / "EVALUATION_RESULTS.json"

    def save(self, evaluation_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        require_storage_root(self.lab_root, "lab")
        record = dict(payload)
        if not record.get("prompt_set_version"):
            raise ValueError("prompt_set_version is required")
        if not isinstance(record.get("results"), list) or not record["results"]:
            raise ValueError("evaluation results are required")
        prompt_ids: list[str] = []
        for result in record["results"]:
            if not isinstance(result, dict) or not all(
                key in result for key in ("prompt_id", "category", "prompt", "output", "duration_seconds")
            ):
                raise ValueError("each evaluation result requires prompt, output, category and duration")
            prompt_ids.append(str(result["prompt_id"]))
        if len(set(prompt_ids)) != len(prompt_ids):
            raise ValueError("evaluation prompt IDs must be unique")
        record["evaluation_id"] = evaluation_id
        record["schema_version"] = "blackhole.evaluation.v1"
        atomic_write_json(self._path(evaluation_id), record)
        return record

    def load(self, evaluation_id: str) -> dict[str, Any]:
        return json.loads(self._path(evaluation_id).read_text(encoding="utf-8"))

    def compare(self, first_id: str, second_id: str) -> dict[str, Any]:
        first = self.load(first_id)
        second = self.load(second_id)
        first_ids = [item["prompt_id"] for item in first["results"]]
        second_ids = [item["prompt_id"] for item in second["results"]]
        same_prompt_set = (
            first.get("prompt_set_version") == second.get("prompt_set_version")
            and first_ids == second_ids
        )
        if not same_prompt_set:
            raise ValueError("checkpoint comparison requires the same prompts in the same order")
        return {
            "schema_version": "blackhole.evaluation-comparison.v1",
            "first_evaluation_id": first_id,
            "second_evaluation_id": second_id,
            "same_prompt_set": True,
            "pairs": [
                {"prompt_id": left["prompt_id"], "first": left, "second": right}
                for left, right in zip(first["results"], second["results"], strict=True)
            ],
        }
