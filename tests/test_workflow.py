from pathlib import Path

import yaml


def test_status_workflow_is_manual_only_and_narrowly_scoped() -> None:
    path = Path(".github/workflows/blackhole-status.yml")
    raw = path.read_text(encoding="utf-8")
    workflow = yaml.load(raw, Loader=yaml.BaseLoader)

    assert set(workflow["on"]) == {"workflow_dispatch"}
    assert workflow["permissions"] == {"contents": "write"}
    assert "schedule:" not in raw
    assert "HF_READ_TOKEN" in raw
    assert "secrets.HF_READ_TOKEN" in raw
    assert "permissions:" not in raw.split("jobs:", 1)[1]
    collect = next(step for step in workflow["jobs"]["snapshot"]["steps"] if step.get("id") == "collect")
    assert collect["continue-on-error"] == "true"
    assert raw.index("Commit changed snapshot") < raw.index("Fail after preserving blocked snapshot")
