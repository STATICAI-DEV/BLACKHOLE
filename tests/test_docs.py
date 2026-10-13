from pathlib import Path

import yaml


REQUIRED_DOCS = {
    "LAB_CHARTER.md": ("empty weights", "authorization"),
    "ARCHITECTURE.md": ("/data", "/lab", "security"),
    "EXPERIMENT_PROTOCOL.md": ("acceptance", "failure"),
    "DATASET_REGISTER.md": ("5,240", "289", "Afrikaans"),
    "TRAINING_RUNBOOK.md": ("resume", "sleep"),
    "EVALUATION_PROTOCOL.md": ("held-out", "same prompts"),
    "GOVERNANCE_PROVENANCE.md": ("mutable", "EVENTS"),
    "MODEL_CARD_TEMPLATE.md": ("parameter", "limitations"),
    "CHANGELOG.md": ("2026-10-10", "NOT STARTED"),
}


def test_required_stable_docs_are_populated() -> None:
    for name, phrases in REQUIRED_DOCS.items():
        path = Path("docs") / name
        content = path.read_text(encoding="utf-8")
        assert len(content) > 300, name
        for phrase in phrases:
            assert phrase.lower() in content.lower(), f"{name}: missing {phrase}"


def test_readme_has_valid_gradio_space_metadata_and_truthful_status() -> None:
    content = Path("README.md").read_text(encoding="utf-8")
    assert content.startswith("---\n")
    metadata = yaml.safe_load(content.split("---", 2)[1])
    assert metadata["sdk"] == "gradio"
    assert metadata["app_file"] == "app.py"
    assert "PRIVATE SPACE RUNNING" in content
    assert "BASELINE TRAINING: 3 completed runs" in content
    assert "STATIC-0T" in content
    assert "STATIC-0X" in content
    assert "STATIC-0K" in content


def test_local_worker_docs_preserve_explicit_gate_and_cost_boundary() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    runbook = Path("docs/TRAINING_RUNBOOK.md").read_text(encoding="utf-8")
    architecture = Path("docs/ARCHITECTURE.md").read_text(encoding="utf-8")
    combined = "\n".join((readme, runbook, architecture))

    assert "powershell -ExecutionPolicy Bypass -File .\\scripts\\start_local_gpu_worker.ps1 -CheckOnly" in combined
    assert "powershell -ExecutionPolicy Bypass -File .\\scripts\\start_local_gpu_worker.ps1 -ExecuteTraining" in combined
    assert "Ctrl+C" in combined
    assert "LOCAL" in combined and "CLOUD" in combined
    assert "no CPU fallback" in combined
    assert "cpu-basic" in combined
    assert "no paid Hugging Face GPU" in combined
    assert "no inbound" in combined.lower()
