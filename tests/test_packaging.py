from pathlib import Path
import tomllib


def test_torch_is_model_optional_but_pinned_for_space_runtime() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
    core = project["dependencies"]
    model = project["optional-dependencies"]["model"]
    space_requirements = Path("requirements.txt").read_text(encoding="utf-8").splitlines()

    assert not any(item.startswith("torch==") for item in core)
    assert model == ["torch==2.14.1"]
    assert "torch==2.14.1" in space_requirements
