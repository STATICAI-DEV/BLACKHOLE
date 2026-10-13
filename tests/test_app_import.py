def test_import_builds_app_without_launching(monkeypatch) -> None:
    import importlib

    import gradio as gr

    launched = False

    def forbidden_launch(*args, **kwargs):
        nonlocal launched
        launched = True
        raise AssertionError("launch must not run during import")

    monkeypatch.setattr(gr.Blocks, "launch", forbidden_launch)
    module = importlib.import_module("app")

    assert isinstance(module.demo, gr.Blocks)
    assert launched is False
