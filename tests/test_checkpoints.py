import hashlib
import json
from pathlib import Path


def _checkpoint(tmp_path: Path, *, adapter: str = "static-0t", trained: bool = True) -> Path:
    root = tmp_path / "checkpoint-10"
    root.mkdir()
    files = {
        "model": "model.bin",
        "optimizer": "optimizer.pt",
        "scheduler": "scheduler.pt",
        "random_state": "random.pt",
    }
    hashes: dict[str, str] = {}
    for key, filename in files.items():
        content = f"{key}-content".encode()
        (root / filename).write_bytes(content)
        hashes[key] = hashlib.sha256(content).hexdigest()
    manifest = {
        "schema_version": "blackhole.checkpoint.v1",
        "checkpoint_id": "checkpoint-10",
        "run_id": "run-001",
        "model_adapter": adapter,
        "trained": trained,
        "step": 10,
        "files": files,
        "sha256": hashes,
    }
    (root / "CHECKPOINT_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_complete_trained_checkpoint_is_chat_and_resume_ready(tmp_path: Path) -> None:
    from blackhole.checkpoints import CheckpointStore

    result = CheckpointStore().inspect(_checkpoint(tmp_path), expected_adapter="static-0t")

    assert result.valid is True
    assert result.chat_ready is True
    assert result.resume_ready is True
    assert result.code == "READY"


def test_checkpoint_hash_or_adapter_mismatch_is_rejected(tmp_path: Path) -> None:
    from blackhole.checkpoints import CheckpointStore

    root = _checkpoint(tmp_path)
    (root / "model.bin").write_bytes(b"tampered")
    result = CheckpointStore().inspect(root, expected_adapter="other-adapter")

    assert result.valid is False
    assert result.chat_ready is False
    assert result.resume_ready is False
    assert result.code == "CHECKPOINT_MISMATCH"
    assert any("adapter" in error for error in result.errors)
    assert any("hash" in error for error in result.errors)


def test_missing_optimizer_state_blocks_exact_resume_only(tmp_path: Path) -> None:
    from blackhole.checkpoints import CheckpointStore

    root = _checkpoint(tmp_path)
    (root / "optimizer.pt").unlink()
    result = CheckpointStore().inspect(root, expected_adapter="static-0t")

    assert result.valid is False
    assert result.resume_ready is False
    assert result.code == "RESUME_UNVERIFIED"


def test_checkpoint_rejects_traversal_and_missing_manifest_identity(tmp_path: Path) -> None:
    from blackhole.checkpoints import CheckpointStore

    root = _checkpoint(tmp_path)
    manifest_path = root / "CHECKPOINT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("run_id")
    manifest["files"]["model"] = "../outside.bin"
    (tmp_path / "outside.bin").write_bytes(b"model-content")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = CheckpointStore().inspect(root)
    assert result.chat_ready is False
    assert any("run_id" in error for error in result.errors)
    assert any("unsafe" in error for error in result.errors)
