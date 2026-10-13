import hashlib
import json
from pathlib import Path


def test_atomic_write_json_replaces_content_and_leaves_no_temp_file(tmp_path: Path) -> None:
    from blackhole.atomic import atomic_write_json

    target = tmp_path / "nested" / "record.json"
    atomic_write_json(target, {"state": "first"})
    atomic_write_json(target, {"state": "final", "unknown": None})

    assert json.loads(target.read_text(encoding="utf-8")) == {
        "state": "final",
        "unknown": None,
    }
    assert list(target.parent.glob(f".{target.name}.*.tmp")) == []


def test_sha256_file_matches_standard_digest(tmp_path: Path) -> None:
    from blackhole.atomic import sha256_file

    target = tmp_path / "artifact.bin"
    target.write_bytes(b"BLACKHOLE\n")

    assert sha256_file(target) == hashlib.sha256(b"BLACKHOLE\n").hexdigest()
