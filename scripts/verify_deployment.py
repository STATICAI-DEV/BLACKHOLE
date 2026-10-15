"""Read-only BLACKHOLE deployment verification with an optional restart probe."""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import UTC, datetime
from blackhole.redaction import redact_text

SPACE_ID = "STATICAI/BLACKHOLE"
STORAGE_BUCKET = "STATICAI/BLACKHOLE"
LAB_SOURCE = f"{STORAGE_BUCKET}/LAB"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--restart-probe",
        action="store_true",
        help="Write one unique lab-bucket probe, safely restart the Space, reread it and remove only that probe",
    )
    return parser


def _clean(value: object) -> str:
    return redact_text(value, limit=240)


def _configured_volumes(info: object, runtime: object) -> list[dict[str, object]]:
    info_runtime = getattr(info, "runtime", None)
    configured = getattr(info_runtime, "volumes", None)
    if configured is None:
        configured = getattr(runtime, "volumes", None)
    return [
        {
            "source": getattr(volume, "source", None),
            "path": getattr(volume, "path", None),
            "mount_path": getattr(volume, "mount_path", None),
            "read_only": getattr(volume, "read_only", None),
        }
        for volume in (configured or [])
    ]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from huggingface_hub import HfApi, HfFileSystem, bucket_info, get_token

    if not get_token():
        print("BLOCKED_AUTH: run `hf auth login` locally; never paste a token into chat")
        return 2
    api = HfApi(token=True)
    identity = api.whoami(token=True)
    if str(identity.get("name") or "").casefold() != "staticai".casefold():
        print("BLOCKED_AUTH: authenticated Hugging Face user is not STATICAI")
        return 2
    info = api.space_info(SPACE_ID, token=True)
    runtime = api.get_space_runtime(SPACE_ID, token=True)
    volumes = _configured_volumes(info, runtime)
    expected = {(STORAGE_BUCKET, "DATA", "/data"), (STORAGE_BUCKET, "LAB", "/lab")}
    observed = {(item["source"], item["path"], item["mount_path"]) for item in volumes}
    bucket = bucket_info(STORAGE_BUCKET, token=True)
    result = {
        "observed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "space": SPACE_ID,
        "private": bool(info.private),
        "revision": info.sha,
        "stage": str(runtime.stage),
        "hardware": runtime.hardware,
        "volumes": volumes,
        "volume_contract_ok": observed == expected,
        "buckets_private": bool(bucket.private),
        "restart_probe": "NOT_RUN",
    }
    if args.restart_probe:
        filesystem = HfFileSystem(token=True)
        probe = f"hf://buckets/{LAB_SOURCE}/governance/provenance/deployment-probe-{uuid.uuid4().hex}.txt"
        payload = uuid.uuid4().hex
        try:
            with filesystem.open(probe, "w") as handle:
                handle.write(payload)
            with filesystem.open(probe, "r") as handle:
                before = handle.read()
            api.restart_space(SPACE_ID, token=True)
            api.wait_for_space(SPACE_ID, token=True)
            with filesystem.open(probe, "r") as handle:
                after = handle.read()
            result["restart_probe"] = "PASS" if before == payload == after else "FAIL"
        finally:
            filesystem.rm(probe)
    try:
        logs = list(api.fetch_space_logs(SPACE_ID, token=True))[-20:]
        result["boot_log_status"] = "AVAILABLE"
        result["boot_log_extract"] = [_clean(line) for line in logs]
    except Exception as exc:
        # Log-reading permission is not required to verify runtime, storage or mounts.
        result["boot_log_status"] = "UNAVAILABLE_PERMISSION_NONCRITICAL"
        result["boot_log_extract"] = None
        result["boot_log_detail"] = _clean(exc)
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["private"] and result["volume_contract_ok"] and result["buckets_private"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
