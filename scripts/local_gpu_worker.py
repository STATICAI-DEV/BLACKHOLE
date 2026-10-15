"""Launch the BLACKHOLE Windows GPU worker without opening inbound ports."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Always import BLACKHOLE from the same refreshed snapshot as this worker script.
# Without this, launching an absolute script path from another working directory can
# resolve a stale editable/local installation first.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_repo_root_text = str(_REPO_ROOT)
if _repo_root_text not in sys.path:
    sys.path.insert(0, _repo_root_text)

from blackhole.adapters.registry import AdapterRegistry, build_default_registry
from blackhole.local_jobs import sanitize_remote_value
from blackhole.local_worker import LocalWorker

SPACE_ID = "STATICAI/BLACKHOLE"


def default_local_root() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    if base:
        return Path(base) / "BLACKHOLE_GPU_WORKER"
    return Path.home() / ".blackhole-worker"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true", help="Authenticate and report CUDA; never claim a job")
    mode.add_argument("--execute-training", action="store_true", help="Poll and execute explicitly LOCAL jobs")
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--space-id", default=SPACE_ID)
    parser.add_argument("--local-root", type=Path, default=default_local_root())
    parser.add_argument("--job-id", help="Claim exactly this already-authorized LOCAL job")
    parser.add_argument("--once", action="store_true", help="Poll exactly once and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.poll_seconds < 1:
        raise SystemExit("--poll-seconds must be at least 1")
    if args.job_id and not args.once:
        raise SystemExit("--job-id requires --once so an approved command cannot drift to another job")
    from gradio_client import Client
    from huggingface_hub import get_token
    import truststore

    token = get_token()
    if not token:
        print(json.dumps({"status": "BLOCKED_AUTH", "error": "run hf auth login locally"}))
        return 2
    truststore.inject_into_ssl()
    try:
        client = Client(args.space_id, token=token, verbose=False, download_files=False)
    except Exception as exc:
        print(
            json.dumps(
                {"status": "BLOCKED_CONNECTION", "error": sanitize_remote_value(str(exc))},
                sort_keys=True,
            )
        )
        return 2
    adapters = AdapterRegistry() if args.check_only else build_default_registry(device="cuda:0")
    worker = LocalWorker(client, adapters, args.local_root)
    if args.check_only:
        result = worker.check_only()
        print(json.dumps(sanitize_remote_value(result), indent=2, sort_keys=True))
        return 0 if result.get("status") == "PASS" else 2

    print(json.dumps({"status": "READY", "mode": "EXECUTE_TRAINING", "space": args.space_id}))
    if args.once:
        result = worker.poll_once(execute_training=True, job_id=args.job_id)
        print(json.dumps(sanitize_remote_value(result), sort_keys=True))
        outcome = result.get("state") or result.get("status")
        return 0 if outcome in {"COMPLETED", "IDLE"} else 2
    try:
        while True:
            result = worker.poll_once(execute_training=True)
            print(json.dumps(sanitize_remote_value(result), sort_keys=True))
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print(json.dumps({"status": "STOPPED", "reason": "operator_interrupt"}))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
