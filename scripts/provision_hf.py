"""Dry-run-first private Hugging Face provisioning for BLACKHOLE."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

OWNER = "STATICAI"
SPACE_ID = "STATICAI/BLACKHOLE"
STORAGE_BUCKET = "STATICAI/BLACKHOLE"


class BlockedAuth(RuntimeError):
    pass


class ProvisioningConflict(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class VolumeSpec:
    source: str
    mount_path: str
    path: str | None = None
    read_only: bool = False


@dataclass(frozen=True, slots=True)
class ProvisioningPlan:
    apply: bool
    authenticated_user: str
    create_buckets: tuple[str, ...]
    create_space: bool
    replace_volumes: bool
    volumes: tuple[VolumeSpec, ...]
    upload_space_files: bool
    repo_root: str
    hardware_change: None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_plan(
    *,
    authenticated_user: str,
    bucket_states: dict[str, bool],
    space: dict[str, Any] | None,
    current_volumes: Iterable[dict[str, Any]],
    repo_root: Path,
) -> ProvisioningPlan:
    if authenticated_user.casefold() != OWNER.casefold():
        raise BlockedAuth(f"BLOCKED_AUTH: expected authenticated Hugging Face user {OWNER}")
    if space is not None and space.get("private") is not True:
        raise ProvisioningConflict("Existing STATICAI/BLACKHOLE Space must be private")
    if STORAGE_BUCKET in bucket_states and bucket_states[STORAGE_BUCKET] is not True:
        raise ProvisioningConflict(f"Existing bucket must be private: {STORAGE_BUCKET}")
    desired = (
        VolumeSpec(STORAGE_BUCKET, "/data", path="DATA"),
        VolumeSpec(STORAGE_BUCKET, "/lab", path="LAB"),
    )
    current = {
        (
            item.get("source"),
            item.get("path"),
            item.get("mount_path"),
            bool(item.get("read_only", False)),
        )
        for item in current_volumes
    }
    expected = {(item.source, item.path, item.mount_path, item.read_only) for item in desired}
    unexpected = current - expected
    if unexpected:
        raise ProvisioningConflict(f"Existing Space has unexpected volume mounts: {sorted(unexpected)}")
    return ProvisioningPlan(
        apply=False,
        authenticated_user=authenticated_user,
        create_buckets=() if STORAGE_BUCKET in bucket_states else (STORAGE_BUCKET,),
        create_space=space is None,
        replace_volumes=space is not None and current != expected,
        volumes=desired,
        upload_space_files=True,
        repo_root=str(repo_root.resolve()),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Apply the displayed private-resource plan")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser


def _configured_volume_records(info: object, runtime: object) -> list[dict[str, object]]:
    info_runtime = getattr(info, "runtime", None)
    configured = getattr(info_runtime, "volumes", None)
    if configured is None:
        configured = getattr(runtime, "volumes", None)
    return [
        {
            "source": getattr(volume, "source", None),
            "path": getattr(volume, "path", None),
            "mount_path": getattr(volume, "mount_path", None),
            "read_only": bool(getattr(volume, "read_only", False)),
        }
        for volume in (configured or [])
    ]


def _inspect(repo_root: Path) -> tuple[Any, str, dict[str, bool], dict[str, Any] | None, list[dict[str, Any]]]:
    from huggingface_hub import HfApi, get_token, list_buckets
    from huggingface_hub.errors import RepositoryNotFoundError

    if not get_token():
        raise BlockedAuth("BLOCKED_AUTH: run `hf auth login` locally; never paste a token into chat")
    api = HfApi(token=True)
    identity = api.whoami(token=True)
    user = str(identity.get("name") or identity.get("user") or "")
    buckets = {
        str(getattr(bucket, "id", "")): bool(getattr(bucket, "private", False))
        for bucket in list_buckets(namespace=OWNER, token=True)
    }
    try:
        info = api.space_info(SPACE_ID, token=True)
        space = {"private": bool(info.private), "sha": info.sha}
        runtime = api.get_space_runtime(SPACE_ID, token=True)
        volumes = _configured_volume_records(info, runtime)
    except RepositoryNotFoundError:
        space = None
        volumes = []
    return api, user, buckets, space, volumes


def _to_hub_volume(item: VolumeSpec) -> Any:
    from huggingface_hub import Volume

    return Volume(
        type="bucket",
        source=item.source,
        path=item.path,
        mount_path=item.mount_path,
        read_only=item.read_only,
    )


def _apply(api: Any, plan: ProvisioningPlan, repo_root: Path) -> None:
    from huggingface_hub import HfFileSystem, create_bucket

    volumes = [_to_hub_volume(item) for item in plan.volumes]
    for bucket in plan.create_buckets:
        create_bucket(bucket, private=True, exist_ok=True, token=True)
    fs = HfFileSystem(token=True)
    for item in plan.volumes:
        logical_source = f"{item.source}/{item.path}" if item.path else item.source
        marker = {"schema_version": "blackhole.volume.v1", "mount_path": item.mount_path, "bucket": logical_source}
        with fs.open(f"hf://buckets/{logical_source}/.blackhole-volume.json", "w") as handle:
            json.dump(marker, handle, sort_keys=True)
    if plan.create_space:
        api.create_repo(
            repo_id=SPACE_ID,
            repo_type="space",
            space_sdk="gradio",
            visibility="private",
            space_volumes=volumes,
            token=True,
        )
    elif plan.replace_volumes:
        api.set_space_volumes(SPACE_ID, volumes=volumes, token=True)
    api.upload_folder(
        repo_id=SPACE_ID,
        repo_type="space",
        folder_path=str(repo_root),
        ignore_patterns=[
            ".git/**",
            ".venv/**",
            ".superpowers/**",
            ".pytest_cache/**",
            ".blackhole-data/**",
            ".blackhole-lab/**",
        ],
        commit_message="Deploy verified BLACKHOLE lab",
        token=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        api, user, buckets, space, volumes = _inspect(args.repo_root)
        plan = make_plan(
            authenticated_user=user,
            bucket_states=buckets,
            space=space,
            current_volumes=volumes,
            repo_root=args.repo_root,
        )
    except (BlockedAuth, ProvisioningConflict) as exc:
        print(str(exc))
        return 2
    plan = replace(plan, apply=bool(args.apply))
    print(json.dumps(plan.to_dict(), indent=2))
    if not args.apply:
        print("DRY_RUN: no Hugging Face state changed; rerun with --apply after review")
        return 0
    _apply(api, plan, args.repo_root)
    print(f"APPLIED: private Space files and volumes submitted for {SPACE_ID}; run verify_deployment.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
