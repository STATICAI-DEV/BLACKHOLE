from pathlib import Path
from types import SimpleNamespace


def test_plan_rejects_wrong_authenticated_identity() -> None:
    import pytest

    from scripts.provision_hf import BlockedAuth, make_plan

    with pytest.raises(BlockedAuth, match="BluStatic"):
        make_plan(
            authenticated_user="someone-else",
            bucket_states={},
            space=None,
            current_volumes=[],
            repo_root=Path("."),
        )


def test_plan_uses_blackhole_bucket_subfolders_and_defaults_to_dry_run(tmp_path: Path) -> None:
    from scripts.provision_hf import STORAGE_BUCKET, make_plan

    plan = make_plan(
        authenticated_user="BluStatic",
        bucket_states={STORAGE_BUCKET: True},
        space=None,
        current_volumes=[],
        repo_root=tmp_path,
    )

    assert plan.apply is False
    assert plan.create_buckets == ()
    assert plan.create_space is True
    assert [(volume.source, volume.path, volume.mount_path) for volume in plan.volumes] == [
        ("BluStatic/BLACKHOLE", "DATA", "/data"),
        ("BluStatic/BLACKHOLE", "LAB", "/lab"),
    ]
    assert plan.upload_space_files is True


def test_hub_volume_serializes_bucket_subfolder_as_path() -> None:
    from scripts.provision_hf import STORAGE_BUCKET, VolumeSpec, _to_hub_volume

    volume = _to_hub_volume(VolumeSpec(STORAGE_BUCKET, "/data", path="DATA"))

    assert volume.to_dict() == {
        "type": "bucket",
        "source": "BluStatic/BLACKHOLE",
        "mountPath": "/data",
        "readOnly": False,
        "path": "DATA",
    }


def test_verifier_reads_configured_volumes_from_space_info() -> None:
    from scripts.verify_deployment import _configured_volumes

    volume = SimpleNamespace(
        source="BluStatic/BLACKHOLE",
        path="DATA",
        mount_path="/data",
        read_only=False,
    )
    info = SimpleNamespace(runtime=SimpleNamespace(volumes=[volume]))
    runtime = SimpleNamespace(volumes=None)

    assert _configured_volumes(info, runtime) == [
        {
            "source": "BluStatic/BLACKHOLE",
            "path": "DATA",
            "mount_path": "/data",
            "read_only": False,
        }
    ]


def test_provisioner_reads_configured_volumes_from_space_info() -> None:
    from scripts.provision_hf import _configured_volume_records

    volume = SimpleNamespace(
        source="BluStatic/BLACKHOLE",
        path="LAB",
        mount_path="/lab",
        read_only=False,
    )
    info = SimpleNamespace(runtime=SimpleNamespace(volumes=[volume]))
    runtime = SimpleNamespace(volumes=None)

    assert _configured_volume_records(info, runtime) == [
        {
            "source": "BluStatic/BLACKHOLE",
            "path": "LAB",
            "mount_path": "/lab",
            "read_only": False,
        }
    ]


def test_existing_public_space_is_rejected() -> None:
    import pytest

    from scripts.provision_hf import ProvisioningConflict, make_plan

    with pytest.raises(ProvisioningConflict, match="private"):
        make_plan(
            authenticated_user="BluStatic",
            bucket_states={"BluStatic/BLACKHOLE": True},
            space={"private": False},
            current_volumes=[],
            repo_root=Path("."),
        )


def test_cli_has_apply_flag_but_no_token_argument() -> None:
    from scripts.provision_hf import build_parser

    parser = build_parser()
    destinations = {action.dest for action in parser._actions}
    assert "apply" in destinations
    assert "token" not in destinations


def test_public_bucket_and_unrelated_existing_volume_are_rejected(tmp_path: Path) -> None:
    import pytest
    from scripts.provision_hf import ProvisioningConflict, make_plan

    with pytest.raises(ProvisioningConflict, match="bucket.*private"):
        make_plan(authenticated_user="BluStatic", bucket_states={"BluStatic/BLACKHOLE": False}, space=None, current_volumes=[], repo_root=tmp_path)
    with pytest.raises(ProvisioningConflict, match="unexpected volume"):
        make_plan(authenticated_user="BluStatic", bucket_states={}, space={"private": True}, current_volumes=[{"source": "someone/else", "mount_path": "/other", "read_only": False}], repo_root=tmp_path)
