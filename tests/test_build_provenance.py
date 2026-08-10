from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from traffic_replay import __version__
from traffic_replay._build_provenance import (
    PROVENANCE_FILENAME,
    make_provenance_record,
    provenance_json,
    source_inventory,
)
from traffic_replay.artifacts import snapshot_source_state
from traffic_replay.run_verification import _generator_reconstructibility


_SDIST_TEST_SUPPORT = {
    "configs/profile_agent_blended.json",
    "configs/profile_agent_stated.json",
    "configs/profile_glm52_canary_illustrative.json",
    "configs/profile_validation_small.json",
    "configs/prompts_example.jsonl",
    "configs/rate_limits_databricks_glm_5_2_enterprise_p2t_2026-08-07.json",
    "configs/run_prompts.json",
    "configs/run_pt_full.json",
    "configs/run_smoke.json",
    "docs/customer/benchmark-your-own-endpoint.html",
    "docs/diagrams/architecture.excalidraw",
    "docs/diagrams/architecture.svg",
    "docs/diagrams/load-model.svg",
    "docs/diagrams/request-sequence.svg",
    "notebooks/smoke_test_e2e_demo.ipynb",
    "scripts/build_customer_pdf.py",
    "scripts/pack_notebook.py",
    "scripts/profile_from_logs.py",
}


def _package(tmp_path: Path) -> Path:
    package = tmp_path / "installed" / "traffic_replay"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        f'__version__ = "{__version__}"\n', encoding="utf-8")
    (package / "worker.py").write_text(
        "def measured_value():\n    return 7\n", encoding="utf-8")
    (package / "data").mkdir()
    (package / "data" / "validation.json").write_text(
        '{"expected":7}\n', encoding="utf-8")
    return package


def _record(package: Path, *, commit: str | None = "a" * 40,
            dirty: bool | None = False,
            status: str | None = "") -> dict:
    tree, files = source_inventory(package)
    status_digest = (hashlib.sha256(status.encode("utf-8")).hexdigest()
                     if status is not None else None)
    return make_provenance_record(
        version=__version__, git_commit=commit, git_dirty=dirty,
        git_status_sha256=status_digest, source_tree_sha256=tree,
        source_file_count=len(files))


def _write(package: Path, record: dict) -> None:
    (package / PROVENANCE_FILENAME).write_text(
        provenance_json(record), encoding="utf-8")


def _assert_rejected(package: Path, reason: str) -> None:
    state = snapshot_source_state(package)
    assert state["source_identity_origin"] == "embedded_build_rejected"
    assert reason in state["embedded_provenance_error"]
    assert state["git_commit"] is None
    assert state["git_dirty"] is None
    assert state["build_id"] is None
    assert _generator_reconstructibility(state)["reconstructible"] is False


def test_installed_package_accepts_consistent_clean_embedded_provenance(
        tmp_path):
    package = _package(tmp_path)
    record = _record(package)
    _write(package, record)

    state = snapshot_source_state(package)

    assert state["source_identity_origin"] == "embedded_build"
    assert state["embedded_provenance_error"] is None
    assert state["git_commit"] == "a" * 40
    assert state["git_dirty"] is False
    assert state["git_status_sha256"] == hashlib.sha256(b"").hexdigest()
    assert state["source_tree_sha256"] == record["source_tree_sha256"]
    assert state["package_version"] == __version__
    assert state["build_id"] == record["build_id"]
    assert record["provenance_schema_version"] == 2
    assert _generator_reconstructibility(state)["reconstructible"] is True


def test_installed_package_rejects_source_tampering(tmp_path):
    package = _package(tmp_path)
    _write(package, _record(package))
    (package / "worker.py").write_text(
        "def measured_value():\n    return 8\n", encoding="utf-8")

    _assert_rejected(package, "source-tree digest mismatch")


def test_installed_package_rejects_instrument_data_tampering(tmp_path):
    package = _package(tmp_path)
    _write(package, _record(package))
    (package / "data" / "validation.json").write_text(
        '{"expected":8}\n', encoding="utf-8")

    _assert_rejected(package, "source-tree digest mismatch")


def test_installed_package_rejects_build_id_tampering(tmp_path):
    package = _package(tmp_path)
    record = _record(package)
    record["build_id"] = "b" * 64
    _write(package, record)

    _assert_rejected(package, "build ID mismatch")


@pytest.mark.parametrize(
    ("record_update", "reason"),
    [
        ({"git_commit": "0" * 40}, "Git commit is invalid"),
        ({"git_dirty": True}, "Git state is dirty or unknown"),
        ({"git_dirty": None}, "Git state is dirty or unknown"),
        ({"package_version": "999.0"}, "package version mismatch"),
    ],
)
def test_installed_package_rejects_invalid_identity_even_with_matching_build_id(
        tmp_path, record_update, reason):
    package = _package(tmp_path)
    base = _record(package)
    base.update(record_update)
    # Recreate the checksum after the mutation. This proves validation is not
    # merely a checksum comparison and still fails closed on identity policy.
    record = make_provenance_record(
        version=base["package_version"],
        git_commit=base["git_commit"],
        git_dirty=base["git_dirty"],
        git_status_sha256=base["git_status_sha256"],
        source_tree_sha256=base["source_tree_sha256"],
        source_file_count=base["source_file_count"],
    )
    _write(package, record)

    _assert_rejected(package, reason)


def test_installed_package_rejects_duplicate_or_unknown_fields(tmp_path):
    package = _package(tmp_path)
    record = _record(package)
    raw = json.dumps(record, sort_keys=True)[:-1] + ',"build_id":"c"}'
    (package / PROVENANCE_FILENAME).write_text(raw, encoding="utf-8")

    _assert_rejected(package, "unreadable")

    record["unexpected"] = True
    _write(package, record)
    _assert_rejected(package, "unknown or missing")


def test_installed_package_rejects_symlinked_provenance(tmp_path):
    package = _package(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text(provenance_json(_record(package)), encoding="utf-8")
    (package / PROVENANCE_FILENAME).symlink_to(outside)

    _assert_rejected(package, "symbolic link")


def test_installed_package_without_embedded_identity_fails_closed(tmp_path):
    package = _package(tmp_path)

    _assert_rejected(package, "is missing")


def test_package_inventory_rejects_symlinked_source_or_data(tmp_path):
    package = _package(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("VALUE = 9\n", encoding="utf-8")
    (package / "linked.py").symlink_to(outside)

    with pytest.raises(ValueError, match="symbolic link: linked.py"):
        source_inventory(package)

    (package / "linked.py").unlink()
    (package / "data" / "linked.json").symlink_to(outside)
    with pytest.raises(ValueError, match=r"symbolic link: data/linked\.json"):
        source_inventory(package)


def test_package_inventory_does_not_follow_a_file_swapped_before_open(
        tmp_path, monkeypatch):
    package = _package(tmp_path)
    worker = package / "worker.py"
    outside = tmp_path / "outside.py"
    outside.write_text("EXTERNAL = True\n", encoding="utf-8")
    real_open = os.open
    swapped = False

    def swap_then_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if Path(path) == worker and not swapped:
            swapped = True
            worker.unlink()
            worker.symlink_to(outside)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr("traffic_replay._build_provenance.os.open",
                        swap_then_open)
    with pytest.raises(OSError):
        source_inventory(package)
    assert swapped is True


def test_sdist_manifest_allowlists_every_test_support_file():
    root = Path(__file__).resolve().parents[1]
    manifest = (root / "MANIFEST.in").read_text(encoding="utf-8")
    included = {
        line.removeprefix("include ").strip()
        for line in manifest.splitlines()
        if line.startswith("include ")
    }

    assert _SDIST_TEST_SUPPORT <= included
    missing = {
        relative for relative in _SDIST_TEST_SUPPORT
        if not (root / relative).is_file()
    }
    # A self-contained notebook cannot recursively carry its own generated
    # payload. Its normalized semantic contract is packed under this name.
    if "notebooks/smoke_test_e2e_demo.ipynb" in missing \
            and (root / "notebooks/smoke_test_e2e_demo.contract.json").is_file():
        missing.remove("notebooks/smoke_test_e2e_demo.ipynb")
    assert not missing
    assert "recursive-include tests *.py" in manifest
    assert "recursive-include configs" not in manifest
    assert "include configs/*.json" not in manifest
