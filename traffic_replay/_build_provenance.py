"""Build and validate immutable package-source provenance.

The runtime normally derives source identity from the Git checkout that owns
the installed ``traffic_replay`` package.  A wheel has no ``.git`` directory,
so release builds carry a small JSON record instead.  The record is accepted
only when its clean Git identity, package version, build ID, and digest of the
shipped Python sources and instrument-owned JSON data are internally
consistent.

The build ID is a deterministic integrity checksum, not a signature.  Release
artifact hashes or attestations remain the external trust boundary.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from pathlib import Path, PurePosixPath
from typing import Mapping


PROVENANCE_FILENAME = "_build_provenance.json"
PROVENANCE_SCHEMA_VERSION = 2
_DISTRIBUTION_NAME = "llm-traffic-replay"
_BUILD_ID_DOMAIN = b"llm-traffic-replay-build-provenance-v2\0"
_MAX_PROVENANCE_INPUT_BYTES = 64 * 1024 * 1024
_EMPTY_STATUS_SHA256 = hashlib.sha256(b"").hexdigest()
_HEX_COMMIT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
_VERSION = re.compile(
    r'^__version__\s*=\s*"([^"]+)"\s*$', re.MULTILINE)
_SIGNED_FIELDS = (
    "provenance_schema_version",
    "distribution_name",
    "package_version",
    "git_commit",
    "git_dirty",
    "git_status_sha256",
    "source_tree_sha256",
    "source_file_count",
)
_RECORD_FIELDS = frozenset((*_SIGNED_FIELDS, "build_id"))


def provenance_input_path(relative_path: str) -> bool:
    """Return whether a package-relative file is bound by provenance.

    This intentionally mirrors ``pyproject.toml``: every Python module is
    shipped, and ``traffic_replay/data/*.json`` is the only package-data rule.
    The generated provenance JSON itself is excluded to avoid a recursive
    digest.
    """
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or not pure.parts \
            or any(part in {"", ".", ".."} for part in pure.parts) \
            or pure.as_posix() != relative_path:
        raise ValueError(
            f"package inventory path is not canonical: {relative_path!r}")
    return (pure.suffix == ".py"
            or (len(pure.parts) == 2 and pure.parts[0] == "data"
                and pure.suffix == ".json"))


def source_inventory_from_contents(
        contents: Mapping[str, bytes]) -> tuple[str, list[dict]]:
    """Return a canonical provenance inventory from package-relative bytes."""
    digest = hashlib.sha256()
    files = []
    for rel in sorted(contents):
        if not provenance_input_path(rel):
            continue
        raw = contents[rel]
        if not isinstance(raw, bytes):
            raise TypeError(f"package inventory bytes are invalid for {rel}")
        file_digest = hashlib.sha256(raw).hexdigest()
        digest.update(rel.encode("utf-8") + b"\0" + raw + b"\0")
        files.append({
            "path": rel,
            "sha256": file_digest,
            "bytes": len(raw),
        })
    if not files:
        raise ValueError("package provenance inventory is empty")
    return digest.hexdigest(), files


def _read_provenance_input(path: Path, relative_path: str) -> bytes:
    """Read one regular input through a no-follow descriptor."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(
                f"package provenance input is not a regular file: "
                f"{relative_path}")
        if info.st_size > _MAX_PROVENANCE_INPUT_BYTES:
            raise ValueError(
                f"package provenance input is unexpectedly large: "
                f"{relative_path}")
        chunks = []
        remaining = info.st_size
        while remaining:
            chunk = os.read(fd, min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError(
                    f"package provenance input was truncated while read: "
                    f"{relative_path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise ValueError(
                f"package provenance input grew while read: {relative_path}")
    finally:
        os.close(fd)
    return b"".join(chunks)


def source_inventory(package_dir: str | Path) -> tuple[str, list[dict]]:
    """Return the canonical digest of every shipped instrument-owned file."""
    supplied = Path(package_dir)
    if supplied.is_symlink():
        raise ValueError("package source directory must not be a symbolic link")
    root = supplied.resolve()
    if not root.is_dir():
        raise ValueError("package source directory does not exist")

    contents: dict[str, bytes] = {}

    def walk_error(error: OSError) -> None:
        raise error

    for current, directories, filenames in os.walk(
            root, topdown=True, followlinks=False, onerror=walk_error):
        current_path = Path(current)
        for name in (*directories, *filenames):
            candidate = current_path / name
            rel = candidate.relative_to(root).as_posix()
            # The generated record is deliberately outside its own digest and
            # has a separate no-follow reader that reports a rejected embedded
            # identity. Every other symlink fails the source inventory itself.
            if candidate.is_symlink() and rel != PROVENANCE_FILENAME:
                raise ValueError(
                    f"package source tree contains a symbolic link: {rel}")
        for name in filenames:
            candidate = current_path / name
            rel = candidate.relative_to(root).as_posix()
            if not provenance_input_path(rel):
                continue
            contents[rel] = _read_provenance_input(candidate, rel)
    return source_inventory_from_contents(contents)


def package_version(package_dir: str | Path) -> str:
    raw = (Path(package_dir) / "__init__.py").read_text(encoding="utf-8")
    matches = _VERSION.findall(raw)
    if len(matches) != 1 or not matches[0].strip():
        raise ValueError("package __version__ must occur exactly once")
    return matches[0]


def _build_id(fields: dict) -> str:
    canonical = json.dumps(
        {name: fields.get(name) for name in _SIGNED_FIELDS},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(_BUILD_ID_DOMAIN + canonical).hexdigest()


def make_provenance_record(
        *, version: str, git_commit: str | None,
        git_dirty: bool | None, git_status_sha256: str | None,
        source_tree_sha256: str, source_file_count: int) -> dict:
    """Create a deterministic record, including fail-closed dirty records."""
    fields = {
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "distribution_name": _DISTRIBUTION_NAME,
        "package_version": version,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "git_status_sha256": git_status_sha256,
        "source_tree_sha256": source_tree_sha256,
        "source_file_count": source_file_count,
    }
    return {**fields, "build_id": _build_id(fields)}


def validate_embedded_provenance(
        value: object, *, expected_version: str,
        source_tree_sha256: str, source_file_count: int) -> tuple[bool, str]:
    """Validate an installed provenance record against the shipped sources."""
    if not isinstance(value, dict) or set(value) != _RECORD_FIELDS:
        return False, "unknown or missing embedded provenance field"
    if value.get("provenance_schema_version") != PROVENANCE_SCHEMA_VERSION:
        return False, "unsupported embedded provenance schema"
    if value.get("distribution_name") != _DISTRIBUTION_NAME:
        return False, "embedded distribution name mismatch"
    if value.get("package_version") != expected_version:
        return False, "embedded package version mismatch"
    commit = value.get("git_commit")
    if not isinstance(commit, str) or not _HEX_COMMIT.fullmatch(commit) \
            or set(commit) == {"0"}:
        return False, "embedded Git commit is invalid"
    if value.get("git_dirty") is not False:
        return False, "embedded Git state is dirty or unknown"
    if value.get("git_status_sha256") != _EMPTY_STATUS_SHA256:
        return False, "embedded clean-tree assertion is inconsistent"
    tree = value.get("source_tree_sha256")
    if not isinstance(tree, str) or not _HEX_SHA256.fullmatch(tree) \
            or set(tree) == {"0"} or tree != source_tree_sha256:
        return False, "embedded source-tree digest mismatch"
    count = value.get("source_file_count")
    if isinstance(count, bool) or not isinstance(count, int) \
            or count < 1 or count != source_file_count:
        return False, "embedded source-file count mismatch"
    build_id = value.get("build_id")
    if not isinstance(build_id, str) or not _HEX_SHA256.fullmatch(build_id) \
            or set(build_id) == {"0"} or build_id != _build_id(value):
        return False, "embedded build ID mismatch"
    return True, "embedded build provenance is internally consistent"


def _json_without_duplicate_keys(raw: bytes) -> object:
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(raw.decode("utf-8"), object_pairs_hook=object_pairs)


def read_embedded_provenance(package_dir: str | Path) -> object:
    path = Path(package_dir) / PROVENANCE_FILENAME
    if path.is_symlink():
        raise ValueError("embedded provenance must not be a symbolic link")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("embedded provenance is not a regular file")
        if info.st_size > 64 * 1024:
            raise ValueError("embedded provenance is unexpectedly large")
        chunks = []
        remaining = info.st_size
        while remaining:
            chunk = os.read(fd, min(remaining, 64 * 1024))
            if not chunk:
                raise ValueError("embedded provenance was truncated while read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise ValueError("embedded provenance grew while read")
    finally:
        os.close(fd)
    return _json_without_duplicate_keys(b"".join(chunks))


def provenance_json(record: dict) -> str:
    return json.dumps(
        record, ensure_ascii=True, indent=2, sort_keys=True) + "\n"


def _run_git(cwd: Path, *args: str) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, timeout=10,
            check=False)
    except (OSError, subprocess.SubprocessError):
        return None


def _has_git_marker(start: Path) -> bool:
    current = start.resolve()
    for candidate in (current, *current.parents):
        try:
            if (candidate / ".git").exists():
                return True
        except OSError:
            return True
    return False


def _git_record(
        package_dir: Path, search_dir: Path, *, version: str,
        source_tree_sha256: str,
        source_file_count: int) -> tuple[dict | None, str]:
    """Return a live record only when Git owns this package's sources."""
    top_result = _run_git(search_dir, "rev-parse", "--show-toplevel")
    if top_result is None:
        return None, "git_unavailable" if _has_git_marker(search_dir) \
            else "not_a_git_source"
    if top_result.returncode != 0:
        return None, "git_unavailable" if _has_git_marker(search_dir) \
            else "not_a_git_source"
    try:
        top = Path(top_result.stdout.decode("utf-8").strip()).resolve()
        anchor = (package_dir / "__init__.py").resolve().relative_to(top)
    except (OSError, UnicodeError, ValueError):
        return None, "not_a_git_source"
    tracked = _run_git(
        top, "ls-files", "--error-unmatch", "--", anchor.as_posix())
    if tracked is None:
        return None, "git_unavailable"
    if tracked.returncode != 0:
        # This covers a wheel installed under an otherwise unrelated checkout.
        return None, "not_a_git_source"
    commit_result = _run_git(top, "rev-parse", "HEAD")
    status_result = _run_git(
        top, "status", "--porcelain=v1", "--untracked-files=all")
    if commit_result is None or status_result is None \
            or commit_result.returncode != 0 or status_result.returncode != 0:
        return None, "git_unavailable"
    try:
        commit = commit_result.stdout.decode("utf-8").strip()
        status = status_result.stdout.decode("utf-8").strip()
    except UnicodeError:
        return None, "git_unavailable"
    status_digest = hashlib.sha256(status.encode("utf-8")).hexdigest()
    return make_provenance_record(
        version=version,
        git_commit=commit,
        git_dirty=bool(status),
        git_status_sha256=status_digest,
        source_tree_sha256=source_tree_sha256,
        source_file_count=source_file_count,
    ), "git"


def build_provenance_for_source(
        package_dir: str | Path,
        search_dir: str | Path | None = None) -> tuple[dict, str, str | None]:
    """Resolve live Git provenance or a valid inherited sdist record.

    The returned record is always serializable so development builds can still
    be produced.  ``origin`` and ``error`` distinguish clean trusted records
    from dirty, missing, or inconsistent provenance; runtime callers must only
    trust ``git`` and ``embedded_build`` origins.
    """
    root = Path(package_dir).resolve()
    version = package_version(root)
    tree, files = source_inventory(root)
    record, git_state = _git_record(
        root, Path(search_dir or root).resolve(), version=version,
        source_tree_sha256=tree, source_file_count=len(files))
    if record is not None:
        return record, "git", None
    if git_state == "git_unavailable":
        unknown = make_provenance_record(
            version=version, git_commit=None, git_dirty=None,
            git_status_sha256=None, source_tree_sha256=tree,
            source_file_count=len(files))
        return unknown, "unavailable", "Git source identity is unavailable"

    try:
        embedded = read_embedded_provenance(root)
    except FileNotFoundError:
        error = "embedded build provenance is missing"
    except (OSError, UnicodeError, ValueError) as exc:
        error = f"embedded build provenance is unreadable: {exc}"
    else:
        valid, reason = validate_embedded_provenance(
            embedded, expected_version=version,
            source_tree_sha256=tree, source_file_count=len(files))
        if valid:
            return dict(embedded), "embedded_build", None
        error = reason
    unknown = make_provenance_record(
        version=version, git_commit=None, git_dirty=None,
        git_status_sha256=None, source_tree_sha256=tree,
        source_file_count=len(files))
    return unknown, "embedded_build_rejected", error
