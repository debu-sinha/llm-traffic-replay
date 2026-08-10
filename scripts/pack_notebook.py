#!/usr/bin/env python3
"""Build and verify the self-contained Databricks diagnostic notebook payload.

The notebook carries the tracked package sources, its real pytest suite, and
an explicit allowlist of public examples plus test dependencies. Packing is
deliberately strict: collection comes from pytest, the unpacked copy must pass
every collected case, and a SHA-256 digest binds the displayed notebook to the
exact canonical payload.

Run after committing runtime or test changes, then commit the notebook:

    python3 scripts/pack_notebook.py

CI can perform the exact payload and metadata drift check without repeating
the full suite:

    python3 scripts/pack_notebook.py --check
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "smoke_test_e2e_demo.ipynb"
NOTEBOOK_CONTRACT = "notebooks/smoke_test_e2e_demo.contract.json"

# This script is also run by absolute path from outside the repository. Make
# its checked-in package parser available without depending on an editable
# install or the caller's current working directory.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from traffic_replay.json_input import (  # noqa: E402
    json_error_detail,
    loads_strict,
)
from traffic_replay._build_provenance import (  # noqa: E402
    PROVENANCE_FILENAME,
    build_provenance_for_source,
    make_provenance_record,
    provenance_input_path,
    provenance_json,
    source_inventory_from_contents,
    validate_embedded_provenance,
)

PUBLIC_CONFIGS = (
    "configs/profile_agent_stated.json",
    "configs/profile_agent_blended.json",
    "configs/profile_glm52_canary_illustrative.json",
    "configs/profile_validation_small.json",
    "configs/prompts_example.jsonl",
    "configs/rate_limits_databricks_glm_5_2_enterprise_p2t_2026-08-07.json",
    "configs/run_smoke.json",
    "configs/run_pt_full.json",
    "configs/run_prompts.json",
)
NOTEBOOK_SUPPORT_FILES = (
    "README.md",
    "CHANGELOG.md",
    "TODO.md",
    "MANIFEST.in",
    "setup.py",
    "scripts/build_customer_pdf.py",
    "scripts/pack_notebook.py",
    "scripts/profile_from_logs.py",
    "docs/ARCHITECTURE.md",
    "docs/OUTPUT_FIELD_REFERENCE.md",
    "docs/PRODUCTION_TESTING.md",
    "docs/RUN_YOUR_OWN_BENCHMARK.md",
    "docs/customer/benchmark-your-own-endpoint.html",
    "docs/diagrams/architecture.excalidraw",
    "docs/diagrams/architecture.svg",
    "docs/diagrams/load-model.svg",
    "docs/diagrams/request-sequence.svg",
)
EXPLICIT_FILES = ("pyproject.toml", *NOTEBOOK_SUPPORT_FILES, *PUBLIC_CONFIGS)
PACKAGE_SUFFIXES = {".py", ".json", ".jsonl", ".txt", ".typed"}
TEST_SUFFIXES = {".py", ".json", ".jsonl", ".txt", ".csv"}
COLLECT_TIMEOUT_S = 180
TEST_TIMEOUT_S = 1_200

# These patterns target credential formats, not benign test words such as
# "token" or "secret". Public samples and tests intentionally exercise
# redaction using unmistakably fake values.
_SECRET_PATTERNS = (
    re.compile(r"\bdapi[A-Za-z0-9]{32,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    re.compile(
        r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\."
        r"[A-Za-z0-9_-]{8,}\b"),
    re.compile(
        r"https://(?:dbc-[A-Za-z0-9-]+\.cloud\.databricks\.com|"
        r"adb-\d+\.\d+\.azuredatabricks\.net)\b", re.IGNORECASE),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~-]{40,}"),
)


def _git_paths(*args: str) -> list[str]:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise SystemExit(f"git could not enumerate notebook inputs: {detail}")
    return [item.decode("utf-8") for item in result.stdout.split(b"\0")
            if item]


def _git_checkout_owns_root() -> bool:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], cwd=ROOT,
            capture_output=True, check=False)
    except OSError:
        return False
    if result.returncode != 0:
        return False
    try:
        top = Path(result.stdout.decode("utf-8").strip()).resolve()
        return top == ROOT.resolve()
    except (OSError, UnicodeError):
        return False


def _select_payload_paths(candidates: list[str]) -> list[str]:
    selected: set[str] = set()
    explicit = set(EXPLICIT_FILES)
    for rel in candidates:
        pure = PurePosixPath(rel)
        if pure.is_absolute() or not pure.parts or ".." in pure.parts \
                or pure.as_posix() != rel:
            raise SystemExit(f"unsafe notebook input path: {rel!r}")
        suffix = PurePosixPath(rel).suffix
        if rel in explicit:
            selected.add(rel)
        elif rel.startswith("traffic_replay/"):
            if suffix not in PACKAGE_SUFFIXES:
                raise SystemExit(
                    f"tracked package file needs an explicit packing decision: {rel}")
            selected.add(rel)
        elif rel.startswith("tests/"):
            if suffix not in TEST_SUFFIXES:
                raise SystemExit(
                    f"tracked test file needs an explicit packing decision: {rel}")
            selected.add(rel)
        elif rel.startswith("configs/"):
            raise SystemExit(
                f"tracked config needs a public packing decision: {rel}")

    missing = sorted(explicit - selected)
    if missing:
        raise SystemExit(
            "required notebook input is missing or untracked: "
            + ", ".join(missing))
    return sorted(selected)


def _sdist_embedded_identity() -> dict:
    record, origin, error = build_provenance_for_source(
        ROOT / "traffic_replay", ROOT)
    if origin != "embedded_build":
        raise SystemExit(
            "Git-less notebook checking requires a source distribution with "
            "valid embedded build provenance"
            + (f": {error}" if error else ""))
    return record


def _sdist_payload_paths() -> list[str]:
    """Enumerate archive-declared inputs in a trusted Git-less sdist."""
    _sdist_embedded_identity()
    manifests = list(ROOT.glob("*.egg-info/SOURCES.txt"))
    if len(manifests) != 1:
        raise SystemExit(
            "Git-less notebook checking requires exactly one sdist "
            "SOURCES.txt manifest")
    try:
        candidates = manifests[0].read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise SystemExit(f"cannot read sdist SOURCES.txt: {exc}") from exc
    if not candidates or len(candidates) != len(set(candidates)):
        raise SystemExit("sdist SOURCES.txt is empty or contains duplicates")
    generated = f"traffic_replay/{PROVENANCE_FILENAME}"
    return _select_payload_paths(
        [relative for relative in candidates if relative != generated])


def _tracked_payload_paths() -> list[str]:
    if not _git_checkout_owns_root():
        return _sdist_payload_paths()
    tracked = _git_paths(
        "ls-files", "-z", "--", "traffic_replay", "tests",
        *EXPLICIT_FILES)
    return _select_payload_paths(tracked)


def _packed_notebook_source_commit() -> str:
    """Recover and validate the payload commit in a Git-less sdist."""
    notebook = _read_notebook()
    source = _notebook_source(notebook)
    payloads = re.findall(
        r'^PAYLOAD = "([^"]+)"$', source, re.MULTILINE)
    digests = re.findall(
        r'^PAYLOAD_SHA256 = "([0-9a-f]{64})"$', source, re.MULTILINE)
    if len(payloads) != 1 or len(digests) != 1:
        raise SystemExit(
            "sdist notebook has ambiguous payload or digest metadata")
    try:
        raw = base64.b64decode(payloads[0], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise SystemExit("sdist notebook payload is not valid base64") from exc
    if hashlib.sha256(raw).hexdigest() != digests[0]:
        raise SystemExit("sdist notebook payload does not match its digest")
    try:
        files = loads_strict(raw)
    except ValueError as exc:
        raise SystemExit(
            f"sdist notebook payload is invalid JSON ({json_error_detail(exc)})"
        ) from exc
    if not isinstance(files, dict) or not all(
            isinstance(rel, str) and isinstance(text, str)
            for rel, text in files.items()):
        raise SystemExit("sdist notebook payload must map paths to text")
    target = f"traffic_replay/{PROVENANCE_FILENAME}"
    try:
        embedded = loads_strict(files[target].encode("utf-8"))
        version_matches = re.findall(
            r'^__version__\s*=\s*"([^"]+)"\s*$',
            files["traffic_replay/__init__.py"], re.MULTILINE)
    except (KeyError, UnicodeError, ValueError) as exc:
        raise SystemExit(
            "sdist notebook payload provenance is missing or invalid") from exc
    if len(version_matches) != 1:
        raise SystemExit("sdist notebook payload package version is invalid")
    tree, count = _source_inventory_from_payload(files)
    valid, reason = validate_embedded_provenance(
        embedded, expected_version=version_matches[0],
        source_tree_sha256=tree, source_file_count=count)
    if not valid:
        raise SystemExit(
            f"sdist notebook payload provenance is invalid: {reason}")
    return embedded["git_commit"]


def _payload_source_commit(paths: list[str]) -> str:
    """Return the clean commit that last changed any payload source.

    The generated notebook is committed after its source inputs. Using the
    last payload-source commit keeps the embedded identity stable across that
    artifact-only commit while still proving that every packed byte is
    recoverable from one Git revision.
    """
    if not _git_checkout_owns_root():
        _sdist_embedded_identity()
        commit = _packed_notebook_source_commit()
        if not isinstance(commit, str) or not re.fullmatch(
                r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit):
            raise SystemExit("sdist notebook source commit is invalid")
        return commit

    history = subprocess.run(
        ["git", "rev-parse", "--is-shallow-repository"], cwd=ROOT,
        capture_output=True, check=False)
    if history.returncode != 0:
        detail = history.stderr.decode("utf-8", errors="replace").strip()
        raise SystemExit(
            f"git could not establish repository history depth: {detail}")
    try:
        shallow = history.stdout.decode("ascii").strip()
    except UnicodeError as exc:
        raise SystemExit("Git history depth was not ASCII") from exc
    if shallow == "true":
        raise SystemExit(
            "notebook packing requires complete Git history; fetch the full "
            "history (GitHub Actions: actions/checkout with fetch-depth: 0)")
    if shallow != "false":
        raise SystemExit("Git returned an invalid repository history depth")

    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT, capture_output=True, check=False)
    if status.returncode != 0:
        detail = status.stderr.decode("utf-8", errors="replace").strip()
        raise SystemExit(f"git could not establish a clean source tree: {detail}")
    if status.stdout:
        raise SystemExit(
            "notebook packing requires a clean Git tree; commit the source "
            "changes before generating the notebook")
    result = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", *paths], cwd=ROOT,
        capture_output=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise SystemExit(
            f"git could not resolve the payload source commit: {detail}")
    try:
        commit = result.stdout.decode("ascii").strip()
    except UnicodeError as exc:
        raise SystemExit("payload source commit was not ASCII") from exc
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit) \
            or set(commit) == {"0"}:
        raise SystemExit("payload source commit is missing or invalid")
    exact = subprocess.run(
        ["git", "diff", "--quiet", commit, "--", *paths], cwd=ROOT,
        capture_output=True, check=False)
    if exact.returncode != 0:
        raise SystemExit(
            "payload inputs are not all reconstructible from the resolved "
            f"source commit {commit}")
    return commit


def _source_inventory_from_payload(files: dict[str, str]) \
        -> tuple[str, int]:
    package_files = {
        rel.removeprefix("traffic_replay/"): text.encode("utf-8")
        for rel, text in files.items()
        if rel.startswith("traffic_replay/")
        and provenance_input_path(rel.removeprefix("traffic_replay/"))
    }
    try:
        digest, inventory = source_inventory_from_contents(package_files)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"notebook package provenance inventory is invalid: {exc}") from exc
    return digest, len(inventory)


def _add_embedded_provenance(
        files: dict[str, str], source_commit: str) -> None:
    target = f"traffic_replay/{PROVENANCE_FILENAME}"
    if target in files:
        raise SystemExit(
            f"{target} must be generated by the notebook packer, not tracked")
    tree_digest, source_count = _source_inventory_from_payload(files)
    version_match = re.findall(
        r'^__version__\s*=\s*"([^"]+)"\s*$',
        files.get("traffic_replay/__init__.py", ""), re.MULTILINE)
    if len(version_match) != 1:
        raise SystemExit("could not derive one package version for provenance")
    record = make_provenance_record(
        version=version_match[0], git_commit=source_commit, git_dirty=False,
        git_status_sha256=hashlib.sha256(b"").hexdigest(),
        source_tree_sha256=tree_digest, source_file_count=source_count)
    valid, reason = validate_embedded_provenance(
        record, expected_version=version_match[0],
        source_tree_sha256=tree_digest, source_file_count=source_count)
    if not valid:
        raise SystemExit(f"generated notebook provenance is invalid: {reason}")
    files[target] = provenance_json(record)


def _assert_public_payload(rel: str, text: str) -> None:
    for pattern in _SECRET_PATTERNS:
        if pattern.search(text):
            raise SystemExit(
                f"possible credential in notebook input {rel}; refusing to pack")


def collect() -> dict[str, str]:
    """Return every public file needed by the diagnostic notebook."""
    files: dict[str, str] = {}
    root_resolved = ROOT.resolve(strict=True)
    paths = _tracked_payload_paths()
    source_commit = _payload_source_commit(paths)
    for rel in paths:
        path = ROOT / rel
        if path.is_symlink() or not path.is_file():
            raise SystemExit(f"notebook input must be a regular file: {rel}")
        try:
            path.resolve(strict=True).relative_to(root_resolved)
        except (OSError, ValueError) as exc:
            raise SystemExit(
                f"notebook input escapes the repository root: {rel}") from exc
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise SystemExit(f"cannot read notebook input {rel}: {exc}") from exc
        _assert_public_payload(rel, text)
        files[rel] = text
    if not files:
        raise SystemExit("notebook payload would be empty")
    contract = _normalized_notebook_contract()
    committed_contract = _notebook_contract_at_commit(source_commit)
    if contract != committed_contract:
        raise SystemExit(
            "the diagnostic notebook contract is not reconstructible from "
            f"payload source commit {source_commit}; commit notebook semantic "
            "changes together with a payload source change before packing")
    _assert_public_payload(NOTEBOOK_CONTRACT, contract)
    files[NOTEBOOK_CONTRACT] = contract
    _add_embedded_provenance(files, source_commit)
    return files


def payload_bytes(files: dict[str, str]) -> bytes:
    """Canonical bytes used for both base64 encoding and the digest."""
    return json.dumps(
        files, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")


def _write_payload(files: dict[str, str], root: Path) -> None:
    if any(root.iterdir()):
        raise SystemExit(f"temporary extraction directory is not empty: {root}")
    for rel, text in files.items():
        pure = PurePosixPath(rel)
        if pure.is_absolute() or not pure.parts or ".." in pure.parts:
            raise SystemExit(f"unsafe notebook payload path: {rel!r}")
        path = root.joinpath(*pure.parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))


def _run(command: list[str], root: Path, timeout_s: int) \
        -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment.pop("PYTEST_ADDOPTS", None)
    environment.pop("PYTEST_PLUGINS", None)
    environment.pop("PYTHONPATH", None)
    try:
        return subprocess.run(
            command, cwd=root, capture_output=True, text=True, check=False,
            timeout=timeout_s, env=environment)
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(
            f"command exceeded {timeout_s}s: {' '.join(command)}") from exc


def _tail(result: subprocess.CompletedProcess[str]) -> str:
    chunks = []
    if result.stdout:
        chunks.append(result.stdout[-4_000:])
    if result.stderr:
        chunks.append(result.stderr[-2_000:])
    return "\n".join(chunks)


def _collect_pytest_cases(root: Path) -> tuple[str, ...]:
    command = [
        sys.executable, "-m", "pytest", "--collect-only", "-q",
        "-o", "addopts=", "-p", "no:cacheprovider",
    ]
    result = _run(command, root, COLLECT_TIMEOUT_S)
    if result.returncode != 0:
        print(_tail(result), file=sys.stderr)
        raise SystemExit(
            f"pytest collection failed with exit code {result.returncode}")
    nodeids = tuple(
        line.strip() for line in result.stdout.splitlines()
        if line.startswith("tests/") and "::" in line)
    if not nodeids:
        raise SystemExit("pytest collected no cases; refusing to publish")
    if len(nodeids) != len(set(nodeids)):
        raise SystemExit("pytest emitted duplicate collected case IDs")
    summary = re.search(r"(?m)^(\d+) tests? collected\b", result.stdout)
    if not summary or int(summary.group(1)) != len(nodeids):
        raise SystemExit(
            "pytest collection summary and collected case IDs disagree: "
            f"summary={summary.group(1) if summary else 'missing'}, "
            f"nodeids={len(nodeids)}")
    return nodeids


def _junit_counts(path: Path) -> dict[str, int]:
    try:
        document = ET.parse(path)
    except (OSError, ET.ParseError) as exc:
        raise SystemExit(f"pytest did not write valid JUnit evidence: {exc}") from exc
    suites = document.getroot().findall("testsuite")
    if len(suites) != 1:
        raise SystemExit(
            f"pytest JUnit evidence has {len(suites)} suites, expected 1")
    try:
        return {name: int(suites[0].attrib.get(name, "0"))
                for name in ("tests", "failures", "errors", "skipped")}
    except ValueError as exc:
        raise SystemExit("pytest JUnit evidence has invalid counters") from exc


def verify(files: dict[str, str]) -> int:
    """Run real pytest against a fresh unpacking and return collected cases."""
    with tempfile.TemporaryDirectory(prefix="notebook-pack-verify-") as tmp:
        root = Path(tmp)
        _write_payload(files, root)
        nodeids = _collect_pytest_cases(root)
        junit = root / ".pytest-results.xml"
        result = _run([
            sys.executable, "-m", "pytest", "-q", "-o", "addopts=",
            "-p", "no:cacheprovider", f"--junitxml={junit}",
        ], root, TEST_TIMEOUT_S)
        if result.returncode != 0:
            print(_tail(result), file=sys.stderr)
            raise SystemExit(
                "packed pytest suite failed with exit code "
                f"{result.returncode}")
        counts = _junit_counts(junit)
        if (counts["tests"] != len(nodeids)
                or any(counts[name] != 0
                       for name in ("failures", "errors", "skipped"))):
            print(_tail(result), file=sys.stderr)
            raise SystemExit(
                "packed pytest suite did not pass completely: "
                f"exit={result.returncode}, collected={len(nodeids)}, "
                f"junit={counts}")
        return len(nodeids)


def collected_count(files: dict[str, str]) -> int:
    """Collect cases from a fresh unpacking without executing them."""
    with tempfile.TemporaryDirectory(prefix="notebook-pack-collect-") as tmp:
        root = Path(tmp)
        _write_payload(files, root)
        return len(_collect_pytest_cases(root))


def metadata(files: dict[str, str], test_count: int) -> tuple[str, int, int]:
    init_match = re.search(
        r'^__version__\s*=\s*"([^"]+)"\s*$',
        files["traffic_replay/__init__.py"], re.MULTILINE)
    project_block = re.search(
        r"(?ms)^\[project\]\s*$\n(.*?)(?=^\[|\Z)",
        files["pyproject.toml"])
    project_match = (re.search(
        r'^version\s*=\s*"([^"]+)"\s*$', project_block.group(1),
        re.MULTILINE) if project_block else None)
    if not init_match or not project_match:
        raise SystemExit("could not derive both package version declarations")
    if init_match.group(1) != project_match.group(1):
        raise SystemExit(
            "package version declarations disagree: "
            f"__init__={init_match.group(1)}, project={project_match.group(1)}")
    if test_count < 1:
        raise SystemExit("notebook metadata cannot claim zero pytest cases")
    return init_match.group(1), len(files), test_count


def _applied(count: int, what: str) -> None:
    if count != 1:
        raise SystemExit(
            f"notebook {what} rewrite matched {count} times, expected 1; "
            "update the packing anchors before publishing")


def _notebook_source(notebook: dict) -> str:
    return "".join(
        "".join(cell.get("source", []))
        if isinstance(cell.get("source", []), list)
        else str(cell.get("source", ""))
        for cell in notebook.get("cells", []))


def _assert_clean_notebook(notebook: dict) -> None:
    cells = notebook.get("cells")
    if not isinstance(cells, list) or not cells:
        raise SystemExit("notebook has no cells")
    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            raise SystemExit(
                f"notebook cell {index} contains outputs; clear them before packing")
        if cell.get("execution_count") is not None:
            raise SystemExit(
                f"notebook cell {index} has an execution count; clear it first")


def _read_notebook() -> dict:
    """Read the checked-in notebook as strict UTF-8 JSON."""
    try:
        notebook = loads_strict(NOTEBOOK.read_bytes())
    except ValueError as exc:
        raise SystemExit(
            f"notebook is invalid JSON ({json_error_detail(exc)})") from exc
    if not isinstance(notebook, dict):
        raise SystemExit("notebook JSON root must be an object")
    return notebook


def _normalize_notebook_contract(notebook: dict) -> str:
    """Remove only generated payload metadata from one notebook object."""
    _assert_clean_notebook(notebook)
    for cell in notebook["cells"]:
        source = ("".join(cell.get("source", []))
                  if isinstance(cell.get("source"), list)
                  else str(cell.get("source", "")))
        source = re.sub(
            r'^PAYLOAD = "[^"]*"$', 'PAYLOAD = ""', source,
            flags=re.MULTILINE)
        source = re.sub(
            r'^PAYLOAD_SHA256 = "[0-9a-f]*"$',
            'PAYLOAD_SHA256 = "' + "0" * 64 + '"', source,
            flags=re.MULTILINE)
        source = re.sub(
            r'^PACKED_VERSION = "[^"]*"$', 'PACKED_VERSION = "CONTRACT"',
            source, flags=re.MULTILINE)
        source = re.sub(
            r"^EXPECTED_PAYLOAD_FILES = \d+$",
            "EXPECTED_PAYLOAD_FILES = 0", source, flags=re.MULTILINE)
        source = re.sub(
            r"^EXPECTED_PYTEST_CASES = \d+$",
            "EXPECTED_PYTEST_CASES = 0", source, flags=re.MULTILINE)
        source = re.sub(
            r"run the full pytest suite \(\d+ cases\)",
            "run the full pytest suite (0 cases)", source)
        source = re.sub(
            r"Self-contained runnable payload \([^)]*\)",
            "Self-contained runnable payload "
            "(vCONTRACT, 0 payload files, 0 pytest cases)", source)
        cell["source"] = source.splitlines(keepends=True)
    return json.dumps(
        notebook, indent=1, ensure_ascii=False, allow_nan=False) + "\n"


def _normalized_notebook_contract() -> str:
    """Return executable-cell intent without recursively packing payload.

    The complete notebook cannot contain itself. This normalized copy removes
    only generated payload metadata, so the pytest suite unpacked from the
    notebook can still assert the paid-canary safety contract it will run.
    """
    return _normalize_notebook_contract(_read_notebook())


def _notebook_contract_at_commit(commit: str) -> str:
    if not _git_checkout_owns_root():
        _sdist_embedded_identity()
        if _packed_notebook_source_commit() != commit:
            raise SystemExit("sdist notebook source commit changed while read")
        return _normalized_notebook_contract()
    relative = NOTEBOOK.relative_to(ROOT).as_posix()
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT,
        capture_output=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise SystemExit(
            f"git could not reconstruct the notebook contract: {detail}")
    try:
        notebook = loads_strict(result.stdout)
    except ValueError as exc:
        raise SystemExit(
            "committed diagnostic notebook is invalid JSON "
            f"({json_error_detail(exc)})") from exc
    if not isinstance(notebook, dict):
        raise SystemExit("committed diagnostic notebook root is not an object")
    return _normalize_notebook_contract(notebook)


def _claims(source: str) -> tuple[tuple[str, int, int], int]:
    headers = re.findall(
        r"Self-contained runnable payload \(v([^,]+), (\d+) payload files, "
        r"(\d+) pytest cases\)", source)
    cells = re.findall(
        r"run the full pytest suite \((\d+) cases\)", source)
    constants = re.findall(
        r"^EXPECTED_PYTEST_CASES = (\d+)$", source, re.MULTILINE)
    runtime_versions = re.findall(
        r'^PACKED_VERSION = "([^"]+)"$', source, re.MULTILINE)
    runtime_files = re.findall(
        r"^EXPECTED_PAYLOAD_FILES = (\d+)$", source, re.MULTILINE)
    groups = (headers, cells, constants, runtime_versions, runtime_files)
    if any(len(group) != 1 for group in groups):
        raise SystemExit(
            "notebook version, file, and pytest claims must occur exactly once")
    header = headers[0]
    if (runtime_versions[0] != header[0]
            or int(runtime_files[0]) != int(header[1])):
        raise SystemExit("notebook header and runtime metadata disagree")
    cell_count = int(cells[0])
    if cell_count != int(constants[0]):
        raise SystemExit("notebook pytest comment and runtime constant disagree")
    return ((header[0], int(header[1]), int(header[2])),
            cell_count)


def check() -> None:
    files = collect()
    expected_raw = payload_bytes(files)
    notebook = _read_notebook()
    _assert_clean_notebook(notebook)
    source = _notebook_source(notebook)

    payload_matches = re.findall(
        r'^PAYLOAD = "([^"]+)"$', source, re.MULTILINE)
    digest_matches = re.findall(
        r'^PAYLOAD_SHA256 = "([0-9a-f]{64})"$', source, re.MULTILINE)
    if len(payload_matches) != 1 or len(digest_matches) != 1:
        raise SystemExit(
            "notebook payload and SHA-256 claim must occur exactly once")
    try:
        actual_raw = base64.b64decode(payload_matches[0], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise SystemExit("notebook payload is not valid base64") from exc
    actual_digest = hashlib.sha256(actual_raw).hexdigest()
    if actual_digest != digest_matches[0]:
        raise SystemExit("notebook payload does not match its SHA-256 claim")
    if actual_raw != expected_raw:
        if not _git_checkout_owns_root():
            raise SystemExit(
                "source distribution contains a stale or noncanonical "
                "notebook payload; rebuild it from an owning Git checkout "
                "after packing the notebook")
        raise SystemExit(
            "notebook payload is stale or noncanonical; run: "
            "python3 scripts/pack_notebook.py")

    tests = collected_count(files)
    actual = metadata(files, tests)
    claimed, cell_count = _claims(source)
    if claimed != actual or cell_count != tests:
        if not _git_checkout_owns_root():
            raise SystemExit(
                "source distribution contains stale notebook metadata; "
                "rebuild it from an owning Git checkout after packing the "
                "notebook")
        raise SystemExit(
            "notebook displayed metadata is stale: "
            f"claims {claimed} / cell cases {cell_count}, tree is {actual}; "
            "run: python3 scripts/pack_notebook.py")
    print(
        f"notebook payload and claims in sync "
        f"(v{actual[0]}, {actual[1]} files, {actual[2]} pytest cases, "
        f"sha256 {actual_digest})")


def pack() -> None:
    if not _git_checkout_owns_root():
        raise SystemExit(
            "notebook generation requires the owning Git checkout; a source "
            "distribution supports --check only")
    files = collect()
    test_count = verify(files)
    raw = payload_bytes(files)
    payload = base64.b64encode(raw).decode("ascii")
    digest = hashlib.sha256(raw).hexdigest()
    version, file_count, expected_tests = metadata(files, test_count)

    notebook = _read_notebook()
    _assert_clean_notebook(notebook)
    hits = {
        "payload": 0,
        "payload digest": 0,
        "runtime version": 0,
        "runtime file count": 0,
        "pytest cell count": 0,
        "pytest runtime count": 0,
        "header": 0,
    }
    for cell in notebook["cells"]:
        source = ("".join(cell["source"])
                  if isinstance(cell.get("source"), list)
                  else str(cell.get("source", "")))
        if "PAYLOAD = " in source:
            source, count = re.subn(
                r'^PAYLOAD = "[^"]*"$', f'PAYLOAD = "{payload}"', source,
                flags=re.MULTILINE)
            hits["payload"] += count
            source, count = re.subn(
                r'^PAYLOAD_SHA256 = "[0-9a-f]*"$',
                f'PAYLOAD_SHA256 = "{digest}"', source,
                flags=re.MULTILINE)
            hits["payload digest"] += count
            source, count = re.subn(
                r'^PACKED_VERSION = "[^"]*"$',
                f'PACKED_VERSION = "{version}"', source,
                flags=re.MULTILINE)
            hits["runtime version"] += count
            source, count = re.subn(
                r"^EXPECTED_PAYLOAD_FILES = \d+$",
                f"EXPECTED_PAYLOAD_FILES = {file_count}", source,
                flags=re.MULTILINE)
            hits["runtime file count"] += count
        if "run the full pytest suite" in source:
            source, count = re.subn(
                r"run the full pytest suite \(\d+ cases\)",
                f"run the full pytest suite ({expected_tests} cases)", source)
            hits["pytest cell count"] += count
            source, count = re.subn(
                r"^EXPECTED_PYTEST_CASES = \d+$",
                f"EXPECTED_PYTEST_CASES = {expected_tests}", source,
                flags=re.MULTILINE)
            hits["pytest runtime count"] += count
        if cell.get("cell_type") == "markdown" \
                and "Self-contained runnable payload" in source:
            source, count = re.subn(
                r"Self-contained runnable payload \([^)]*\)",
                f"Self-contained runnable payload (v{version}, "
                f"{file_count} payload files, {expected_tests} pytest cases)",
                source)
            hits["header"] += count
        cell["source"] = source.splitlines(keepends=True)
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    for what, count in hits.items():
        _applied(count, what)
    NOTEBOOK.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(
        f"packed v{version}: {file_count} files, {expected_tests} pytest "
        f"cases, sha256 {digest}, payload {len(payload)} chars")


def main() -> None:
    args = sys.argv[1:]
    if args == ["--check"]:
        check()
    elif not args:
        pack()
    else:
        raise SystemExit("usage: pack_notebook.py [--check]")


if __name__ == "__main__":
    main()
