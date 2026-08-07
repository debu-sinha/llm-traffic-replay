#!/usr/bin/env python3
"""Build and verify the self-contained Databricks smoke notebook payload.

The notebook carries the tracked package, its real pytest suite, and a small
allowlist of public sample inputs. Packing is deliberately strict: collection
comes from pytest, the unpacked copy must pass every collected case, and a
SHA-256 digest binds the displayed notebook to the exact canonical payload.

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

PUBLIC_CONFIGS = (
    "configs/profile_agent_stated.json",
    "configs/profile_agent_blended.json",
    "configs/profile_validation_small.json",
    "configs/prompts_example.jsonl",
    "configs/run_smoke.json",
    "configs/run_pt_full.json",
    "configs/run_prompts.json",
)
EXPLICIT_FILES = ("pyproject.toml", "scripts/profile_from_logs.py",
                  *PUBLIC_CONFIGS)
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


def _tracked_payload_paths() -> list[str]:
    tracked = _git_paths(
        "ls-files", "-z", "--", "traffic_replay", "tests", "configs",
        "pyproject.toml", "scripts/profile_from_logs.py")
    selected: set[str] = set()
    explicit = set(EXPLICIT_FILES)
    for rel in tracked:
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


def _assert_public_payload(rel: str, text: str) -> None:
    for pattern in _SECRET_PATTERNS:
        if pattern.search(text):
            raise SystemExit(
                f"possible credential in notebook input {rel}; refusing to pack")


def collect() -> dict[str, str]:
    """Return every tracked public file needed by the smoke notebook."""
    files: dict[str, str] = {}
    root_resolved = ROOT.resolve(strict=True)
    for rel in _tracked_payload_paths():
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
                "packed pytest suite is not completely green: "
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


def _claims(source: str) -> tuple[tuple[str, int, int], int]:
    headers = re.findall(
        r"Self-contained runnable payload \(v([^,]+), (\d+) tracked files, "
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
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
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
        raise SystemExit(
            "notebook payload is stale or noncanonical; run: "
            "python3 scripts/pack_notebook.py")

    tests = collected_count(files)
    actual = metadata(files, tests)
    claimed, cell_count = _claims(source)
    if claimed != actual or cell_count != tests:
        raise SystemExit(
            "notebook displayed metadata is stale: "
            f"claims {claimed} / cell cases {cell_count}, tree is {actual}; "
            "run: python3 scripts/pack_notebook.py")
    print(
        f"notebook payload and claims in sync "
        f"(v{actual[0]}, {actual[1]} files, {actual[2]} pytest cases, "
        f"sha256 {actual_digest})")


def pack() -> None:
    files = collect()
    test_count = verify(files)
    raw = payload_bytes(files)
    payload = base64.b64encode(raw).decode("ascii")
    digest = hashlib.sha256(raw).hexdigest()
    version, file_count, expected_tests = metadata(files, test_count)

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
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
                f"{file_count} tracked files, {expected_tests} pytest cases)",
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
