#!/usr/bin/env python3
"""Rebuild the self-contained payload inside notebooks/smoke_test_e2e_demo.ipynb.

The notebook carries a base64 copy of the package so it can be dropped into a
Databricks workspace and run with no git, no pip install, and no cluster
library. That copy goes stale the moment the package changes, and a stale copy
is worse than no copy: the notebook is the path the README sends people to, so
it would quietly measure old code. Version 0.3.0 changed what TTFT includes,
which is exactly the kind of drift that must not ship silently.

Run this after any change to the package, and commit the notebook:

    python3 scripts/pack_notebook.py

It rewrites the payload, the file count and the test count from the tree, then
verifies the packed copy imports and its tests pass in a scratch directory.
"""

from __future__ import annotations

import base64
import ast
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "smoke_test_e2e_demo.ipynb"


def collect() -> dict[str, str]:
    """Everything the notebook needs to run the suite and a smoke replay."""
    files: dict[str, str] = {}
    for pat in (
        "traffic_replay/*.py",
        "tests/test_*.py",
        # allowlist, NOT a glob. a user following the README writes their
        # log-derived profile into configs/, and globbing would pack a real
        # customer's traffic quantiles into a shareable notebook.
        "configs/profile_agent_stated.json",
        "configs/profile_agent_blended.json",
        "configs/profile_validation_small.json",
        "configs/prompts_example.jsonl",
        "configs/run_smoke.json",
        "configs/run_pt_full.json",
        "configs/run_prompts.json",
        "scripts/run_tests_stdlib.py",
    ):
        for p in sorted(ROOT.glob(pat)):
            files[str(p.relative_to(ROOT))] = p.read_text()
    return files


def verify(files: dict[str, str]) -> int:
    """Unpack to a scratch dir and run the packed suite. Returns test count."""
    with tempfile.TemporaryDirectory(prefix="packcheck-") as tmp:
        root = Path(tmp)
        for rel, text in files.items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        r = subprocess.run(
            [sys.executable, "scripts/run_tests_stdlib.py"],
            cwd=root,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0 or " 0 failed" not in r.stdout:
            print(r.stdout[-3000:])
            print(r.stderr[-2000:], file=sys.stderr)
            raise SystemExit("packed copy does not pass its own tests")
        m = re.search(r"(\d+) passed", r.stdout)
        n = int(m.group(1)) if m else 0
        # "0 passed, 0 failed" also satisfies the check above, so a payload
        # that collected nothing would otherwise ship as green.
        if n < 1:
            raise SystemExit("packed copy collected no tests, refusing to write")
        return n


def metadata(files: dict[str, str]) -> tuple[str, int, int]:
    """Return version, payload file count, and statically collected tests.

    The bundled stdlib runner executes top-level callables named ``test_*``.
    Counting the same definitions here lets ``--check`` validate notebook
    claims without re-running the expensive suite. The full pack path still
    executes the suite and uses its observed count as the authority.
    """
    match = re.search(
        r'__version__ = "([^"]+)"', files["traffic_replay/__init__.py"])
    if not match:
        raise SystemExit("could not read package version for notebook")
    tests = 0
    for path, source in files.items():
        if not path.startswith("tests/test_") or not path.endswith(".py"):
            continue
        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError as exc:
            raise SystemExit(f"cannot parse packed test {path}: {exc}") from exc
        tests += sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
            for node in tree.body
        )
    if tests < 1:
        raise SystemExit("packed tree contains no statically collectible tests")
    return match.group(1), len(files), tests


def _applied(count: int, what: str) -> None:
    """A rewrite that silently matched nothing would freeze a stale count in
    the notebook while the payload moved on, which is the drift this script
    exists to stop."""
    if count != 1:
        raise SystemExit(
            f"notebook {what} rewrite matched {count} times, expected 1. "
            "the anchor text changed, fix this script before packing.")


def check() -> None:
    """Compare the packed payload to the tree without re-running the suite.

    CI uses this: the expensive part of packing is running the packed tests,
    and CI already runs them from the tree. All CI needs to know is whether
    the payload drifted.
    """
    files = collect()
    nb = json.loads(NOTEBOOK.read_text())
    src = "".join("".join(c["source"]) for c in nb["cells"])
    m = re.search(r'PAYLOAD = "([^"]+)"', src)
    if not m:
        raise SystemExit("no PAYLOAD found in the notebook")
    packed = json.loads(base64.b64decode(m.group(1)))
    missing = sorted(set(files) - set(packed))
    extra = sorted(set(packed) - set(files))
    stale = sorted(k for k in set(files) & set(packed) if files[k] != packed[k])
    if missing or extra or stale:
        for label, items in (("missing from payload", missing),
                             ("no longer in the tree", extra),
                             ("out of date", stale)):
            if items:
                print(f"{label}: {', '.join(items)}")
        raise SystemExit(
            "notebook payload is stale. run: python3 scripts/pack_notebook.py")

    version, file_count, test_count = metadata(files)
    header = re.search(
        r"Self-contained copy of the repo \(v([^,]+), (\d+) files, "
        r"(\d+) tests\)", src)
    cell_count = re.search(
        r"run the full test suite \((\d+) tests\)", src)
    if not header or not cell_count:
        raise SystemExit("notebook version/file/test claims are missing")
    claimed = (header.group(1), int(header.group(2)), int(header.group(3)))
    actual = (version, file_count, test_count)
    if claimed != actual or int(cell_count.group(1)) != test_count:
        raise SystemExit(
            "notebook displayed metadata is stale: "
            f"claims {claimed} / cell tests {cell_count.group(1)}, "
            f"tree is {actual}. run: python3 scripts/pack_notebook.py")
    print(
        f"notebook payload and claims in sync "
        f"(v{version}, {file_count} files, {test_count} tests)")


def main() -> None:
    if "--check" in sys.argv:
        check()
        return
    files = collect()
    n_tests = verify(files)
    payload = base64.b64encode(json.dumps(files).encode()).decode()

    hits = {"payload": 0, "test count": 0, "header": 0}
    nb = json.loads(NOTEBOOK.read_text())
    version, file_count, static_test_count = metadata(files)
    if static_test_count != n_tests:
        raise SystemExit(
            "packed runner and static collection disagree: "
            f"runner={n_tests}, static={static_test_count}. Update the "
            "zero-dependency runner before publishing its test count.")

    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if "PAYLOAD = " in src:
            src, k = re.subn(r'PAYLOAD = "[^"]*"', f'PAYLOAD = "{payload}"', src)
            hits["payload"] += k
            cell["source"] = src.splitlines(keepends=True)
        elif "run the full test suite" in src:
            src, k = re.subn(
                r"run the full test suite \(\d+ tests\)",
                f"run the full test suite ({n_tests} tests)",
                src,
            )
            hits["test count"] += k
            cell["source"] = src.splitlines(keepends=True)
        elif cell["cell_type"] == "markdown" and "Self-contained copy" in src:
            src, k = re.subn(
                r"Self-contained copy of the repo \([^)]*\)",
                f"Self-contained copy of the repo (v{version}, "
                f"{file_count} files, {n_tests} tests)",
                src,
            )
            hits["header"] += k
            cell["source"] = src.splitlines(keepends=True)

    # a branch that never ran is the silent failure: the payload moves
    # while the counts freeze. assert every rewrite landed exactly once.
    for what, k in hits.items():
        _applied(k, what)

    NOTEBOOK.write_text(json.dumps(nb, indent=1) + "\n")
    print(
        f"packed v{version}: {file_count} files, {n_tests} tests, "
        f"payload {len(payload)} chars"
    )


if __name__ == "__main__":
    main()
