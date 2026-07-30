#!/usr/bin/env python3
"""Zero-dependency test runner.

Runs the real files under tests/ through a minimal pytest-compatible shim
(fixture, raises, tmp_path_factory), so environments without pytest can
still verify the suite. With pytest installed, prefer: python -m pytest
"""
from __future__ import annotations

import importlib.util
import inspect
import sys
import tempfile
import traceback
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------- pytest shim ----------------
class _Raises:
    def __init__(self, exc_type):
        self.exc_type = exc_type

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            raise AssertionError(f"expected {self.exc_type.__name__}, "
                                 f"nothing raised")
        return issubclass(et, self.exc_type)


class _TmpPathFactory:
    def mktemp(self, name: str) -> Path:
        return Path(tempfile.mkdtemp(prefix=f"{name}-"))


def _make_shim() -> types.ModuleType:
    shim = types.ModuleType("pytest")
    shim._fixtures = {}

    def fixture(fn=None, *, scope="function"):
        def deco(f):
            f.__is_fixture__ = True
            return f
        return deco(fn) if fn else deco

    shim.fixture = fixture
    shim.raises = _Raises

    class _Mark:
        def __getattr__(self, name):
            def deco(f=None, *a, **k):
                return f if f is not None else (lambda g: g)
            return deco

    shim.mark = _Mark()
    return shim


def _load_module(path: Path, shim: types.ModuleType):
    sys.modules["pytest"] = shim
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_module(path: Path) -> tuple[int, int, list[str]]:
    shim = _make_shim()
    mod = _load_module(path, shim)

    fixtures = {n: f for n, f in vars(mod).items()
                if callable(f) and getattr(f, "__is_fixture__", False)}
    cache: dict[str, object] = {}
    teardowns: list = []

    def resolve(name: str):
        if name == "tmp_path_factory":
            return _TmpPathFactory()
        if name in cache:
            return cache[name]
        if name not in fixtures:
            raise KeyError(f"unknown fixture {name!r} in {path.name}")
        f = fixtures[name]
        kwargs = {p: resolve(p) for p in inspect.signature(f).parameters}
        val = f(**kwargs)
        if inspect.isgenerator(val):
            gen = val
            val = next(gen)
            teardowns.append(gen)
        cache[name] = val
        return val

    passed = failed = 0
    failures: list[str] = []
    for name, fn in vars(mod).items():
        if not (name.startswith("test_") and callable(fn)):
            continue
        try:
            kwargs = {p: resolve(p) for p in inspect.signature(fn).parameters}
            fn(**kwargs)
            passed += 1
            print(f"  PASS {path.name}::{name}")
        except Exception:
            failed += 1
            failures.append(f"{path.name}::{name}\n"
                            + traceback.format_exc(limit=4))
            print(f"  FAIL {path.name}::{name}")
    for gen in teardowns:
        try:
            next(gen, None)
        except Exception:
            pass
    return passed, failed, failures


def main() -> int:
    test_dir = ROOT / "tests"
    total_p = total_f = 0
    all_failures: list[str] = []
    for path in sorted(test_dir.glob("test_*.py")):
        print(f"[{path.name}]")
        p, f, fails = _run_module(path)
        total_p += p
        total_f += f
        all_failures += fails
    print(f"\n{total_p} passed, {total_f} failed")
    for msg in all_failures:
        print("\n" + "=" * 70 + "\n" + msg)
    return 1 if total_f else 0


if __name__ == "__main__":
    sys.exit(main())
