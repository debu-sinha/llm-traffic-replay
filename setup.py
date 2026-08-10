"""Setuptools hooks for immutable wheel and sdist build provenance."""
from __future__ import annotations

import runpy
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.sdist import sdist as _sdist


ROOT = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT / "traffic_replay"
_PROVENANCE = runpy.run_path(str(PACKAGE_DIR / "_build_provenance.py"))
_FILENAME = _PROVENANCE["PROVENANCE_FILENAME"]
_resolve = _PROVENANCE["build_provenance_for_source"]
_json = _PROVENANCE["provenance_json"]


def _record() -> dict:
    record, _origin, _error = _resolve(PACKAGE_DIR, ROOT)
    return record


class build_py(_build_py):
    def run(self):
        super().run()
        target = Path(self.build_lib) / "traffic_replay" / _FILENAME
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_json(_record()), encoding="utf-8")

    def get_outputs(self, include_bytecode=1):
        outputs = list(super().get_outputs(include_bytecode))
        generated = str(Path(self.build_lib) / "traffic_replay" / _FILENAME)
        if generated not in outputs:
            outputs.append(generated)
        return outputs


class sdist(_sdist):
    def make_release_tree(self, base_dir, files):
        record = _record()
        super().make_release_tree(base_dir, files)
        target = Path(base_dir) / "traffic_replay" / _FILENAME
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_json(record), encoding="utf-8")


setup(cmdclass={"build_py": build_py, "sdist": sdist})
