"""Packaging of the GDN kernels: torch-free imports, and every kernel source the registry names ships (torch-free)."""

import fnmatch
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GDN = REPO_ROOT / "miles_ext_kernels" / "gdn"

_NO_TORCH_ON_IMPORT = """
import argparse
import sys
from pathlib import Path

import miles_ext_kernels
import miles_ext_kernels.gdn
from miles_ext_kernels.gdn import arguments

assert Path(miles_ext_kernels.__file__).resolve().parents[1] == Path(sys.argv[1]).resolve(), miles_ext_kernels.__file__
miles_ext_kernels.add_arguments(argparse.ArgumentParser())
loaded = sorted(n for n in sys.modules if n == "torch" or n.startswith(("torch.", "miles_ext_kernels.gdn.")))
assert loaded == ["miles_ext_kernels.gdn.arguments"], loaded
"""


def _registry_sources() -> set[str]:
    sources = set()
    for record in json.loads((GDN / "registry.json").read_text()).values():
        sources.update(record.get("sources", ()))  # a stage built differently per arch has them in per_arch only
        for per_arch in (record.get("per_arch") or {}).values():
            sources.update(per_arch.get("sources", ()))
    return sources


def test_importing_the_package_and_registering_its_arguments_does_not_import_torch():
    pythonpath = os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", _NO_TORCH_ON_IMPORT, str(REPO_ROOT)],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": pythonpath},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_every_source_the_registry_names_is_under_csrc():
    sources = _registry_sources()

    assert sources
    assert sorted(source for source in sources if not (GDN / "csrc" / source).is_file()) == []


def test_the_package_data_ships_the_registry_and_every_kernel_source():
    tomllib = pytest.importorskip("tomllib")
    setuptools = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["tool"]["setuptools"]
    packages = setuptools["packages"]["find"]["include"]
    assert any(fnmatch.fnmatch("miles_ext_kernels.gdn", pattern) for pattern in packages)

    shipped = set()
    for pattern in setuptools["package-data"]["*"]:
        shipped.update(path.relative_to(GDN) for path in GDN.glob(pattern) if path.is_file())

    required = {Path("registry.json"), *(Path("csrc") / source for source in _registry_sources())}
    assert sorted(map(str, required - shipped)) == []
    # the headers the bindings include ship too: every file under csrc/
    csrc = {path.relative_to(GDN) for path in (GDN / "csrc").rglob("*") if path.is_file()}
    assert sorted(map(str, csrc - shipped)) == []
