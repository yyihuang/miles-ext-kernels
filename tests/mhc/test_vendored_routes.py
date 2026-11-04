"""The vendored mHC packages serve what ``miles_ext_kernels.mhc`` relies on (torch-free, no device).

Each package's ``cake_jit.py`` is loaded by file path (its module level imports neither torch nor the package): every
library a route (or the head) can launch on sm_100a / sm_103a has its ``build.json`` entry, so ``mhc.prebuild`` and the
first calls never ask for a library the manifest does not build, and the shapes, constants and architectures of the
three packages are the ones ``miles_ext_kernels.mhc`` validates against.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from miles_ext_kernels.mhc import arguments

VENDOR = Path(arguments.__file__).resolve().parent / "_vendor"
DET, FAST, HEAD = "cake_mhc_det_train_backend", "cake/cake_mhc_train_backend", "cake_mhc_det_train_head_backend"
ARCHES = ("sm_100a", "sm_103a")

if not (VENDOR / "VENDORED.json").is_file():
    pytest.skip("the mHC Cake packages are not vendored in this checkout", allow_module_level=True)


def _cake_jit(package: str):
    path = VENDOR / package / "cake_jit.py"
    spec = importlib.util.spec_from_file_location(f"_mhc_routes_{package.replace('/', '_')}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _libraries(package: str) -> set[str]:
    manifest = json.loads((VENDOR / package / "build.json").read_text())
    assert manifest["schema"] == "cake-build/1"
    return {entry["name"] for entry in manifest["libraries"]}


def test_the_three_packages_are_vendored():
    packages = json.loads((VENDOR / "VENDORED.json").read_text())["packages"]

    assert {Path(DET).name, Path(FAST).name, Path(HEAD).name} <= set(packages)


@pytest.mark.parametrize("package", [DET, FAST])
@pytest.mark.parametrize("arch", ARCHES)
def test_every_route_library_is_in_the_manifest(package, arch):
    jit = _cake_jit(package)
    libraries = _libraries(package)

    assert jit.supported_hidden_sizes(arch) == arguments.HIDDEN_SIZES
    for hidden in arguments.HIDDEN_SIZES:
        for kind in jit.KINDS:
            for record in jit.route_modules(arch, hidden, kind):
                assert jit.MODULES[record]["hidden"] == hidden
                assert jit.library_name(record, arch) in libraries, (package, arch, hidden, kind, record)


@pytest.mark.parametrize("package", [DET, FAST])
def test_the_connection_constants_are_the_ones_miles_runs(package):
    jit = _cake_jit(package)

    assert jit.HIDDEN_SIZES == arguments.HIDDEN_SIZES
    assert jit.STREAMS == arguments.STREAMS
    assert jit.MAPPING_ROWS == 2 * arguments.STREAMS + arguments.STREAMS**2
    # the defaults of miles_ext_kernels.mhc.ops (RMS_EPS, HC_EPS, SINKHORN_ITERS)
    assert (jit.RMS_EPS, jit.HC_EPS, jit.SINKHORN_ITERS) == (1e-5, 1e-6, 20)
    assert {(10, 0): "sm_100a", (10, 3): "sm_103a"}.items() <= jit.ARCH_BY_CAPABILITY.items()


@pytest.mark.parametrize("arch", ARCHES)
def test_the_head_libraries_are_in_the_manifest(arch):
    jit = _cake_jit(HEAD)
    libraries = _libraries(HEAD)

    assert jit.supported_hidden_sizes(arch) == arguments.HIDDEN_SIZES
    assert jit.STREAMS == arguments.STREAMS
    for role in ("head_fwd", "head_bwd"):
        assert jit.library_name(jit.ROLES[role], arch) in libraries
