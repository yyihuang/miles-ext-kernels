"""The GQA kernels' arguments, registered and validated through the package hooks miles calls (torch-free)."""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import miles_ext_kernels
from miles_ext_kernels.gqa import arguments

REPO_ROOT = Path(__file__).resolve().parents[2]
REJECTION = "--full-attention-backend cake is not bit-reproducible"

_NO_TORCH_ON_IMPORT = """
import argparse
import sys

import miles_ext_kernels.gqa
from miles_ext_kernels.gqa import arguments

arguments.add_arguments(argparse.ArgumentParser())
arguments.validate_args(argparse.Namespace(full_attention_backend="cake"))
loaded = sorted(n for n in sys.modules if n in ("torch", "tvm_ffi") or n.startswith(("torch.", "miles_ext_kernels.gqa.")))
assert loaded == ["miles_ext_kernels.gqa.arguments"], loaded
"""


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    miles_ext_kernels.add_arguments(parser)
    return parser.parse_args(argv)


def test_the_package_is_registered_and_the_defaults_validate():
    assert arguments in miles_ext_kernels._KERNEL_PACKAGES
    miles_ext_kernels.validate_args(_parse([]))


def test_the_package_registers_no_argument():
    parser = argparse.ArgumentParser()
    before = list(parser._actions)
    arguments.add_arguments(parser.add_argument_group("miles-ext-kernels"))

    assert parser._actions == before


@pytest.mark.parametrize("backend", [None, "megatron", "cake"])
def test_every_backend_validates_outside_deterministic_mode(backend):
    args = argparse.Namespace(deterministic_mode=False)
    if backend is not None:
        args.full_attention_backend = backend

    arguments.validate_args(args)


def test_the_cake_backend_is_rejected_in_deterministic_mode():
    args = argparse.Namespace(full_attention_backend="cake", deterministic_mode=True)

    with pytest.raises(ValueError, match=REJECTION):
        arguments.validate_args(args)


def test_the_typed_configuration_s_deterministic_mode_rejects_the_cake_backend_too():
    # the configuration miles builds after parsing carries the flag as args.backend.deterministic_mode
    args = SimpleNamespace(full_attention_backend="cake", backend=SimpleNamespace(deterministic_mode=True))

    with pytest.raises(ValueError, match=REJECTION):
        arguments.validate_args(args)


def test_megatron_attention_validates_in_deterministic_mode():
    arguments.validate_args(argparse.Namespace(full_attention_backend="megatron", deterministic_mode=True))


def test_importing_the_package_and_validating_does_not_import_torch():
    pythonpath = os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", _NO_TORCH_ON_IMPORT],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": pythonpath},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_the_package_exports_its_entry_points_lazily():
    import miles_ext_kernels.gqa as gqa

    assert set(gqa.__all__) == {"SUPPORTED_CAPABILITIES", "device_arch", "prebuild", "varlen_gqa_attention"}
    assert set(gqa.__all__) <= set(dir(gqa))
    with pytest.raises(AttributeError):
        gqa.not_an_entry_point
