"""The MoE grouped GEMM's argument hooks (torch-free): its variant flag, resolved as miles' deterministic mode and the
torchtitan backend require, and handed to the kernels by configure."""

import argparse
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from miles_ext_kernels.moe_grouped_gemm import arguments

REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse(argv: list[str], **values) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    arguments.add_arguments(parser.add_argument_group("miles-ext-kernels"))
    args = parser.parse_args(argv)
    for name, value in values.items():
        setattr(args, name, value)
    return args


def test_the_variant_flag_is_registered_unset():
    args = _parse([])

    assert vars(args) == {"cake_moe_grouped_gemm_variant": None}
    with pytest.raises(SystemExit):
        _parse(["--cake-moe-grouped-gemm-variant", "fastest"])


def test_without_the_cake_backend_the_variant_stays_unset():
    args = _parse([], moe_grouped_gemm_backend="torch", train_backend="megatron")

    arguments.validate_args(args)

    assert args.cake_moe_grouped_gemm_variant is None


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_a_variant_without_the_cake_backend_is_rejected(variant):
    args = _parse(["--cake-moe-grouped-gemm-variant", variant], moe_grouped_gemm_backend="torch")

    with pytest.raises(ValueError, match="requires --moe-grouped-gemm-backend cake"):
        arguments.validate_args(args)


@pytest.mark.parametrize(
    "argv, deterministic_mode, expected",
    [
        ([], None, "fast"),
        ([], False, "fast"),
        ([], True, "det"),
        (["--cake-moe-grouped-gemm-variant", "det"], None, "det"),
        (["--cake-moe-grouped-gemm-variant", "det"], True, "det"),
        (["--cake-moe-grouped-gemm-variant", "fast"], None, "fast"),
    ],
)
def test_the_cake_backend_resolves_the_variant_and_writes_it_back(argv, deterministic_mode, expected):
    args = _parse(argv, moe_grouped_gemm_backend="cake", train_backend="torchtitan")
    if deterministic_mode is not None:
        args.deterministic_mode = deterministic_mode

    arguments.validate_args(args)

    assert args.cake_moe_grouped_gemm_variant == expected


def test_an_explicit_fast_variant_under_deterministic_mode_is_rejected():
    args = _parse(
        ["--cake-moe-grouped-gemm-variant", "fast"],
        moe_grouped_gemm_backend="cake",
        train_backend="torchtitan",
        deterministic_mode=True,
    )

    with pytest.raises(ValueError, match="not bit-reproducible"):
        arguments.validate_args(args)


@pytest.mark.parametrize("train_backend", ["megatron", "fsdp", None])
def test_the_cake_backend_outside_torchtitan_is_rejected(train_backend):
    args = _parse([], moe_grouped_gemm_backend="cake")
    if train_backend is not None:
        args.train_backend = train_backend

    with pytest.raises(ValueError, match="requires --train-backend torchtitan"):
        arguments.validate_args(args)


@pytest.mark.parametrize("variant, expected", [(None, "fast"), ("det", "det"), ("fast", "fast")])
def test_configure_sets_the_variant_of_the_kernels(monkeypatch, variant, expected):
    # a stand-in for the torch-side module, which configure imports by name
    ops = types.ModuleType("miles_ext_kernels.moe_grouped_gemm.ops")
    ops.VARIANT = None
    monkeypatch.setitem(sys.modules, ops.__name__, ops)

    arguments.configure(argparse.Namespace(cake_moe_grouped_gemm_variant=variant))
    arguments.configure(argparse.Namespace(cake_moe_grouped_gemm_variant=variant))  # idempotent

    assert ops.VARIANT == expected


_WITHOUT_TORCH = """
import argparse
import sys

sys.modules["torch"] = None  # any import of torch or tvm_ffi raises ImportError
sys.modules["tvm_ffi"] = None

import miles_ext_kernels.moe_grouped_gemm as package
from miles_ext_kernels.moe_grouped_gemm import arguments

assert "grouped_mm" in dir(package) and "prebuild" in dir(package)
parser = argparse.ArgumentParser()
arguments.add_arguments(parser)
args = parser.parse_args(["--cake-moe-grouped-gemm-variant", "det"])
args.moe_grouped_gemm_backend, args.train_backend = "cake", "torchtitan"
arguments.validate_args(args)
assert package.configure is arguments.configure
print("ok")
"""


def test_the_package_and_its_argument_hooks_import_without_torch():
    pythonpath = os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", _WITHOUT_TORCH],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": pythonpath},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
