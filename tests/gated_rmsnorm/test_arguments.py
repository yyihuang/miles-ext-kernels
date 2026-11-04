"""The Cake gated output-norm arguments: registration, variant resolution, rejection and configure (torch-free)."""

import argparse
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from miles_ext_kernels.gated_rmsnorm import arguments

REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse(argv: list[str], **miles_args) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    arguments.add_arguments(parser.add_argument_group("miles-ext-kernels"))
    args = parser.parse_args(argv)
    for name, value in miles_args.items():  # the miles arguments validate_args reads from the same namespace
        setattr(args, name, value)
    return args


def test_the_defaults_validate_and_leave_the_variant_unset():
    args = _parse([])

    arguments.validate_args(args)
    assert args.cake_gated_rmsnorm_variant is None


@pytest.mark.parametrize(
    "deterministic_mode, expected", [(False, "fast"), (True, "det")], ids=["default", "deterministic-mode"]
)
def test_the_cake_backend_resolves_the_variant(deterministic_mode, expected):
    args = _parse([], linear_attention_output_norm_backend="cake", deterministic_mode=deterministic_mode)

    arguments.validate_args(args)
    assert args.cake_gated_rmsnorm_variant == expected


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_an_explicit_variant_is_kept(variant):
    args = _parse(["--cake-gated-rmsnorm-variant", variant], linear_attention_output_norm_backend="cake")

    arguments.validate_args(args)
    assert args.cake_gated_rmsnorm_variant == variant


def test_explicit_fast_under_deterministic_mode_is_rejected():
    args = _parse(
        ["--cake-gated-rmsnorm-variant", "fast"], linear_attention_output_norm_backend="cake", deterministic_mode=True
    )

    with pytest.raises(ValueError, match="--cake-gated-rmsnorm-variant fast is not bit-reproducible"):
        arguments.validate_args(args)


@pytest.mark.parametrize("backend", [None, "fla"])
def test_the_variant_without_the_cake_backend_is_rejected(backend):
    args = _parse(["--cake-gated-rmsnorm-variant", "det"])
    if backend is not None:
        args.linear_attention_output_norm_backend = backend

    with pytest.raises(
        ValueError, match="--cake-gated-rmsnorm-variant requires --linear-attention-output-norm-backend cake"
    ):
        arguments.validate_args(args)


@pytest.fixture
def ops(monkeypatch):
    # a stand-in for the torch-backed ops module: configure must reach it without importing torch here
    module = types.ModuleType("miles_ext_kernels.gated_rmsnorm.ops")
    module.VARIANT = None
    monkeypatch.setitem(sys.modules, "miles_ext_kernels.gated_rmsnorm.ops", module)
    return module


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_configure_applies_the_resolved_variant(ops, variant):
    args = _parse(["--cake-gated-rmsnorm-variant", variant], linear_attention_output_norm_backend="cake")
    arguments.validate_args(args)

    for _ in range(2):  # miles calls it once per layer
        arguments.configure(argparse.Namespace(cake_gated_rmsnorm_variant=args.cake_gated_rmsnorm_variant))
        assert ops.VARIANT == variant


def test_configure_without_the_argument_runs_the_default(ops):
    arguments.configure(argparse.Namespace())
    assert ops.VARIANT == "fast"

    arguments.configure(argparse.Namespace(deterministic_mode=True))
    assert ops.VARIANT == "det"


def test_configure_rejects_an_unknown_variant(ops):
    with pytest.raises(ValueError, match="--cake-gated-rmsnorm-variant must be one of"):
        arguments.configure(argparse.Namespace(cake_gated_rmsnorm_variant="exact"))


def test_the_package_exports_configure_lazily():
    from miles_ext_kernels.gated_rmsnorm import configure

    assert configure is arguments.configure


_NO_TORCH = """
import argparse
import sys

sys.modules["torch"] = None  # any import of torch or tvm_ffi raises ImportError
sys.modules["tvm_ffi"] = None

import miles_ext_kernels.gated_rmsnorm
from miles_ext_kernels.gated_rmsnorm import arguments, configure

parser = argparse.ArgumentParser()
arguments.add_arguments(parser.add_argument_group("miles-ext-kernels"))
args = parser.parse_args([])
args.linear_attention_output_norm_backend = "cake"
arguments.validate_args(args)
assert args.cake_gated_rmsnorm_variant == "fast", args
"""


def test_the_package_and_its_arguments_import_without_torch():
    pythonpath = os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", _NO_TORCH],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": pythonpath},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
