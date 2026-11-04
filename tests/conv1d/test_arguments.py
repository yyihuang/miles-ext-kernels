"""The Cake short-conv arguments: registration, variant resolution, rejection rules and configure (torch-free)."""

import argparse
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from miles_ext_kernels.conv1d import arguments

REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse(argv: list[str], **miles_args) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    arguments.add_arguments(parser.add_argument_group("miles-ext-kernels"))
    args = parser.parse_args(argv)
    for name, value in miles_args.items():  # the miles arguments validate_args reads from the same namespace
        setattr(args, name, value)
    return args


def test_the_defaults_validate_and_leave_the_arguments_unset():
    args = _parse([])

    arguments.validate_args(args)
    assert args.cake_conv1d_variant is None
    assert args.cake_conv1d_silu_grad_preactivation is None


@pytest.mark.parametrize(
    "deterministic_mode, expected", [(False, "fast"), (True, "det")], ids=["default", "deterministic-mode"]
)
def test_the_cake_backend_resolves_the_variant_and_the_silu_policy(deterministic_mode, expected):
    args = _parse([], linear_attention_conv_backend="cake", deterministic_mode=deterministic_mode)

    arguments.validate_args(args)
    assert args.cake_conv1d_variant == expected
    assert args.cake_conv1d_silu_grad_preactivation == "bf16"


@pytest.mark.parametrize("variant", ["det", "fast"])
@pytest.mark.parametrize("policy", ["bf16", "fp32"])
def test_explicit_settings_are_kept(variant, policy):
    args = _parse(
        ["--cake-conv1d-variant", variant, "--cake-conv1d-silu-grad-preactivation", policy],
        linear_attention_conv_backend="cake",
    )

    arguments.validate_args(args)
    assert (args.cake_conv1d_variant, args.cake_conv1d_silu_grad_preactivation) == (variant, policy)


def test_the_typed_config_carries_deterministic_mode_too():
    args = _parse([], linear_attention_conv_backend="cake")
    args.backend = types.SimpleNamespace(deterministic_mode=True)

    arguments.validate_args(args)
    assert args.cake_conv1d_variant == "det"


def test_explicit_fast_under_deterministic_mode_is_rejected():
    args = _parse(["--cake-conv1d-variant", "fast"], linear_attention_conv_backend="cake", deterministic_mode=True)

    with pytest.raises(ValueError, match="--cake-conv1d-variant fast is not bit-reproducible"):
        arguments.validate_args(args)


@pytest.mark.parametrize("backend", [None, "fla"])
@pytest.mark.parametrize(
    "argv, flag",
    [
        (["--cake-conv1d-variant", "det"], "--cake-conv1d-variant"),
        (["--cake-conv1d-silu-grad-preactivation", "fp32"], "--cake-conv1d-silu-grad-preactivation"),
    ],
)
def test_an_argument_without_the_cake_backend_is_rejected(backend, argv, flag):
    args = _parse(argv)
    if backend is not None:
        args.linear_attention_conv_backend = backend

    with pytest.raises(ValueError, match=f"{flag} requires --linear-attention-conv-backend cake"):
        arguments.validate_args(args)


def test_context_parallelism_is_rejected():
    args = _parse([], linear_attention_conv_backend="cake", context_parallel_size=2)

    with pytest.raises(ValueError, match="does not support context parallelism"):
        arguments.validate_args(args)
    # without the cake backend, context parallelism is not this package's concern
    arguments.validate_args(_parse([], linear_attention_conv_backend="fla", context_parallel_size=2))
    arguments.validate_args(_parse([], linear_attention_conv_backend="cake", context_parallel_size=1))


@pytest.fixture
def ops(monkeypatch):
    # a stand-in for the torch-backed ops module: configure must reach it without importing torch here
    module = types.ModuleType("miles_ext_kernels.conv1d.ops")
    module.VARIANT = None
    module.SILU_GRAD_PREACTIVATION = None
    monkeypatch.setitem(sys.modules, "miles_ext_kernels.conv1d.ops", module)
    return module


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_configure_applies_the_resolved_arguments(ops, variant):
    args = _parse(
        ["--cake-conv1d-variant", variant, "--cake-conv1d-silu-grad-preactivation", "fp32"],
        linear_attention_conv_backend="cake",
    )
    arguments.validate_args(args)
    # miles hands configure just this package's values
    values = argparse.Namespace(**{name: getattr(args, name) for name in vars(_parse([]))})

    for _ in range(2):  # miles calls it once per layer
        arguments.configure(values)
        assert (ops.VARIANT, ops.SILU_GRAD_PREACTIVATION) == (variant, "fp32")


def test_configure_without_the_arguments_runs_the_defaults(ops):
    arguments.configure(argparse.Namespace())
    assert (ops.VARIANT, ops.SILU_GRAD_PREACTIVATION) == ("fast", "bf16")

    arguments.configure(argparse.Namespace(deterministic_mode=True))
    assert ops.VARIANT == "det"


def test_configure_rejects_an_unknown_variant(ops):
    with pytest.raises(ValueError, match="--cake-conv1d-variant must be one of"):
        arguments.configure(argparse.Namespace(cake_conv1d_variant="exact"))


def test_the_package_exports_configure_lazily():
    from miles_ext_kernels.conv1d import configure

    assert configure is arguments.configure


_NO_TORCH = """
import argparse
import sys

sys.modules["torch"] = None  # any import of torch or tvm_ffi raises ImportError
sys.modules["tvm_ffi"] = None

import miles_ext_kernels.conv1d
from miles_ext_kernels.conv1d import arguments, configure

parser = argparse.ArgumentParser()
arguments.add_arguments(parser.add_argument_group("miles-ext-kernels"))
args = parser.parse_args([])
args.linear_attention_conv_backend = "cake"
arguments.validate_args(args)
assert args.cake_conv1d_variant == "fast", args
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
