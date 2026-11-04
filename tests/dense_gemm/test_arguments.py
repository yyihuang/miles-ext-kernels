"""The dense projection GEMMs' argument hooks (torch-free): no arguments of their own, and the cake backends they serve
run only under miles' Megatron train backend."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pytest

from miles_ext_kernels.dense_gemm import arguments

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DESTS = list(arguments.BACKENDS)


def test_no_argument_is_registered_and_the_defaults_validate():
    parser = argparse.ArgumentParser()
    arguments.add_arguments(parser.add_argument_group("miles-ext-kernels"))
    args = parser.parse_args([])

    assert vars(args) == {}
    arguments.validate_args(args)


@pytest.mark.parametrize("dest", BACKEND_DESTS)
@pytest.mark.parametrize("train_backend", ["fsdp", "torchtitan", None])
def test_a_cake_backend_outside_megatron_is_rejected(dest, train_backend):
    args = argparse.Namespace(**{dest: "cake"})
    if train_backend is not None:
        args.train_backend = train_backend

    with pytest.raises(ValueError, match="requires --train-backend megatron"):
        arguments.validate_args(args)


@pytest.mark.parametrize("dest", BACKEND_DESTS)
@pytest.mark.parametrize("deterministic_mode", [False, True])
def test_a_cake_backend_under_megatron_is_accepted_in_either_mode(dest, deterministic_mode):
    # one deterministic variant: deterministic mode changes nothing
    args = argparse.Namespace(**{dest: "cake"}, train_backend="megatron", deterministic_mode=deterministic_mode)

    arguments.validate_args(args)


def test_the_other_backends_are_left_to_miles():
    args = argparse.Namespace(
        batched_projection_gemm_backend="torch", moe_router_gemm_backend="megatron", train_backend="fsdp"
    )

    arguments.validate_args(args)


_WITHOUT_TORCH = """
import argparse
import sys

sys.modules["torch"] = None  # any import of torch or tvm_ffi raises ImportError
sys.modules["tvm_ffi"] = None

import miles_ext_kernels.dense_gemm as package
from miles_ext_kernels.dense_gemm import arguments

assert "router_gating" in dir(package) and "batched_projection" in dir(package)
args = argparse.Namespace(
    batched_projection_gemm_backend="cake", moe_router_gemm_backend="cake", train_backend="megatron"
)
arguments.validate_args(args)
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
