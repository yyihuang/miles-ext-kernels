"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Arguments of the MoE grouped GEMM (``--moe-grouped-gemm-backend cake``): its variant flag.

The ``fast`` weight gradient reduces part of its tiles in a run-dependent order, so deterministic mode runs ``det``
and refuses an explicit ``fast``. miles calls these kernels from its torchtitan backend only. Registered by
``miles_ext_kernels.add_arguments`` under ``--use-miles-ext-kernels``; miles hands the parsed values to
:func:`configure` where it builds the model. Torch-free: :func:`configure` imports the kernels lazily.
"""

from argparse import Namespace
from importlib import import_module

from miles_ext_kernels import _variant

FLAG = "--cake-moe-grouped-gemm-variant"
BACKEND = "--moe-grouped-gemm-backend cake"


def add_arguments(group) -> None:
    _variant.add_variant_argument(group, FLAG, BACKEND)


def validate_args(args: Namespace) -> None:
    selected = getattr(args, "moe_grouped_gemm_backend", None) == "cake"
    _variant.resolve(args, FLAG, selected, fast_unordered=True, backend=BACKEND)
    if not selected:
        return
    backend = getattr(args, "train_backend", None)
    if backend != "torchtitan":
        raise ValueError(
            f"{BACKEND} replaces the expert GEMMs of miles' torchtitan models: it requires --train-backend "
            f"torchtitan (got {backend!r})"
        )


def configure(args) -> None:
    """Apply the parsed arguments to the kernels: idempotent, safe to call once per model build."""
    ops = import_module("miles_ext_kernels.moe_grouped_gemm.ops")
    ops.VARIANT = _variant.resolve(args, FLAG, True, fast_unordered=True, backend=BACKEND)
