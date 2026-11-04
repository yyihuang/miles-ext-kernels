"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Arguments of the Cake short-convolution kernels (``--linear-attention-conv-backend cake``).

Registered by ``miles_ext_kernels.add_arguments`` under ``--use-miles-ext-kernels``. miles hands the parsed values
to :func:`configure` where it builds a ``cake`` short convolution. Torch-free: :func:`configure` imports the kernels
lazily.
"""

from argparse import Namespace
from importlib import import_module

from miles_ext_kernels import _variant

BACKEND = "--linear-attention-conv-backend cake"
VARIANT_FLAG = "--cake-conv1d-variant"
SILU_GRAD_FLAG = "--cake-conv1d-silu-grad-preactivation"
SILU_GRAD_PREACTIVATIONS = ("bf16", "fp32")
DEFAULT_SILU_GRAD_PREACTIVATION = "bf16"


def add_arguments(group) -> None:
    _variant.add_variant_argument(group, VARIANT_FLAG, BACKEND, fast_unordered=True)
    group.add_argument(
        SILU_GRAD_FLAG,
        choices=SILU_GRAD_PREACTIVATIONS,
        default=None,
        help=(
            "Where the short-conv backward evaluates the SiLU derivative: 'bf16' at the BF16-rounded "
            "preactivation (flash-linear-attention's rounding point), 'fp32' at the unrounded FP32 preactivation "
            f"with an FP64 weight-gradient sum (a slower program). Default: 'bf16'. Only with {BACKEND}."
        ),
    )


def validate_args(args: Namespace) -> None:
    selected = getattr(args, "linear_attention_conv_backend", None) == "cake"
    # fast sums the weight gradient across CTAs in a run-dependent order
    _variant.resolve(args, VARIANT_FLAG, selected, fast_unordered=True, backend=BACKEND)
    policy = getattr(args, "cake_conv1d_silu_grad_preactivation", None)
    if not selected:
        if policy is not None:
            raise ValueError(f"{SILU_GRAD_FLAG} requires {BACKEND}")
        return
    if policy is None:
        args.cake_conv1d_silu_grad_preactivation = DEFAULT_SILU_GRAD_PREACTIVATION
    elif policy not in SILU_GRAD_PREACTIVATIONS:
        raise ValueError(f"{SILU_GRAD_FLAG} must be one of {SILU_GRAD_PREACTIVATIONS}, not {policy!r}")
    context_parallel_size = getattr(args, "context_parallel_size", None) or 1
    if context_parallel_size > 1:
        raise ValueError(
            f"{BACKEND} does not support context parallelism yet (--context-parallel-size "
            f"{context_parallel_size}): the Cake short conv needs its context-parallel route, which miles does not "
            "run; use --linear-attention-conv-backend fla"
        )


def configure(args) -> None:
    """Apply the parsed arguments to the kernels: idempotent, safe to call once per layer.

    ``args`` holds the values :func:`validate_args` resolved; an unset variant runs ``fast``, or ``det`` when ``args``
    says miles runs in deterministic mode.
    """
    variant = getattr(args, "cake_conv1d_variant", None)
    if variant is None:
        variant = "det" if _variant.deterministic_mode(args) else "fast"
    if variant not in _variant.VARIANTS:
        raise ValueError(f"{VARIANT_FLAG} must be one of {_variant.VARIANTS}, not {variant!r}")
    policy = getattr(args, "cake_conv1d_silu_grad_preactivation", None) or DEFAULT_SILU_GRAD_PREACTIVATION
    if policy not in SILU_GRAD_PREACTIVATIONS:
        raise ValueError(f"{SILU_GRAD_FLAG} must be one of {SILU_GRAD_PREACTIVATIONS}, not {policy!r}")
    ops = import_module("miles_ext_kernels.conv1d.ops")
    ops.VARIANT = variant
    ops.SILU_GRAD_PREACTIVATION = policy
