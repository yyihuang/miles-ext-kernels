"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Arguments of the Cake input RMSNorm kernels (``--linear-attention-input-norm cake``).

Registered by ``miles_ext_kernels.add_arguments`` under ``--use-miles-ext-kernels``. miles hands the parsed values
to :func:`configure` where it builds a ``cake`` input norm. Torch-free: :func:`configure` imports the kernels lazily.
"""

from argparse import Namespace
from importlib import import_module

from miles_ext_kernels import _variant

BACKEND = "--linear-attention-input-norm cake"
VARIANT_FLAG = "--cake-rmsnorm-variant"


def add_arguments(group) -> None:
    _variant.add_variant_argument(group, VARIANT_FLAG, BACKEND)


def validate_args(args: Namespace) -> None:
    selected = getattr(args, "linear_attention_input_norm", None) == "cake"
    # fast adds the weight gradient's partial sums with FP64 atomics in some token windows, in a run-dependent order
    _variant.resolve(args, VARIANT_FLAG, selected, fast_unordered=True, backend=BACKEND)


def configure(args) -> None:
    """Apply the parsed arguments to the kernels: idempotent, safe to call once per layer.

    ``args`` holds the values :func:`validate_args` resolved; an unset variant runs ``fast``, or ``det`` when ``args``
    says miles runs in deterministic mode.
    """
    variant = getattr(args, "cake_rmsnorm_variant", None)
    if variant is None:
        variant = "det" if _variant.deterministic_mode(args) else "fast"
    if variant not in _variant.VARIANTS:
        raise ValueError(f"{VARIANT_FLAG} must be one of {_variant.VARIANTS}, not {variant!r}")
    import_module("miles_ext_kernels.rmsnorm.ops").VARIANT = variant
