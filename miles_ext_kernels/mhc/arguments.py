"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Arguments of the Cake mHC kernels (``--mhc-backend cake``).

Registered by ``miles_ext_kernels.add_arguments`` under ``--use-miles-ext-kernels``. miles hands the parsed values
to :func:`configure` where it builds its ``cake`` hyper-connection layers. Torch-free: :func:`configure` imports the
kernels lazily.
"""

from argparse import Namespace
from importlib import import_module

from .. import _variant

FLAG = "--cake-mhc-variant"
BACKEND = "--mhc-backend cake"
# The connection shapes the kernels are built for: four residual streams of one of these hidden sizes.
STREAMS = 4
HIDDEN_SIZES = (4096, 5120, 7168)


def add_arguments(group) -> None:
    _variant.add_variant_argument(group, FLAG, BACKEND, fast_unordered=True)


def validate_args(args: Namespace) -> None:
    """Resolve ``--cake-mhc-variant`` (written back as ``det`` or ``fast``) and check the model's hidden size.

    ``fast`` adds the ``dfn`` partials with FP32 atomics, so an explicit ``fast`` under deterministic mode is
    rejected, as is the flag without ``--mhc-backend cake``.
    """
    selected = getattr(args, "mhc_backend", None) == "cake"
    _variant.resolve(args, FLAG, selected, fast_unordered=True, backend=BACKEND)
    if not selected:
        return
    hidden = getattr(args, "hidden_size", None)
    if hidden is not None and hidden not in HIDDEN_SIZES:
        raise ValueError(f"{BACKEND} runs the hidden sizes {HIDDEN_SIZES}, not --hidden-size {hidden}")


def configure(args) -> None:
    """Apply the parsed arguments to the kernels: idempotent, safe to call once per layer.

    An unset variant (a namespace :func:`validate_args` did not resolve) runs ``fast``, which itself runs ``det``
    while torch's deterministic algorithms are enabled.
    """
    variant = getattr(args, "cake_mhc_variant", None)
    if variant is not None and variant not in _variant.VARIANTS:
        raise ValueError(f"{FLAG} must be one of {_variant.VARIANTS}, not {variant!r}")
    ops = import_module("miles_ext_kernels.mhc.ops")
    ops.VARIANT = variant or "fast"
