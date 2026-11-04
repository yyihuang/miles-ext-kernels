"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Arguments of the chunked GDN kernels (``--linear-attention-backend cake``).

Registered by ``miles_ext_kernels.add_arguments`` under ``--use-miles-ext-kernels``. ``--cake-gdn-variant`` picks
``det``, the bit-deterministic kernels (the only ones with context parallelism), or ``fast``, the Cake
non-deterministic package. Every launch of ``fast`` is fixed-order, so it repeats bit for bit and deterministic mode
may run it; unset, the variant is ``fast``, or ``det`` under ``--deterministic-mode``. miles hands the parsed values to
:func:`configure` where it builds a ``cake`` GDN layer. Torch-free: :func:`configure` imports the kernels lazily.
"""

from argparse import Namespace
from importlib import import_module

from miles_ext_kernels import _variant

VARIANT_FLAG = "--cake-gdn-variant"
BACKEND = "--linear-attention-backend cake"


def add_arguments(group) -> None:
    _variant.add_variant_argument(group, VARIANT_FLAG, BACKEND, fast_unordered=False)
    group.add_argument(
        "--cake-gdn-recompute-state",
        action="store_true",
        default=False,
        help=(
            "Recompute w/u/h/v_new in the backward instead of keeping them on the autograd tape: less activation "
            "memory, more backward time. det gives the same gradient bits; fast gives gradients within the BF16 "
            "accuracy budgets of the saved backward. Either repeats bit for bit. Only with --linear-attention-backend "
            "cake."
        ),
    )


def validate_args(args: Namespace) -> None:
    """Reject the arguments without the cake backend and resolve ``--cake-gdn-variant`` in place."""
    backend_selected = getattr(args, "linear_attention_backend", None) == "cake"
    if getattr(args, "cake_gdn_recompute_state", False) and not backend_selected:
        raise ValueError(f"--cake-gdn-recompute-state requires {BACKEND}")
    # fast_unordered=False: every launch of the fast package is fixed-order, so deterministic mode may run it
    _variant.resolve(args, VARIANT_FLAG, backend_selected, False, backend=BACKEND)


def configure(args) -> None:
    """Apply the parsed arguments to the kernels: idempotent, safe to call once per layer.

    ``args`` carries the values :func:`validate_args` resolved; a namespace without a variant (no
    ``--use-miles-ext-kernels`` registration, or never validated) runs ``det``.
    """
    variant = getattr(args, "cake_gdn_variant", None) or "det"
    if variant not in _variant.VARIANTS:
        raise ValueError(f"{VARIANT_FLAG} must be one of {_variant.VARIANTS}, not {variant!r}")
    ops = import_module("miles_ext_kernels.gdn.ops")
    ops.SAVE_STATE = not getattr(args, "cake_gdn_recompute_state", False)
    ops.VARIANT = variant
