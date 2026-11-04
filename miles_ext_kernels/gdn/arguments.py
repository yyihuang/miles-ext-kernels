"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Arguments of the deterministic chunked GDN kernels (``--linear-attention-backend cake``).

Registered by ``miles_ext_kernels.add_arguments`` under ``--use-miles-ext-kernels``. miles hands the parsed values
to :func:`configure` where it builds a ``cake`` GDN layer. Torch-free: :func:`configure` imports the kernels lazily.
"""

from argparse import Namespace
from importlib import import_module


def add_arguments(group) -> None:
    group.add_argument(
        "--cake-gdn-recompute-state",
        action="store_true",
        default=False,
        help=(
            "Recompute w/u/h/v_new in the backward instead of keeping them on the autograd tape: same gradient bits, "
            "less activation memory, more backward time. Only with --linear-attention-backend cake."
        ),
    )


def validate_args(args: Namespace) -> None:
    if getattr(args, "cake_gdn_recompute_state", False) and getattr(args, "linear_attention_backend", None) != "cake":
        raise ValueError("--cake-gdn-recompute-state requires --linear-attention-backend cake")


def configure(args) -> None:
    """Apply the parsed arguments to the kernels: idempotent, safe to call once per layer."""
    ops = import_module("miles_ext_kernels.gdn.ops")
    ops.SAVE_STATE = not getattr(args, "cake_gdn_recompute_state", False)
