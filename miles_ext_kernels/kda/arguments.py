"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Arguments of the deterministic KDA kernels (``--kda-backend cake``).

The package ships one variant and no switch of its own, so it registers no argument. :func:`validate_args` rejects,
at startup, the miles settings the kernels do not serve. Torch-free.
"""

from argparse import Namespace


def add_arguments(group) -> None:
    """Register the package's arguments: none (one deterministic variant, no tuning switch)."""


def validate_args(args: Namespace) -> None:
    """Under ``--kda-backend cake``: reject context parallelism and a layout other than ``thd``.

    The kernels run whole documents of one packed sequence on one rank: miles' context-parallel KDA path hands each
    rank a shard of every document with a state passed between ranks, which they do not take, and ``bshd`` is not
    a packed sequence.
    """
    if getattr(args, "kda_backend", None) != "cake":
        return
    context_parallel_size = getattr(args, "context_parallel_size", None) or 1
    if context_parallel_size > 1:
        raise ValueError(
            f"--kda-backend cake does not support context parallelism yet (--context-parallel-size "
            f"{context_parallel_size}); use --kda-backend fla"
        )
    qkv_format = getattr(args, "qkv_format", "thd")
    if qkv_format != "thd":
        raise ValueError(f"--kda-backend cake needs --qkv-format thd (one packed sequence), got {qkv_format!r}")


def configure(args) -> None:
    """Apply the parsed arguments to the kernels: the package has none, so there is nothing to apply."""
