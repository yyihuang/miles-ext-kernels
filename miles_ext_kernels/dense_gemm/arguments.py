"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Arguments of the dense projection GEMMs (``--batched-projection-gemm-backend cake``, ``--moe-router-gemm-backend
cake``).

They have none of their own: both GEMMs are deterministic only, and the miles flags select them. :func:`validate_args`
checks that the selected backend runs where miles calls these kernels, Megatron's model code. The model-dependent
conditions (the router dtype and bias, FP8, CUDA graphs) are checked where miles builds the layers, from their
configuration. Torch-free.
"""

from argparse import Namespace

# the miles backend flags that select these kernels, by their argument destination
BACKENDS = {
    "batched_projection_gemm_backend": "--batched-projection-gemm-backend cake",
    "moe_router_gemm_backend": "--moe-router-gemm-backend cake",
}


def add_arguments(group) -> None:
    """The dense projection GEMMs register no arguments: they have one (deterministic) variant and no options."""


def validate_args(args: Namespace) -> None:
    for dest, flag in BACKENDS.items():
        if getattr(args, dest, None) != "cake":
            continue
        backend = getattr(args, "train_backend", None)
        if backend != "megatron":
            raise ValueError(
                f"{flag} replaces a GEMM of miles' Megatron model code: it requires --train-backend megatron "
                f"(got {backend!r})"
            )
