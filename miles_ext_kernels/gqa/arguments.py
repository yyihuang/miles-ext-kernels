"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Arguments of the packed causal GQA attention kernels (``--full-attention-backend cake``).

The kernels have no argument of their own and a single variant, whose backward sums dQ with FP32 atomic adds in a
run-dependent order: :func:`validate_args` rejects the cake backend in deterministic mode. Torch-free.
"""

from argparse import Namespace

from miles_ext_kernels._variant import deterministic_mode


def add_arguments(group) -> None:
    """None to register: ``--full-attention-backend cake`` alone selects the kernels."""


def validate_args(args: Namespace) -> None:
    if getattr(args, "full_attention_backend", None) == "cake" and deterministic_mode(args):
        raise ValueError(
            "--full-attention-backend cake is not bit-reproducible: its backward sums dQ with FP32 atomic adds in a "
            "run-dependent order. --deterministic-mode needs --full-attention-backend megatron."
        )
