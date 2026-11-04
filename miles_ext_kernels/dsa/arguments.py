"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Arguments of the DSA kernels (``--miles-dsa-kernel-backend cake``).

Registered by ``miles_ext_kernels.add_arguments`` under ``--use-miles-ext-kernels``. ``validate_args`` resolves
``--cake-dsa-variant`` (unset: ``fast``, or ``det`` under ``--deterministic-mode``; an explicit ``fast`` is rejected
there, since its key-side gradients are summed by atomics in a run-dependent order) and writes it back. miles hands
the values to :func:`configure` where it builds a ``cake`` DSA layer. Torch-free: :func:`configure` imports the
kernels lazily.
"""

from argparse import Namespace
from importlib import import_module

from miles_ext_kernels import _variant

BACKEND = "--miles-dsa-kernel-backend cake"
VARIANT_FLAG = "--cake-dsa-variant"
_EXT_FLAGS = ("--cake-dsa-bwd-workspace-bytes", "--cake-dsa-lazy-build")


def add_arguments(group) -> None:
    _variant.add_variant_argument(group, VARIANT_FLAG, BACKEND, fast_unordered=True)
    group.add_argument(
        "--cake-dsa-bwd-workspace-bytes",
        type=int,
        default=None,
        help=(
            "Bytes of the FP32 key-side partials of the det sparse-attention backward: query rows run in chunks "
            f"whose partials fit (default 2 GiB). Only with {BACKEND}."
        ),
    )
    group.add_argument(
        "--cake-dsa-lazy-build",
        action="store_true",
        default=False,
        help=(
            "Build each DSA kernel on its first launch instead of every kernel of the GPU's architecture on the "
            "first launch of any (a kernel first used mid-run then compiles inside a training step). "
            f"Only with {BACKEND}."
        ),
    )


def validate_args(args: Namespace) -> None:
    selected = getattr(args, "miles_dsa_kernel_backend", None) == "cake"
    _variant.resolve(args, VARIANT_FLAG, selected, fast_unordered=True, backend=BACKEND)
    workspace = getattr(args, "cake_dsa_bwd_workspace_bytes", None)
    given = (workspace is not None, bool(getattr(args, "cake_dsa_lazy_build", False)))
    if not selected:
        for flag, flag_given in zip(_EXT_FLAGS, given):
            if flag_given:
                raise ValueError(f"{flag} requires {BACKEND}")
        return
    if workspace is not None and workspace <= 0:
        raise ValueError(f"--cake-dsa-bwd-workspace-bytes must be positive, got {workspace}")


def configure(args) -> None:
    """Apply the parsed arguments to the kernels: idempotent, safe to call once per layer.

    A namespace without the variant (``validate_args`` did not run) selects ``det``.
    """
    variant = getattr(args, "cake_dsa_variant", None) or "det"
    if variant not in _variant.VARIANTS:
        raise ValueError(f"{VARIANT_FLAG} must be one of {_variant.VARIANTS}, not {variant!r}")
    ops = import_module("miles_ext_kernels.dsa.ops")
    kernels = import_module("miles_ext_kernels.dsa._kernels")
    reduction = import_module("miles_ext_kernels.dsa.reduction")
    workspace = getattr(args, "cake_dsa_bwd_workspace_bytes", None)
    ops.VARIANT = variant
    reduction.BWD_WORKSPACE_BYTES = reduction.DEFAULT_WORKSPACE_BYTES if workspace is None else int(workspace)
    kernels.LAZY_BUILD = bool(getattr(args, "cake_dsa_lazy_build", False))
