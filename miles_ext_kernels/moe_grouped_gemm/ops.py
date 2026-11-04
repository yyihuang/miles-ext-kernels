"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

The ragged BF16 MoE grouped GEMM miles calls (SM100a / SM103a), on two vendored FlashInfer packages whose libraries
:mod:`miles_ext_kernels.cake_native` builds and loads: ``_vendor/cake_moe_grouped_det_gemm`` (``det``) and
``_vendor/cake/cake_moe_grouped_gemm`` (``fast``).

:func:`grouped_mm` is ``torch._grouped_mm(x, w.transpose(-2, -1), offs=offs)`` for ``x [sum_m, K]``, ``w [E, N, K]``
and the int32 cumulative row ends ``offs [E]`` of the groups, differentiable in ``x`` and ``w``. The kernels read
``offs`` on the device (no host synchronisation); groups may be empty, one row long or of any size. Every product is
an FP32 accumulation of exact BF16 products rounded once to BF16 (the cuBLAS / CUTLASS numerics, not their bits).

``det`` reduces every output in a fixed order: the output and both gradients are bit-identical call to call. ``fast``
gives the same output and input gradient bits; its weight gradient, where it picks the 512-wide k tile (``K`` a
multiple of 512 and at least 4096 rows per group on average), adds the split tail of its last wave with unordered
FP32 atomics instead of a second, ordered reduction launch, so those tiles may differ in the last bit run to run.

A launch prepares its plan on the host from the shapes (no synchronisation) and keeps per-stream workspaces, so do
not capture these calls in a CUDA graph. :func:`prebuild` compiles the programs ahead of the first call.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

from miles_ext_kernels import _variant, cake_native

__all__ = ["SUPPORTED_CAPABILITIES", "VARIANT", "check_grouped_dims", "grouped_mm", "prebuild"]

SUPPORTED_CAPABILITIES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}
# The variant grouped_mm runs when its caller names none; arguments.configure sets it from the parsed arguments.
VARIANT = "fast"
_FAMILY = "moe_grouped_gemm"
# The routes of one forward and backward with BF16 weights: (op, record_name keywords).
_ROUTES = (
    ("fwd", {}),
    ("dgrad", {}),
    ("wgrad", {"out_dtype": "bfloat16", "tile_k": 256}),
    ("wgrad", {"out_dtype": "bfloat16", "tile_k": 512}),
)


@functools.cache
def _det():
    """``(cake_autograd, cake_jit, cake_build)`` of the deterministic package, building through cake_native."""
    from ._vendor.cake_moe_grouped_det_gemm import cake_build

    cake_native.install(cake_build)
    from ._vendor.cake_moe_grouped_det_gemm import cake_autograd, cake_jit

    return cake_autograd, cake_jit, cake_build


@functools.cache
def _fast():
    """``(cake_autograd, cake_jit, cake_build)`` of the non-deterministic package, building through cake_native."""
    from ._vendor.cake.cake_moe_grouped_gemm import cake_build

    cake_native.install(cake_build)
    from ._vendor.cake.cake_moe_grouped_gemm import cake_autograd, cake_jit

    return cake_autograd, cake_jit, cake_build


def _package(variant: str):
    return _det() if variant == "det" else _fast()


def _device_index(device) -> int:
    device = torch.device("cuda") if device is None else torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"the Cake MoE grouped GEMM runs on CUDA devices, got {device}")
    return torch.cuda.current_device() if device.index is None else device.index


@functools.cache
def _arch_of(index: int) -> str:
    major, minor = torch.cuda.get_device_capability(index)
    arch = SUPPORTED_CAPABILITIES.get((major, minor))
    if arch is None:
        raise RuntimeError(
            f"the Cake MoE grouped GEMM runs on SM100a / SM103a (B200, B300, GB300); cuda:{index} has compute "
            f"capability {major}.{minor}"
        )
    return arch


def _check_variant(variant: str | None) -> str:
    variant = VARIANT if variant is None else variant
    if variant not in _variant.VARIANTS:
        raise ValueError(f"variant must be one of {_variant.VARIANTS}, got {variant!r}")
    return variant


def check_grouped_dims(n: int, k: int) -> None:
    """Raise ``ValueError`` unless the programs serve the expert weight ``[E, N, K]`` in all three GEMMs: the forward
    and the weight gradient need ``N`` a multiple of 256, the input and weight gradients ``K`` a multiple of 256."""
    for name, dim in (("output features N", n), ("input features K", k)):
        if dim < 256 or dim % 256:
            raise ValueError(f"the Cake MoE grouped GEMM needs expert {name} that are a multiple of 256, got {dim}")


def grouped_mm(x: torch.Tensor, w: torch.Tensor, offs: torch.Tensor, *, variant: str | None = None) -> torch.Tensor:
    """The ragged grouped GEMM ``Y[offs[e-1]:offs[e]] = X[offs[e-1]:offs[e]] @ W[e].T``: ``[sum_m, N]`` BF16.

    ``x [sum_m, K]`` and ``w [E, N, K]`` are BF16 with unit stride on their last axis, ``offs`` the int32 cumulative
    row ends ``[E]`` on ``x``'s device (``torch._grouped_mm``'s ``offs``). The backward returns ``x.grad`` in BF16 and
    ``w.grad`` in ``w``'s dtype; an empty group gets an exactly zero weight gradient. ``variant`` (default
    :data:`VARIANT`) is ``"det"`` or ``"fast"``; under ``torch.use_deterministic_algorithms(True)`` a ``"fast"`` call
    runs ``"det"`` and says so once. The variant is fixed at the forward call: its backward runs the same package.
    """
    variant = _check_variant(variant)
    if x.dim() != 2 or w.dim() != 3 or x.shape[1] != w.shape[2]:
        raise ValueError(f"grouped_mm takes x [sum_m, K] and w [E, N, K], got {tuple(x.shape)} and {tuple(w.shape)}")
    if x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
        raise ValueError(f"grouped_mm takes BF16 x and w, got {x.dtype} and {w.dtype}")
    if offs.dtype != torch.int32 or offs.dim() != 1 or offs.numel() != w.shape[0]:
        raise ValueError(
            f"offs must be the int32 cumulative row ends [E] of the {w.shape[0]} groups, got {offs.dtype} "
            f"{tuple(offs.shape)}"
        )
    if w.device != x.device or offs.device != x.device:
        raise ValueError(f"x, w and offs must be on one device, got {x.device}, {w.device} and {offs.device}")
    check_grouped_dims(w.shape[1], w.shape[2])
    _arch_of(_device_index(x.device))
    if variant == "fast" and torch.are_deterministic_algorithms_enabled():
        _variant.warn_det_fallback(_FAMILY, "torch.use_deterministic_algorithms(True)")
        variant = "det"
    if variant == "det":
        return _det()[0].cake_grouped_det_mm(x, w, offs)
    return _fast()[0].cake_grouped_mm(x, w, offs, deterministic=False)


def prebuild(*, variant: str | None = None, device=None, max_workers: int | None = None) -> list[Path]:
    """Compile, ahead of their first call, every program :func:`grouped_mm` of ``variant`` (default :data:`VARIANT`)
    can launch on ``device`` (default: the current device) with BF16 weights: the forward, input-gradient and both
    weight-gradient routes, every stage built for its architecture. The stage a call runs depends on its shapes, so
    this compiles them all (about a dozen libraries). Returns the cache paths of the libraries.

    The input gradient also compiles a small probe of the device's memory domains on first use (through tvm-ffi's
    own cache); until it is built the kernels split their work evenly, with the same output bits.
    """
    variant = _check_variant(variant)
    arch = _arch_of(_device_index(device))
    _, cake_jit, cake_build = _package(variant)
    names = []
    for op, keys in _ROUTES:
        if not cake_jit.program_registered(op, arch, **keys):
            continue
        for stage in cake_jit.ROUTES[cake_jit.record_name(op, **keys)]["stages"]:
            if arch in cake_jit.PROGRAMS[stage["program"]]["arches"]:
                names.append(cake_jit.library_name(stage["program"], arch, stage["specializations"]))
    return cake_native.prebuild(cake_build, names, max_workers)
