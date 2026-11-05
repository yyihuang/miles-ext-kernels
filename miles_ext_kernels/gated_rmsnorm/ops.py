"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

SiLU-gated RMSNorm on the vendored Cake kernels (SM100a / SM103a / SM107a).

:func:`rmsnorm_silu_gate` is the output norm of miles' GDN layers: per row of 128 of BF16 ``x`` and of the gate ``z``
(``[T, H, 128]`` or ``[T * H, 128]``, each with its own token and head strides) and a BF16 ``weight`` ``[128]``
shared by every token and head::

    rstd = 1 / sqrt(mean(x^2) + eps)                     # FP32
    y    = bf16(((x * rstd) * weight) * silu(z))         # FP32, one rounding

``det`` runs FlashInfer's deterministic package (``_vendor/cake_rmsnorm_silu_gate_det_train_backend``): the weight
gradient is summed in FP32 per CTA chunk and in FP64 across the chunks in an order fixed by the row count, so it is
bitwise reproducible. ``fast`` runs the non-deterministic package (``_vendor/cake_rmsnorm_silu_gate_train_backend``):
the same forward arithmetic (``y`` is ``det``'s bits) and a one-kernel backward whose ``dx`` and ``dz`` are ``det``'s
bits and whose CTAs add their FP64 column sums with atomics, so the weight gradient may differ between runs, always
within ``det``'s bound ``|dw| * 2^-7 + sum|terms| * 2^-20``. ``fast`` serves every call ``det`` serves; it runs
``det``, with a one-time warning, only while ``torch.use_deterministic_algorithms(True)`` is in effect.

The vendored packages compile their libraries on first use through :mod:`miles_ext_kernels.cake_native`;
:func:`prebuild` builds them before a layer's first step. SM107a needs a CUDA 13.4 or newer toolkit.
"""

from __future__ import annotations

import functools
from types import ModuleType

import torch

from miles_ext_kernels import cake_native
from miles_ext_kernels._variant import VARIANTS, warn_det_fallback

__all__ = ["SUPPORTED_CAPABILITIES", "check_supported", "prebuild", "rmsnorm_silu_gate"]

SUPPORTED_CAPABILITIES = tuple(cake_native.ARCH_BY_CAPABILITY)
HEAD_DIM = 128
# Set by ``miles_ext_kernels.gated_rmsnorm.configure`` (--cake-gated-rmsnorm-variant) and read at call time.
VARIANT = "fast"

_FAMILY = "gated_rmsnorm"
_BACKEND = "--linear-attention-output-norm-backend cake"
# (variant, arch) already built by prebuild
_prebuilt: set[tuple[str, str]] = set()


@functools.cache
def _package(variant: str) -> ModuleType:
    """The vendored package of ``variant``, building its libraries through cake_native."""
    if variant == "det":
        from ._vendor import cake_rmsnorm_silu_gate_det_train_backend as package
    else:
        from ._vendor import cake_rmsnorm_silu_gate_train_backend as package
    cake_native.install(package.cake_build)
    return package


def _resolve_variant(variant: str | None) -> str:
    """``variant`` (default: the configured :data:`VARIANT`); ``det`` for ``fast`` while torch's deterministic
    algorithms are enabled, which the non-deterministic backward refuses."""
    variant = VARIANT if variant is None else variant
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, not {variant!r}")
    if variant == "fast" and torch.are_deterministic_algorithms_enabled():
        warn_det_fallback(_FAMILY, "torch.use_deterministic_algorithms(True)")
        return "det"
    return variant


def _capability(device) -> tuple[int, int]:
    """The compute capability of ``device`` (default: the current CUDA device), checked to be supported and its
    kernels to build with the toolkit."""
    if not torch.cuda.is_available():
        raise ValueError(f"{_BACKEND} needs a CUDA device of compute capability {SUPPORTED_CAPABILITIES}")
    capability = tuple(torch.cuda.get_device_capability(device))
    if capability not in SUPPORTED_CAPABILITIES:
        raise ValueError(
            f"{_BACKEND} runs on compute capability {SUPPORTED_CAPABILITIES} (SM100a / SM103a / SM107a), "
            f"not {capability}; use --linear-attention-output-norm-backend fla"
        )
    cake_native.check_toolkit(cake_native.ARCH_BY_CAPABILITY[capability], "the Cake gated RMSNorm kernels")
    return capability


def check_supported(
    head_dim: int, activation: str, weight_dtype: torch.dtype, *, has_bias: bool = False, device=None
) -> None:
    """Raise ``ValueError`` unless the Cake kernels run this gated output norm on ``device`` (default: the current
    CUDA device): a SiLU gate, rows of 128, a BF16 weight without bias, compute capability 10.0, 10.3 or 10.7;
    ``RuntimeError`` when the toolkit cannot build the device's kernels (SM107a needs CUDA 13.4 or newer)."""
    if activation != "silu":
        raise ValueError(
            f"{_BACKEND} computes RMSNorm(x) * w * SiLU(z), but this model gates its output norm with "
            f"{activation!r}; use --linear-attention-output-norm-backend fla"
        )
    if head_dim != HEAD_DIM:
        raise ValueError(f"{_BACKEND} normalizes rows of {HEAD_DIM}, got a value head dimension of {head_dim}")
    if has_bias:
        raise ValueError(f"{_BACKEND} has no bias")
    if weight_dtype != torch.bfloat16:
        raise ValueError(f"{_BACKEND} takes a BF16 weight, got {weight_dtype}")
    _capability(device)


def prebuild(device=None, *, variant: str | None = None) -> None:
    """Build every library of the variant's package for ``device`` (default: the current CUDA device), so no step
    compiles one. Idempotent."""
    variant = _resolve_variant(variant)
    package = _package(variant)
    jit = package.cake_jit
    arch = jit.arch_for_capability(_capability(device))
    if (variant, arch) in _prebuilt:
        return
    names = [jit.library_name(module, arch) for module in jit.ROUTES[arch].values()]
    cake_native.prebuild(package.cake_build, names)
    _prebuilt.add((variant, arch))


def rmsnorm_silu_gate(
    x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, *, variant: str | None = None
) -> torch.Tensor:
    """Autograd-enabled ``RMSNorm(x) * weight * SiLU(z)`` over the last dimension (128).

    ``x`` and ``z``: BF16 of one ``[..., 128]`` shape (a gate that is a column slice of a wider projection is read in
    place when its strides are multiples of 16 elements, else copied once); ``weight``: BF16 ``[128]``. ``variant``
    (``det`` / ``fast``) defaults to the configured :data:`VARIANT`. Returns BF16 ``y`` of ``x``'s shape; the weight
    gradient comes in BF16. The autograd node saves only ``x``, ``z`` and ``weight``.
    """
    variant = _resolve_variant(variant)
    package = _package(variant)
    if variant == "det":
        return package.cake_rmsnorm_silu_gate_det(x, z, weight, eps)
    return package.cake_rmsnorm_silu_gate(x, z, weight, eps, deterministic=False)
