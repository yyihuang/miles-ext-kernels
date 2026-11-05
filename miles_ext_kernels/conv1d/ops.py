"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Packed causal depthwise conv1d + SiLU (kernel size 4) on the vendored Cake kernels (SM100a / SM103a / SM107a).

:func:`causal_conv1d` is the short convolution of miles' GDN and KDA layers: BF16 ``x`` ``[1, S, C]`` (or
``[S, C]``) holds the packed documents ``cu_seqlens`` delimits, ``weight`` ``[C, 4]`` the FP32 or BF16 taps
(``weight[:, 3]`` multiplies the current token), no bias::

    y[t, c] = bf16(silu(sum_k weight[c, k] * x[t - 3 + k, c]))      # x of another document is 0

``det`` runs FlashInfer's deterministic package (``_vendor/cake_causal_conv1d_det_train_backend``): its backward
writes per-CTA partial rows of the weight gradient and a second launch sums them in a fixed order, so the gradients
are bitwise reproducible for a given ``(S, C, device)``. ``fast`` runs the non-deterministic package
(``_vendor/cake/cake_causal_conv1d_train_backend``): the same forward programs (``y`` is ``det``'s bits) and a
one-launch backward whose ``dx`` is ``det``'s bits and whose weight gradient is summed across CTAs with unordered FP32
reductions (FP64 under ``silu_grad_preactivation="fp32"``), within ``det``'s error budget. Both packages serve every
route miles uses (FP32 or BF16 taps, either SiLU' policy), so ``fast`` never falls back per call; it runs ``det``,
with a one-time warning, only while ``torch.use_deterministic_algorithms(True)`` is in effect.

The vendored packages compile their libraries on first use through :mod:`miles_ext_kernels.cake_native`;
:func:`prebuild` builds the ones a layer runs before its first step. SM107a needs a CUDA 13.4 or newer toolkit.
"""

from __future__ import annotations

import functools
from types import ModuleType

import torch

from miles_ext_kernels import cake_native
from miles_ext_kernels._variant import VARIANTS, warn_det_fallback

__all__ = ["SUPPORTED_CAPABILITIES", "causal_conv1d", "check_supported", "prebuild"]

SUPPORTED_CAPABILITIES = tuple(cake_native.ARCH_BY_CAPABILITY)
KERNEL_SIZE = 4
CHANNEL_MULTIPLE = 8
SILU_GRAD_PREACTIVATIONS = ("bf16", "fp32")
TAP_DTYPES = (torch.float32, torch.bfloat16)
# Set by ``miles_ext_kernels.conv1d.configure`` (--cake-conv1d-variant, --cake-conv1d-silu-grad-preactivation) and
# read at call time.
VARIANT = "fast"
SILU_GRAD_PREACTIVATION = "bf16"

_FAMILY = "conv1d"
_BACKEND = "--linear-attention-conv-backend cake"
# (variant, arch, forward route, backward route) already built by prebuild
_prebuilt: set[tuple[str, str, str, str]] = set()


@functools.cache
def _package(variant: str) -> ModuleType:
    """The vendored package of ``variant``, building its libraries through cake_native."""
    if variant == "det":
        from ._vendor import cake_causal_conv1d_det_train_backend as package
    else:
        from ._vendor.cake import cake_causal_conv1d_train_backend as package
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


def _policy(silu_grad_preactivation: str | None) -> str:
    policy = SILU_GRAD_PREACTIVATION if silu_grad_preactivation is None else silu_grad_preactivation
    if policy not in SILU_GRAD_PREACTIVATIONS:
        raise ValueError(f"silu_grad_preactivation must be one of {SILU_GRAD_PREACTIVATIONS}, not {policy!r}")
    return policy


def _capability(device) -> tuple[int, int]:
    """The compute capability of ``device`` (default: the current CUDA device), checked to be supported and its
    kernels to build with the toolkit."""
    if not torch.cuda.is_available():
        raise ValueError(f"{_BACKEND} needs a CUDA device of compute capability {SUPPORTED_CAPABILITIES}")
    capability = tuple(torch.cuda.get_device_capability(device))
    if capability not in SUPPORTED_CAPABILITIES:
        raise ValueError(
            f"{_BACKEND} runs on compute capability {SUPPORTED_CAPABILITIES} (SM100a / SM103a / SM107a), "
            f"not {capability}; use --linear-attention-conv-backend fla"
        )
    cake_native.check_toolkit(cake_native.ARCH_BY_CAPABILITY[capability], "the Cake short-conv kernels")
    return capability


def check_supported(channels: int, kernel_size: int, activation: str, device=None) -> None:
    """Raise ``ValueError`` unless the Cake kernels run this short convolution on ``device`` (default: the current
    CUDA device): kernel size 4, SiLU, a channel count that is a multiple of 8, compute capability 10.0, 10.3 or 10.7;
    ``RuntimeError`` when the toolkit cannot build the device's kernels (SM107a needs CUDA 13.4 or newer)."""
    if kernel_size != KERNEL_SIZE:
        raise ValueError(f"{_BACKEND} needs kernel size {KERNEL_SIZE}, got {kernel_size}")
    if activation != "silu":
        raise ValueError(f"{_BACKEND} applies SiLU, got activation {activation!r}")
    if channels % CHANNEL_MULTIPLE:
        raise ValueError(f"{_BACKEND} needs a channel count that is a multiple of {CHANNEL_MULTIPLE}, got {channels}")
    _capability(device)


def prebuild(
    weight_dtype: torch.dtype,
    device=None,
    *,
    variant: str | None = None,
    silu_grad_preactivation: str | None = None,
) -> None:
    """Build the libraries :func:`causal_conv1d` runs for taps of ``weight_dtype`` on ``device`` (default: the
    current CUDA device): every program of its forward and backward routes, so no step compiles one. Idempotent."""
    if weight_dtype not in TAP_DTYPES:
        raise ValueError(f"{_BACKEND} takes FP32 or BF16 taps, got {weight_dtype}")
    variant = _resolve_variant(variant)
    policy = _policy(silu_grad_preactivation)
    package = _package(variant)
    jit = package.cake_jit
    arch = jit.arch_for_capability(_capability(device))
    bf16 = weight_dtype == torch.bfloat16
    routes = (
        jit.forward_route(bias=False, bf16_taps=bf16),
        jit.backward_route(bias=False, fp32_grad=policy == "fp32", bf16_taps_or_dweight=bf16),
    )
    key = (variant, arch, *routes)
    if key in _prebuilt:
        return
    names = []
    for route in routes:
        for module in jit.route_modules(arch, route):
            names.append(jit.library_name(module, arch))
            reduce = jit.MODULES[module].get("reduce")  # the det backward's fixed-order reduce program
            if reduce:
                names.append(jit.library_name(reduce, arch))
    cake_native.prebuild(package.cake_build, names)
    _prebuilt.add(key)


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    *,
    variant: str | None = None,
    silu_grad_preactivation: str | None = None,
) -> torch.Tensor:
    """Autograd-enabled ``silu(causal depthwise conv1d)`` of the packed documents of ``x``.

    ``x``: BF16 ``[1, S, C]`` or ``[S, C]`` (copied once when not contiguous); ``weight``: FP32 or BF16 ``[C, 4]``;
    ``cu_seqlens``: ``[num_docs + 1]`` prefix sums of the document lengths on ``x``'s device (``None``: one
    document). ``variant`` (``det`` / ``fast``) and ``silu_grad_preactivation`` (``bf16`` / ``fp32``) default to the
    configured :data:`VARIANT` and :data:`SILU_GRAD_PREACTIVATION`. Returns BF16 ``y`` of ``x``'s shape; the weight
    gradient comes in ``weight``'s dtype.
    """
    if x.dim() == 3 and x.shape[0] != 1:
        raise ValueError(
            f"{_BACKEND} takes one packed batch [1, S, C] whose documents cu_seqlens delimits, got shape "
            f"{tuple(x.shape)}"
        )
    variant = _resolve_variant(variant)
    policy = _policy(silu_grad_preactivation)
    x = x.contiguous()
    if cu_seqlens is None:
        cu_seqlens = torch.tensor([0, x.shape[-2]], dtype=torch.int32, device=x.device)
    else:
        cu_seqlens = cu_seqlens.to(torch.int32).contiguous()
    weight = weight.contiguous()
    package = _package(variant)
    if variant == "det":
        return package.cake_causal_conv1d_det(x, weight, cu_seqlens, None, silu_grad_preactivation=policy)
    return package.cake_causal_conv1d(x, weight, cu_seqlens, None, silu_grad_preactivation=policy, deterministic=False)
