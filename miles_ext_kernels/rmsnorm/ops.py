"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Zero-centred RMSNorm on the vendored Cake RMSNorm kernels (SM100a / SM103a / SM107a).

:class:`ZeroCenteredRMSNorm` is the input layernorm of miles' GDN layers, ``y = bf16(x * rstd * (1 + w))`` over the
last dimension with ``rstd = 1 / sqrt(mean(x^2) + eps)`` in FP32. It holds ``w`` as its single ``weight`` parameter,
as Transformer Engine's ``zero_centered_gamma`` norm and HF's ``Qwen3NextRMSNorm`` do, so checkpoints and weight
conversion see the same parameter, and hands the kernels the FP32 effective weight ``1 + w`` (their ``_w32`` routes).
The kernels return the weight gradient in FP32; autograd rounds it into the parameter's dtype, as TE's dgamma.
SM107a needs a CUDA 13.4 or newer toolkit to build its kernels.

Both variants go through FlashInfer's non-deterministic package (``_vendor/cake_rmsnorm_train_backend``):
``deterministic=True`` runs its deterministic sibling (``_vendor/cake_rmsnorm_det_train_backend``: a main kernel and
a fixed-order reduce, ``dw`` bitwise reproducible for a given token count and SM count); ``deterministic=False`` runs
the same forward programs and, per token-count window, a single-kernel backward whose CTAs add their ``dw`` sums with
FP64 atomics (``dw`` not bitwise reproducible; ``dx`` reproducible, within one BF16 ulp of ``det``'s) or ``det``'s
kernel pair. ``fast`` serves every hidden size ``det`` serves; it runs ``det``, with a one-time warning, only while
``torch.use_deterministic_algorithms(True)`` is in effect.

Only hidden sizes with FP32-weight routes in both packages run (:func:`supported_hidden_sizes`); the module raises at
construction for any other. The kernels pick a program per token-count window, so the module builds every program of
its routes when it is constructed (:func:`prebuild`) rather than when a step first reaches a window.
"""

from __future__ import annotations

import functools
from types import ModuleType

import torch

from miles_ext_kernels import cake_native
from miles_ext_kernels._variant import VARIANTS, warn_det_fallback

__all__ = [
    "SUPPORTED_CAPABILITIES",
    "ZeroCenteredRMSNorm",
    "check_supported",
    "prebuild",
    "supported_hidden_sizes",
    "zero_centered_rmsnorm",
]

SUPPORTED_CAPABILITIES = tuple(cake_native.ARCH_BY_CAPABILITY)
# Set by ``miles_ext_kernels.rmsnorm.configure`` (--cake-rmsnorm-variant) and read when a module is built and called.
VARIANT = "fast"

_ARCH_OF = cake_native.ARCH_BY_CAPABILITY
_FAMILY = "rmsnorm"
_BACKEND = "--linear-attention-input-norm cake"
# (variant, arch, hidden) already built by prebuild
_prebuilt: set[tuple[str, str, int]] = set()


@functools.cache
def _packages() -> tuple[ModuleType, ModuleType]:
    """``(det, fast)``: the vendored packages, building their libraries through cake_native. ``fast`` imports
    ``det`` as its sibling for ``deterministic=True``."""
    from ._vendor import cake_rmsnorm_det_train_backend as det
    from ._vendor import cake_rmsnorm_train_backend as fast

    cake_native.install(det.cake_build)
    cake_native.install(fast.cake_build)
    return det, fast


def _resolve_variant(variant: str | None) -> str:
    """``variant`` (default: the configured :data:`VARIANT`); ``det`` for ``fast`` while torch's deterministic
    algorithms are enabled, which the non-deterministic package refuses."""
    variant = VARIANT if variant is None else variant
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, not {variant!r}")
    if variant == "fast" and torch.are_deterministic_algorithms_enabled():
        warn_det_fallback(_FAMILY, "torch.use_deterministic_algorithms(True)")
        return "det"
    return variant


def supported_hidden_sizes(arch: str) -> tuple[int, ...]:
    """The hidden sizes whose FP32-weight forward and backward both packages ship for ``arch``."""
    det, fast = _packages()
    return tuple(
        hidden
        for hidden in det.cake_jit.HIDDEN_SIZES
        if all(
            "f32" in package.cake_jit.weight_dtypes(arch, hidden, kind, False)
            for package in (det, fast)
            for kind in ("fwd", "bwd")
        )
    )


def _arch(device) -> str:
    """The architecture of ``device`` (default: the current CUDA device), checked to be supported and its kernels to
    build with the toolkit."""
    if not torch.cuda.is_available():
        raise ValueError(f"{_BACKEND} needs a CUDA device of compute capability {SUPPORTED_CAPABILITIES}")
    capability = tuple(torch.cuda.get_device_capability(device))
    if capability not in _ARCH_OF:
        raise ValueError(
            f"{_BACKEND} runs on compute capability {SUPPORTED_CAPABILITIES} (SM100a / SM103a / SM107a), "
            f"not {capability}; use --linear-attention-input-norm te"
        )
    arch = _ARCH_OF[capability]
    cake_native.check_toolkit(arch, "the Cake RMSNorm kernels")
    return arch


def check_supported(hidden_size: int, device=None) -> None:
    """Raise ``ValueError`` unless the Cake kernels normalize ``hidden_size`` features on ``device`` (default: the
    current CUDA device). The hidden size is checked first, so a CPU-only process gets that error too. ``RuntimeError``
    when the toolkit cannot build the device's kernels (SM107a needs CUDA 13.4 or newer)."""
    sizes = {arch: supported_hidden_sizes(arch) for arch in _ARCH_OF.values()}
    if not any(hidden_size in supported for supported in sizes.values()):
        raise ValueError(
            f"{_BACKEND} supports hidden sizes {sorted(set().union(*sizes.values()))}, not {hidden_size}; "
            "use --linear-attention-input-norm te"
        )
    arch = _arch(device)
    if hidden_size not in sizes[arch]:
        raise ValueError(
            f"{_BACKEND} supports hidden sizes {list(sizes[arch])} on {arch}, not {hidden_size}; "
            "use --linear-attention-input-norm te"
        )


def prebuild(hidden_size: int, device=None, *, variant: str | None = None) -> None:
    """Build every program of the FP32-weight forward and backward routes of ``hidden_size`` that ``variant`` runs
    on ``device`` (default: the current CUDA device), one per token-count window. Idempotent."""
    variant = _resolve_variant(variant)
    arch = _arch(device)
    key = (variant, arch, int(hidden_size))
    if key in _prebuilt:
        return
    det, fast = _packages()
    package = det if variant == "det" else fast
    jit = package.cake_jit
    names = [
        jit.library_name(module, arch)
        for kind in ("fwd", "bwd")
        for module in jit.route_modules(arch, int(hidden_size), kind, False, "f32")
    ]
    cake_native.prebuild(package.cake_build, names)
    _prebuilt.add(key)


def zero_centered_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, *, variant: str | None = None
) -> torch.Tensor:
    """Autograd-enabled ``bf16(x * rstd * (1 + weight))`` over the last dimension of BF16 ``x`` ``[..., H]``.

    The leading dimensions of ``x`` collapse into rows (miles hands ``[T, 1, H]``); its rows may be strided
    (``stride(-2)`` a multiple of 8 elements, 32-byte aligned storage) and are copied once otherwise. ``weight``
    ``[H]`` is ``w`` of the ``(1 + w)`` scale, in any float dtype; its gradient comes in its dtype. ``variant``
    (``det`` / ``fast``) defaults to the configured :data:`VARIANT`. Returns BF16 ``y`` of ``x``'s shape.
    """
    variant = _resolve_variant(variant)
    hidden = x.shape[-1]
    _, fast = _packages()
    y = fast.cake_rmsnorm(x.reshape(-1, hidden), 1.0 + weight.float(), eps, deterministic=variant == "det")
    return y.view(x.shape)


class ZeroCenteredRMSNorm(torch.nn.Module):
    """``y = bf16(x * rstd * (1 + weight))`` over the last dimension, ``weight`` initialized to zeros.

    Raises ``ValueError`` at construction for a hidden size or device the kernels do not serve, then builds the
    programs it runs (:func:`prebuild`). ``variant`` pins ``det`` / ``fast``; ``None`` runs the configured
    :data:`VARIANT`, read at construction (prebuild) and at every call.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        params_dtype: torch.dtype = torch.bfloat16,
        device=None,
        variant: str | None = None,
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else None
        check_supported(hidden_size, device)
        self.hidden_size = int(hidden_size)
        self.eps = float(eps)
        self.variant = variant
        self.weight = torch.nn.Parameter(torch.zeros(self.hidden_size, dtype=params_dtype, device=device))
        prebuild(self.hidden_size, device, variant=variant)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.hidden_size:
            raise ValueError(f"expected [..., {self.hidden_size}] input, got {tuple(x.shape)}")
        return zero_centered_rmsnorm(x, self.weight, self.eps, variant=self.variant)

    def extra_repr(self) -> str:
        return f"{self.hidden_size}, eps={self.eps}"
