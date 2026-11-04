"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

The entry points of the Cake four-stream mHC training kernels (SM100a / SM103a).

The function of one connection, per token ``t`` (FP32 arithmetic, BF16 only at the activation boundaries)::

    u = float32(X[t]).reshape(4 H);  r = rsqrt(mean(u * u) + rms_eps);  z = (u r) @ fn^T    # [24]
    pre  = sigmoid(z[0:4] scale[0] + base[0:4]) + hc_eps
    post = 2 sigmoid(z[4:8] scale[1] + base[4:8])
    C    = Sinkhorn(z[8:24] scale[2] + base[8:24])     # 4 x 4: row softmax + hc_eps, then sinkhorn_iters
                                                       # column / row normalisations, each sum + hc_eps
    U[t]    = BF16(sum_i pre_i X[t, i])                                  # mhc_prepare
    Y[t, j] = BF16(post_j B[t] + sum_i C_ij X[t, i])                     # mhc_finish, B = branch(U)

The coefficients stay FP32 (Megatron's hyper-connection module casts them to the activation dtype first). Tensor
contract: ``X`` BF16 ``[T, 4, H]``, ``fn`` FP32 ``[24, 4 H]``, ``scale`` FP32 ``[3]``, ``base`` FP32 ``[24]``,
``B`` BF16 ``[T, H]``, ``H`` in :data:`HIDDEN_SIZES`, all contiguous with 32-byte aligned storage: the kernels read
every operand in place, and an operand they cannot read is a ``ValueError``, never a silent copy. ``T`` is a run-time
scalar (``T == 0`` launches nothing).

The kernels are three vendored FlashInfer packages (``_vendor/``, byte-identical, compiled with nvcc on first use
through :mod:`miles_ext_kernels.cake_native`):

* ``cake_mhc_det_train_backend`` (``det``): forward, and a backward whose ``dfn`` / ``dscale`` / ``dbase`` are
  fixed-order reductions, bitwise reproducible for a token count on a device.
* ``cake/cake_mhc_train_backend`` (``fast``): copies of the ``det`` forward programs (the same bits) and a backward
  that adds the ``dfn`` partial of every (column slab, token chunk) unit with FP32 atomics, so only the summation
  order of ``dfn`` varies from run to run (each element within ``P <= 32`` FP32 roundings of the fixed-order sum of
  its per-chunk partials). ``dX`` is the ``det`` bits; ``dscale`` / ``dbase`` are the ``det`` package's per-CTA
  reduction, whose low-order bits differ from ``det`` at hidden 4096 above 14927 tokens (``det`` sums them over
  8-token group rows there).
* ``cake_mhc_det_train_head_backend``: the model head's stream mean, bit for bit torch's BF16 ``X.mean(dim=1)`` and
  its gradient (no variant).

:func:`mhc_prepare` / :func:`mhc_finish` route between the first two packages as FlashInfer's
``flashinfer/experimental/cake/cake_mhc_train.py`` facade does (the facade itself is not vendored: it imports
FlashInfer's API logging). ``fast`` does not serve a call made while torch's deterministic algorithms are enabled:
that call runs ``det`` and :func:`miles_ext_kernels._variant.warn_det_fallback` logs it once. A ``fast`` connection
whose backward runs after the deterministic algorithms were enabled raises PyTorch's alert for an operation without a
deterministic implementation. ``--cake-mhc-variant`` (:func:`miles_ext_kernels.mhc.configure`) sets the variant of
calls that name none.

Each package keeps one default workspace per device, hidden size and token-count bucket, shared by every call on the
device: the connections of a model run one after another on the current stream. :func:`prebuild` compiles every
library the routes of one hidden size can launch, ahead of the first step (all ranks may call it at once).
"""

from __future__ import annotations

import sys
from functools import cache
from types import ModuleType
from typing import Any

import torch

from .. import _variant, cake_native
from .arguments import HIDDEN_SIZES, STREAMS

__all__ = [
    "HIDDEN_SIZES",
    "STREAMS",
    "SUPPORTED_CAPABILITIES",
    "VARIANT",
    "is_supported",
    "mhc_finish",
    "mhc_head",
    "mhc_prepare",
    "prebuild",
]

# The architectures this package ships for (FlashInfer's packages also export sm_107a at hidden 4096).
SUPPORTED_CAPABILITIES = ((10, 0), (10, 3))
# FlashInfer's defaults of the epsilons and the Sinkhorn count (the epsilons are launch arguments of every program).
RMS_EPS = 1e-5
HC_EPS = 1e-6
SINKHORN_ITERS = 20
# The variant of a call that names none: ``fast`` unless ``configure`` set ``det``; read at call time.
VARIANT = "fast"

_FAST_PACKAGE = f"{__package__}._vendor.cake.cake_mhc_train_backend"
# (variant or "head", arch, hidden, sinkhorn_iters) already handed to cake_native.prebuild in this process
_PREBUILT: set[tuple[str, str, int, int]] = set()


@cache
def _det() -> ModuleType:
    from ._vendor import cake_mhc_det_train_backend as package
    from ._vendor.cake_mhc_det_train_backend import cake_build

    cake_native.install(cake_build)
    return package


@cache
def _fast() -> ModuleType:
    from ._vendor.cake import cake_mhc_train_backend as package
    from ._vendor.cake.cake_mhc_train_backend import cake_build

    cake_native.install(cake_build)
    return package


@cache
def _head() -> ModuleType:
    from ._vendor import cake_mhc_det_train_head_backend as package
    from ._vendor.cake_mhc_det_train_head_backend import cake_build

    cake_native.install(cake_build)
    return package


def _device(device: Any) -> torch.device:
    return torch.device("cuda", device) if isinstance(device, int) else torch.device(device)


def _resolve(variant: str | None) -> str:
    """The package a call runs: ``variant`` (``None``: :data:`VARIANT`), or ``det`` for a ``fast`` call while torch's
    deterministic algorithms are enabled."""
    variant = VARIANT if variant is None else variant
    if variant not in _variant.VARIANTS:
        raise ValueError(f"variant must be one of {_variant.VARIANTS}, not {variant!r}")
    if variant == "fast" and torch.are_deterministic_algorithms_enabled():
        _variant.warn_det_fallback("mhc", "calls under torch.use_deterministic_algorithms(True)")
        return "det"
    return variant


def is_supported(device: Any, hidden: int | None = None) -> bool:
    """Whether ``device`` (and ``hidden``, when given) runs every entry point of both variants and the head."""
    device = _device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    if tuple(torch.cuda.get_device_capability(device)) not in SUPPORTED_CAPABILITIES:
        return False
    if hidden is not None and int(hidden) not in HIDDEN_SIZES:
        return False
    return (
        _det().is_cake_mhc_det_train_supported(device, hidden)
        and _fast().is_cake_mhc_train_supported(device, hidden)
        and _head().is_cake_mhc_det_train_head_supported(device, hidden)
    )


def mhc_prepare(
    X: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    rms_eps: float = RMS_EPS,
    hc_eps: float = HC_EPS,
    sinkhorn_iters: int = SINKHORN_ITERS,
    variant: str | None = None,
) -> tuple[torch.Tensor, Any]:
    """Autograd-enabled mapping, coefficients and collapse of one connection: ``(U, handle)``.

    ``U`` is BF16 ``[T, H]``; hand ``handle`` to one :func:`mhc_finish` with the same ``X``. ``X``, ``fn``,
    ``scale`` and ``base`` may require gradients (``rms_eps`` / ``hc_eps`` / ``sinkhorn_iters`` as in the module
    docstring; ``sinkhorn_iters`` other than 20 runs the run-time-count programs). ``variant`` (``det`` / ``fast``;
    default :data:`VARIANT`) picks the backward of this connection.
    """
    if _resolve(variant) == "fast":
        return _fast().cake_mhc_train_prepare(
            X,
            fn,
            scale,
            base,
            rms_eps=rms_eps,
            hc_eps=hc_eps,
            sinkhorn_iters=sinkhorn_iters,
            deterministic=False,
        )
    return _det().cake_mhc_det_train_prepare(
        X, fn, scale, base, rms_eps=rms_eps, hc_eps=hc_eps, sinkhorn_iters=sinkhorn_iters
    )


def mhc_finish(X: torch.Tensor, B: torch.Tensor, handle: Any) -> torch.Tensor:
    """Autograd-enabled residual mixing: BF16 ``Y`` ``[T, 4, H]`` from ``X``, the branch output ``B`` ``[T, H]`` and
    the ``handle`` of :func:`mhc_prepare`, in the package that made the handle.

    The backward returns ``dB`` and no gradient for ``X``: the residual-mixing contribution is part of the single-cast
    ``dX`` of the prepare node. A reentrant checkpoint region holds both calls or starts after the prepare.
    """
    fast = sys.modules.get(_FAST_PACKAGE)
    if fast is not None and isinstance(handle, fast._MHCTrainHandle):
        return _fast().cake_mhc_train_finish(X, B, handle, deterministic=False)
    det = _det()
    if isinstance(handle, det._MHCTrainHandle):
        return det.cake_mhc_det_train_finish(X, B, handle)
    raise TypeError(f"handle must be the handle mhc_prepare returned, not {type(handle).__name__}")


def mhc_head(X: torch.Tensor) -> torch.Tensor:
    """Autograd-enabled head: BF16 ``Y = X.mean(dim=1)`` ``[T, H]`` of the four streams ``X`` ``[T, 4, H]``, and its
    gradient, bit for bit torch's."""
    return _head().cake_mhc_det_train_head(X)


def _route_libraries(jit: ModuleType, arch: str, hidden: int, sinkhorn_iters: int) -> tuple[str, ...]:
    """The ``build.json`` libraries every route of ``hidden`` on ``arch`` may launch, in route order.

    The run-time Sinkhorn count programs serve counts other than the compiled one only; the dual-layout fn split of the
    checkpoint carry serves ``save_stats=False`` only, which :func:`mhc_prepare` does not use.
    """
    names = []
    for kind in jit.KINDS:
        for record in jit.route_modules(arch, hidden, kind):
            rule = jit.MODULES[record]["select_rule"] or {}
            if rule.get("kind") == "run_time_iters" and sinkhorn_iters == jit.SINKHORN_ITERS:
                continue
            names.append(jit.library_name(record, arch))
    return tuple(dict.fromkeys(names))


def prebuild(
    device: Any,
    hidden: int,
    *,
    variant: str | None = None,
    sinkhorn_iters: int = SINKHORN_ITERS,
    head: bool = False,
    max_workers: int | None = None,
) -> tuple[str, ...]:
    """Compile every library :func:`mhc_prepare` / :func:`mhc_finish` (and :func:`mhc_head` with ``head``) may launch
    at ``hidden`` on ``device`` with ``variant`` (default :data:`VARIANT`), ``max_workers`` nvcc processes at a time.

    Returns the library names. Libraries already in the cache are not rebuilt, and a repeated call in a process
    compiles nothing; ranks starting at once build each library once (``cake_native``'s file lock).
    """
    device = _device(device)
    capability = tuple(torch.cuda.get_device_capability(device))
    if capability not in SUPPORTED_CAPABILITIES:
        raise ValueError(f"the Cake mHC kernels run compute capability {SUPPORTED_CAPABILITIES}, not {capability}")
    variant = _resolve(variant)
    package = _fast() if variant == "fast" else _det()
    arch = package.cake_jit.arch_for_capability(capability)
    names = _route_libraries(package.cake_jit, arch, int(hidden), int(sinkhorn_iters))
    head_names: tuple[str, ...] = ()
    if head:
        head_jit = _head().cake_jit
        if int(hidden) not in head_jit.supported_hidden_sizes(arch):
            raise ValueError(
                f"the Cake mHC head runs the hidden sizes {head_jit.supported_hidden_sizes(arch)} on {arch}"
            )
        head_names = tuple(head_jit.library_name(head_jit.ROLES[role], arch) for role in ("head_fwd", "head_bwd"))
    parts = [((variant, arch, int(hidden), int(sinkhorn_iters)), package.cake_build, names)]
    if head:
        parts.append((("head", arch, int(hidden), 0), _head().cake_build, head_names))
    for key, cake_build, libraries in parts:
        if key not in _PREBUILT:
            cake_native.prebuild(cake_build, libraries, max_workers)
            _PREBUILT.add(key)
    return names + head_names
