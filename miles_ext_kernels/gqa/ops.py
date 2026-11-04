"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Packed causal GQA attention, head dimension 256, forward and backward on the vendored Cake kernels (SM100a / SM103a).

``q [T_q, H_q, 256]``, ``k`` / ``v [T_k, H_kv, 256]`` (BF16, ``H_q / H_kv`` even), segments packed by int32 device
``cu_seqlens_q`` / ``cu_seqlens_k``; query row ``i`` of a segment with ``q_len`` rows and ``k_len`` keys sees the keys
``j <= i + k_len - q_len`` (FlashAttention's bottom-right alignment). These are the semantics of FlashInfer's
``flashinfer.gqa_train.varlen_gqa_attention``: its backend package is vendored byte-identical under
``_vendor/cake_gqa_train`` and builds its libraries through :mod:`miles_ext_kernels.cake_native` (nvcc and tvm-ffi,
no FlashInfer). The kernels read no tensor on the host and never synchronize the device.

Numerics: FP32 accumulation throughout, P and dS rounded to BF16 only as MMA operands. The output, the LSE and the
BF16 dK / dV are bit-identical run to run. dQ is summed in FP32 with atomic adds, one partial per 128-key block, in a
scheduling order, and rounded to BF16 once: two runs differ by the FP32 reordering error before that rounding.
"""

from __future__ import annotations

import functools
from pathlib import Path
from types import ModuleType

import torch

from miles_ext_kernels import cake_native

__all__ = [
    "HEAD_DIM",
    "STAGES",
    "SUPPORTED_CAPABILITIES",
    "backend",
    "device_arch",
    "libraries",
    "prebuild",
    "varlen_gqa_attention",
]

HEAD_DIM = 256
SUPPORTED_CAPABILITIES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}
# The stages one forward and backward of :func:`varlen_gqa_attention` launch (BF16 dK / dV), in launch order.
STAGES = ("fwd", "bwd_pre", "bwd_bf16", "bwd_post")


@functools.cache
def backend() -> ModuleType:
    """The vendored ``cake_backend`` module, building and loading its libraries through ``cake_native``."""
    from ._vendor.cake_gqa_train import cake_backend, cake_build

    cake_native.install(cake_build)
    return cake_backend


def device_arch(device=None) -> str:
    """The architecture tag of a CUDA device (default: the current one); ``RuntimeError`` on another GPU."""
    capability = tuple(torch.cuda.get_device_capability(device))
    try:
        return SUPPORTED_CAPABILITIES[capability]
    except KeyError:
        raise RuntimeError(
            "the Cake GQA training attention kernels run on SM100a or SM103a (B200 / GB200, B300 / GB300), "
            f"not on capability {capability}"
        ) from None


_device_arch = functools.cache(device_arch)  # per device index, on the call path


def libraries(arch: str) -> list[str]:
    """The ``build.json`` libraries of :data:`STAGES` on ``arch``."""
    cake_backend = backend()
    return [cake_backend.library_name(cake_backend.stage_record(stage, arch), arch) for stage in STAGES]


@functools.cache
def _prebuild(arch: str) -> tuple[Path, ...]:
    return tuple(cake_native.prebuild(backend().cake_build, libraries(arch)))


def prebuild(device=None) -> tuple[Path, ...]:
    """Build the libraries of :data:`STAGES` for the architecture of ``device`` (default: the current one) before the
    first call, in parallel; once per architecture and process. Returns their paths in the cache."""
    return _prebuild(device_arch(device))


def varlen_gqa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    max_seqlen_k: int | None = None,
) -> torch.Tensor:
    """Packed causal GQA attention with autograd: BF16 ``out [T_q, H_q, 256]``; ``q``, ``k`` and ``v`` get BF16
    gradients.

    ``cu_seqlens_q`` / ``cu_seqlens_k``: int32 ``[B + 1]`` on the device, ``cu[0] = 0``, non-decreasing,
    ``cu_q[B] = T_q``, ``cu_k[B] = T_k``. ``softmax_scale``: default ``256 ** -0.5``. ``max_seqlen_k``: a host int at
    least every segment's key count (as FlashAttention's), which sizes the backward grid; ``None`` bounds it by
    ``T_k``. Operands the kernels do not take (dtype, head dim, odd group, strides, ``cu_seqlens``) raise
    ``ValueError``, another GPU ``RuntimeError``.
    """
    if q.is_cuda:
        _device_arch(q.device.index)
    return backend().varlen_gqa_attention(
        q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale=softmax_scale, max_seqlen_k=max_seqlen_k
    )
