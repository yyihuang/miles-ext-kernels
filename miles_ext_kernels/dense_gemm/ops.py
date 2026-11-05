"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

The dense projection GEMMs miles calls (SM100a / SM103a / SM107a), on the vendored FlashInfer package
``_vendor/dense_projection_det_gemm``, whose libraries :mod:`miles_ext_kernels.cake_native` builds and loads (SM107a
with a CUDA 13.4 or newer toolkit).

K1 (``dense_projection_det_gemm``) computes ``out[l, m, n] = sum_k A[l, m, k] B[l, k, n]`` over strided BF16 views with
FP32 accumulation and one rounding at the store. Its launch plan, split-K and stream-K fixup order are a pure function
of the views' geometry and the device, so every call is bit-reproducible. It reads a view in place when the view starts
at a 16-byte aligned address, has unit stride on one matrix axis and its other strides in whole 16-byte units, and it
needs the output columns ``N`` to be a multiple of 8. :func:`batched_projection` hands it the token-major ``[T, H, D]``
activations and the per-head weight views as they are, and copies a view only when it breaks that contract (a weight
whose storage is not 16-byte aligned in its parameter buffer, a gradient that is not contiguous).

K2 (``router_fp32_det_gemm``) is an FP32 GEMM through the exact three-way BF16 split of both operands: the products
of the splits accumulate with chunked FP32 promotion and the split-K partials are summed in a fixed order, so it is
bit-reproducible with FP32-class error (a relative Frobenius error of about 2e-7 against FP64).

The vendored entry points plan once per (device, stream, view geometry) and keep a bounded cache of planned launches;
a new token count plans a new launch on its first call (tenths of a millisecond on the host) and compiles its program
if no earlier shape used it. :func:`prebuild_batched_projection` and :func:`prebuild_router` compile the programs of
given token counts ahead. The cached launches of a stream share a growable workspace, so do not capture these calls
in a CUDA graph.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable
from pathlib import Path

import torch

from miles_ext_kernels import cake_native

__all__ = [
    "SUPPORTED_CAPABILITIES",
    "WEIGHT_LAYOUTS",
    "batched_projection",
    "check_projection_dims",
    "prebuild_batched_projection",
    "prebuild_router",
    "router_gating",
]

SUPPORTED_CAPABILITIES = dict(cake_native.ARCH_BY_CAPABILITY)
# The orientation of the per-head weights of batched_projection: [H, D_in, D_out] or [H, D_out, D_in].
WEIGHT_LAYOUTS = ("in_out", "out_in")


@functools.cache
def _vendored():
    """``(cake_backend, cake_jit, cake_build)`` of the vendored package, building its libraries through cake_native."""
    from ._vendor.dense_projection_det_gemm import cake_build

    cake_native.install(cake_build)
    from ._vendor.dense_projection_det_gemm import cake_backend, cake_jit

    return cake_backend, cake_jit, cake_build


def _device_index(device) -> int:
    device = torch.device("cuda") if device is None else torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"the Cake dense projection GEMMs run on CUDA devices, got {device}")
    return torch.cuda.current_device() if device.index is None else device.index


@functools.cache
def _arch_of(index: int) -> str:
    """The architecture of device ``index``, checked to be supported and its programs to build with the toolkit."""
    major, minor = torch.cuda.get_device_capability(index)
    arch = SUPPORTED_CAPABILITIES.get((major, minor))
    if arch is None:
        raise RuntimeError(
            f"the Cake dense projection GEMMs run on SM100a / SM103a / SM107a (B200, B300, GB300, R200); cuda:{index} "
            f"has compute capability {major}.{minor}"
        )
    cake_native.check_toolkit(arch, "the Cake dense projection GEMMs")
    return arch


def _arch(device=None) -> str:
    return _arch_of(_device_index(device))


def check_projection_dims(d_in: int, d_out: int) -> None:
    """Raise ``ValueError`` unless the K1 programs serve a per-head projection ``d_in -> d_out``: each is the output
    width of one of its GEMMs (the forward, the input gradient), so both must be positive multiples of 8."""
    for name, dim in (("input", d_in), ("output", d_out)):
        if dim < 8 or dim % 8:
            raise ValueError(
                f"the Cake batched projection needs per-head {name} features that are a positive multiple of 8, "
                f"got {dim}"
            )


def _k1_view(t: torch.Tensor) -> torch.Tensor:
    """``t`` when the K1 programs read it in place (unit stride on its last axis, its other strides in whole 16-byte
    units, a 16-byte aligned start), else a contiguous copy."""
    if t.stride(-1) == 1 and all(stride % 8 == 0 for stride in t.stride()[:-1]) and t.data_ptr() % 16 == 0:
        return t
    return t.clone(memory_format=torch.contiguous_format)


def _k2_view(t: torch.Tensor) -> torch.Tensor:
    """The FP32 matrix ``t`` when the K2 programs read it in place (unit inner stride, 16-byte rows and start), else a
    contiguous copy."""
    if t.stride(1) == 1 and t.stride(0) % 4 == 0 and t.data_ptr() % 16 == 0:
        return t
    return t.clone(memory_format=torch.contiguous_format)


# The three GEMMs of a projection, as K1 calls ``gemm(A, B, out)`` over [H, ...] views of the token-major tensors
# ``x [T, H, D_in]``, ``g`` / ``out [T, H, D_out]`` and of ``weight`` ([H, D_in, D_out] in_out, [H, D_out, D_in]
# out_in). The weight gradient is computed in the weight's own orientation.
def _forward(gemm: Callable, x, weight, out, in_out: bool) -> None:
    gemm(x.transpose(0, 1), weight if in_out else weight.transpose(1, 2), out.transpose(0, 1))


def _dgrad(gemm: Callable, g, weight, dx, in_out: bool) -> None:
    gemm(g.transpose(0, 1), weight.transpose(1, 2) if in_out else weight, dx.transpose(0, 1))


def _wgrad(gemm: Callable, x, g, dw, in_out: bool) -> None:
    if in_out:
        gemm(x.permute(1, 2, 0), g.transpose(0, 1), dw)
    else:
        gemm(g.permute(1, 2, 0), x.transpose(0, 1), dw)


class _BatchedProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, in_out: bool) -> torch.Tensor:
        backend, _, _ = _vendored()
        x, weight = _k1_view(x), _k1_view(weight)
        d_out = weight.shape[2] if in_out else weight.shape[1]
        out = torch.empty(x.shape[0], x.shape[1], d_out, dtype=torch.bfloat16, device=x.device)
        _forward(backend.dense_projection_det_gemm, x, weight, out, in_out)
        ctx.save_for_backward(x, weight)
        ctx.in_out = in_out
        return out

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        x, weight = ctx.saved_tensors
        # The autograd engine may run this on a thread that has not used the CUDA runtime yet; the programs encode
        # their tensor maps through the driver API, which needs the device's primary context current on this thread.
        torch.cuda.current_stream(x.device).query()
        backend, _, _ = _vendored()
        g = _k1_view(grad)
        dx = dw = None
        if ctx.needs_input_grad[0]:
            dx = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
            _dgrad(backend.dense_projection_det_gemm, g, weight, dx, ctx.in_out)
        if ctx.needs_input_grad[1]:
            dw = torch.empty(weight.shape, dtype=weight.dtype, device=weight.device)
            _wgrad(backend.dense_projection_det_gemm, x, g, dw, ctx.in_out)
        return dx, dw, None


def batched_projection(x: torch.Tensor, weight: torch.Tensor, *, weight_layout: str) -> torch.Tensor:
    """The per-head projection ``x [T, H, D_in] -> [T, H, D_out]`` in BF16, differentiable in ``x`` and ``weight``.

    ``weight_layout`` is the orientation of the per-head weights: ``"in_out"`` is ``weight [H, D_in, D_out]``, the
    product ``torch.einsum("thd,hdm->thm", x, weight)`` (the MLA q-absorb projection); ``"out_in"`` is
    ``weight [H, D_out, D_in]``, ``torch.einsum("thm,hdm->thd", x, weight)`` (the value projection, DeepSeek-V4's
    output-group projection). Both may be views: ``x`` a slice of a wider head slot, ``weight`` a split of a
    parameter. The result is a new contiguous ``[T, H, D_out]`` tensor (the kernel writes its ``[H, T, D_out]`` view
    in place) and the gradient of ``weight`` keeps the weight's orientation.

    As cuBLAS computes the einsum, every element is an FP32 sum of exact BF16 products rounded once to BF16, but in a
    fixed order: the output and both gradients are bit-identical call to call (and not bitwise equal to cuBLAS).
    """
    if weight_layout not in WEIGHT_LAYOUTS:
        raise ValueError(f"weight_layout must be one of {WEIGHT_LAYOUTS}, got {weight_layout!r}")
    if x.dim() != 3 or weight.dim() != 3:
        raise ValueError(
            f"batched_projection takes x [T, H, D_in] and a 3-D weight, got {tuple(x.shape)} and {tuple(weight.shape)}"
        )
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError(f"batched_projection takes BF16 x and weight, got {x.dtype} and {weight.dtype}")
    in_out = weight_layout == "in_out"
    heads, d_in = weight.shape[0], weight.shape[1 if in_out else 2]
    d_out = weight.shape[2 if in_out else 1]
    if (x.shape[1], x.shape[2]) != (heads, d_in):
        raise ValueError(
            f"x [T, H, D_in] = {tuple(x.shape)} does not match the {weight_layout} weight {tuple(weight.shape)} "
            f"(H = {heads}, D_in = {d_in})"
        )
    check_projection_dims(d_in, d_out)
    if weight.device != x.device:
        raise ValueError(f"x and weight must be on one device, got {x.device} and {weight.device}")
    _arch(x.device)
    return _BatchedProjection.apply(x, weight, in_out)


def _router_weight_operand(weight: torch.Tensor) -> torch.Tensor:
    """The router weight ``[E, H]`` as the K1 ``B`` operand of the logits ``x @ weight.T``: ``weight.T`` stored
    ``[H, E]`` with unit stride on ``E`` (N-major). K1 serves an FP32 output with a K-major ``A`` only for an N-major
    ``B`` (its ``kn`` programs; it registers no ``kk`` program with FP32 output), so the forward copies the weight
    (``E * H`` BF16 elements, small next to the activations) into that layout."""
    return weight.t().contiguous()


class _RouterGating(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        backend, _, _ = _vendored()
        x2 = _k1_view(x.reshape(-1, x.shape[-1]))
        logits = torch.empty(x2.shape[0], weight.shape[0], dtype=torch.float32, device=x.device)
        backend.dense_projection_det_gemm(x2, _router_weight_operand(weight), logits)
        ctx.save_for_backward(x2, weight)
        ctx.x_shape = x.shape
        return logits.view(*x.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        x2, weight = ctx.saved_tensors
        torch.cuda.current_stream(x2.device).query()  # see _BatchedProjection.backward
        backend, _, _ = _vendored()
        g = _k2_view(grad.reshape(-1, grad.shape[-1]).float())
        dx = dw = None
        if ctx.needs_input_grad[0]:
            dx = backend.router_det_dgrad(g, weight.float()).to(x2.dtype).view(ctx.x_shape)
        if ctx.needs_input_grad[1]:
            dw = backend.router_det_wgrad(g, x2.float()).to(weight.dtype)
        return dx, dw


def router_gating(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """FP32 router logits ``x @ weight.T`` for BF16 ``x [..., H]`` and ``weight [E, H]``: ``[..., E]`` FP32.

    The computation of Megatron's ``RouterGatingLinearFunction`` with ``--moe-router-dtype fp32`` (no bias): the
    forward multiplies the BF16 operands with FP32 accumulation and output (K1), the backward computes
    ``dx = g @ weight.float()`` and ``dweight = g.T @ x.float()`` in FP32 (K2) and rounds them to the dtypes of ``x``
    and ``weight``. Bit-reproducible call to call; ``E`` and ``H`` must be multiples of 8.
    """
    if weight.dim() != 2 or x.shape[-1] != weight.shape[1]:
        raise ValueError(
            f"router_gating takes x [..., H] and weight [E, H], got {tuple(x.shape)} and {tuple(weight.shape)}"
        )
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError(f"router_gating takes BF16 x and weight, got {x.dtype} and {weight.dtype}")
    experts, hidden = weight.shape
    if experts % 8 or hidden % 8:
        raise ValueError(
            f"router_gating needs the expert count and the hidden size to be multiples of 8, got {experts} and "
            f"{hidden}"
        )
    if weight.device != x.device:
        raise ValueError(f"x and weight must be on one device, got {x.device} and {weight.device}")
    _arch(x.device)
    return _RouterGating.apply(x, weight)


def _plan_recorder(modules: set[str], index: int) -> Callable:
    """A ``gemm(A, B, out)`` that routes a call of the K1 entry point ``dense_projection_det_gemm`` (on ``meta`` views)
    for device ``index`` as it does and records the module it would launch."""
    backend, cake_jit, _ = _vendored()
    arch, facts = _arch_of(index), backend.device_facts(index)

    def plan(a, b, out):
        # a batched BF16 GEMM of few rows over K-major operands runs the swapped program, when one fits its size
        if out.dtype == torch.bfloat16 and a.dim() == 3 and a.stride(2) == 1 and b.stride(1) == 1:
            batches, rows, depth = a.shape
            cols = b.shape[2]
            if backend.swap_small_m(batches, rows, cols, False) and depth % backend.BLOCK_K == 0:
                template = backend.swap_ab_template(batches, rows, cols, facts.sm_count, arch)
                if template is not None:
                    modules.add(cake_jit.select_module(arch, template))
                    return
        planned, *_ = backend.plan_dense_projection_det_gemm(
            a, b, out, sm_count=facts.sm_count, l2_bytes=facts.l2_bytes, arch=arch
        )
        modules.add(cake_jit.select_module(arch, planned.template))

    return plan


def _prebuild(modules: set[str], arch: str, max_workers: int | None) -> list[Path]:
    _, cake_jit, cake_build = _vendored()
    names = sorted(cake_jit.library_name(module, arch) for module in modules)
    return cake_native.prebuild(cake_build, names, max_workers)


def _meta(*shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return torch.empty(*shape, dtype=dtype, device="meta")


def prebuild_batched_projection(
    num_heads: int,
    d_in: int,
    d_out: int,
    *,
    weight_layout: str,
    token_counts: Iterable[int],
    device=None,
    max_workers: int | None = None,
) -> list[Path]:
    """Compile, ahead of their first call, the programs :func:`batched_projection` runs on ``device`` (default: the
    current device) for ``num_heads`` heads of ``d_in -> d_out`` at each of ``token_counts``: its forward, input and
    weight gradients. The plan, and so the program, can change with the token count; a count not listed compiles its
    program on its first call when no listed one planned it. Returns the cache paths of the libraries."""
    if weight_layout not in WEIGHT_LAYOUTS:
        raise ValueError(f"weight_layout must be one of {WEIGHT_LAYOUTS}, got {weight_layout!r}")
    index = _device_index(device)
    in_out = weight_layout == "in_out"
    modules: set[str] = set()
    plan = _plan_recorder(modules, index)
    weight = _meta(num_heads, d_in, d_out) if in_out else _meta(num_heads, d_out, d_in)
    for tokens in token_counts:
        x, g = _meta(tokens, num_heads, d_in), _meta(tokens, num_heads, d_out)
        _forward(plan, x, weight, g, in_out)
        _dgrad(plan, g, weight, x, in_out)
        _wgrad(plan, x, g, weight, in_out)
    return _prebuild(modules, _arch_of(index), max_workers)


def prebuild_router(
    num_experts: int,
    hidden_size: int,
    *,
    token_counts: Iterable[int],
    device=None,
    max_workers: int | None = None,
) -> list[Path]:
    """Compile, ahead of their first call, the programs :func:`router_gating` runs on ``device`` (default: the current
    device) for ``num_experts`` experts of ``hidden_size`` at each of ``token_counts``: the K1 forward (planned per
    token count), the K2 input and weight gradients and the BF16x3 split. Returns the cache paths of the libraries."""
    backend, cake_jit, _ = _vendored()
    index = _device_index(device)
    arch = _arch_of(index)
    modules: set[str] = set()
    plan = _plan_recorder(modules, index)
    weight = _meta(num_experts, hidden_size)
    fp32 = torch.float32
    for tokens in token_counts:
        plan(_meta(tokens, hidden_size), _meta(hidden_size, num_experts), _meta(tokens, num_experts, dtype=fp32))
        g, x = _meta(tokens, num_experts, dtype=fp32), _meta(tokens, hidden_size, dtype=fp32)
        dx, dw = _meta(tokens, hidden_size, dtype=fp32), _meta(num_experts, hidden_size, dtype=fp32)
        for a, b, out in ((g, weight.float(), dx), (g.t(), x, dw)):  # the K2 input and weight gradients
            planned, *_ = backend.plan_router_fp32_det_gemm(a, b, out)
            modules.add(cake_jit.select_module(arch, planned.template))
    split = backend.fused_split_module(arch)
    if split is not None:
        modules.add(split)
    return _prebuild(modules, arch, max_workers)
