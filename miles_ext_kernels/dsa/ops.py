"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

DSA (DeepSeek Sparse Attention) training operators: sparse attention and lightning indexer, forward and backward.

Public surface (all layouts of :mod:`.layouts`):

* :func:`sparse_attention` -- autograd sparse MQA/MLA attention over top-k indices with an optional FP32 attention
  sink, in the variant :data:`VARIANT` names (``--cake-dsa-variant``):

  - ``det``: the generated deterministic kernels of this package (``registry.json`` + ``csrc/``, loaded by
    :mod:`._jit`). Every gradient is bit-reproducible run to run: no floating-point atomics, and the key-side
    gradients are reduced from per-slot FP32 partials in an order fixed by the indices.
  - ``fast``: FlashInfer's Cake DSA training kernels (``_vendor/cake_dsa_train``, built by
    :mod:`miles_ext_kernels.cake_native`) for 64 local query heads, ``d_v = 512`` with a 64-channel RoPE tail or
    none, and no attention sink. ``dq`` is bit-reproducible; ``dkv`` is accumulated with FP32 ``red.global`` atomics,
    so its summation order changes from run to run. A call ``fast`` does not serve (:func:`fast_unserved_reason`)
    runs ``det`` and says so once per reason.

* :func:`indexer_logits`, :func:`select_topk`, :func:`indexer_topk_scores`, :func:`lightning_indexer` -- the
  deterministic indexer pipeline over a caller-supplied top-k callable (:func:`torch_topk` by default), with an
  explicit ``topk_indices`` override for rollout replay.
* :func:`sparse_attention_forward` / :func:`sparse_attention_backward` / :func:`indexer_backward` -- non-autograd
  ``det`` entry points for tests and benchmarks.
* :func:`check_device` and :func:`prebuild` -- the checks and the kernel warm-up of a layer's construction.

Parallelism is not handled here: the kernels see local heads and the local-or-gathered key rows.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import torch

from miles_ext_kernels import _variant

from . import _jit, _kernels, layouts, reduction
from ._jit import device_arch
from .layouts import RowMeta

FAMILY = "dsa"
# The sparse-attention variant of a call that names none (``--cake-dsa-variant``, resolved by ``validate_args`` and
# set by ``miles_ext_kernels.dsa.configure``; ``det`` until then).
VARIANT = "det"

ATTENTION_TOPK_MULTIPLE = 64
LOG2E = 1.4426950408889634
# Reducer of the lightning-indexer backward: 4 = ``dsa_segmented_reduce_split4`` (fixed 4-way tree per key,
# deterministic FP32); 1 = the serial per-key reducer ``dsa_segmented_reduce``.
INDEXER_REDUCE_SPLIT = 4
# Reducer of the sparse-attention backward: the same fixed 4-way tree (bandwidth-bound; the serial reducer's
# per-key latency chain is the tail of every chunk).  Fixed-order FP32.
ATTENTION_REDUCE_SPLIT = 4

# The calls the fast variant serves: 64 local query heads, d_v 512 and d_qk 576 (the RoPE program) or 512 (the NoPE
# program), no attention sink.
FAST_HEADS = 64
FAST_D_V = 512
FAST_D_ROPE = 64

TopkFn = Callable[[torch.Tensor, int], torch.Tensor]


# --------------------------------------------------------------------------- #
# Device checks and warm-up
# --------------------------------------------------------------------------- #


def check_device(device=None) -> str:
    """The architecture tag of ``device`` (``sm_100a`` / ``sm_103a`` / ``sm_107a``); raises ``NotImplementedError``
    elsewhere, and ``RuntimeError`` when the toolkit cannot build the device's kernels (SM107a needs CUDA 13.4)."""
    if not torch.cuda.is_available():
        raise NotImplementedError("the Cake DSA kernels need a CUDA device (SM100a, SM103a or SM107a)")
    arch = device_arch(device)
    _jit._check_toolkit(arch)
    return arch


def fast_unserved_reason(heads: int, d_qk: int, d_v: int, *, has_sink: bool, num_kv: int = 1) -> str | None:
    """Why the fast variant cannot run a call of this geometry (``None`` when it can); such a call runs ``det``."""
    if has_sink:
        return "calls with an attention sink"
    if heads != FAST_HEADS:
        return f"{heads} local query heads (only {FAST_HEADS})"
    if d_v != FAST_D_V or d_qk not in (FAST_D_V, FAST_D_V + FAST_D_ROPE):
        return f"d_qk {d_qk} with d_v {d_v} (only d_v {FAST_D_V} with d_qk {FAST_D_V + FAST_D_ROPE} or {FAST_D_V})"
    if num_kv == 0:
        return "an empty key set"
    return None


@functools.cache
def _fast_backend():
    """``(cake_build, cake_jit, cake_backend)`` of the vendored FlashInfer package, its loader on cake_native."""
    from miles_ext_kernels import cake_native

    from ._vendor.cake_dsa_train import cake_build

    cake_native.install(cake_build)
    from ._vendor.cake_dsa_train import cake_backend, cake_jit

    return cake_build, cake_jit, cake_backend


def _fast_program(d_qk: int) -> str:
    _, cake_jit, _ = _fast_backend()
    return cake_jit.PROGRAM if d_qk == FAST_D_V + FAST_D_ROPE else cake_jit.NOPE_PROGRAM


def _build_fast_program(program: str, arch: str) -> None:
    """Build every library of the FlashInfer ``program`` for ``arch``, in parallel (cached ones are skipped)."""
    from miles_ext_kernels import cake_native

    cake_build, cake_jit, _ = _fast_backend()
    names = [cake_jit.library_name(stage, arch, program) for stage in cake_jit.registered_stages(program)]
    cake_native.prebuild(cake_build, names)


_FAST_SEEN: set[tuple[str, str]] = set()


def _fast_first_call(program: str, arch: str) -> None:
    """On the first fast call of ``program`` on ``arch``, build all its libraries unless builds are lazy."""
    if (program, arch) in _FAST_SEEN:
        return
    _FAST_SEEN.add((program, arch))
    if not _kernels.LAZY_BUILD:
        _build_fast_program(program, arch)


def prebuild(
    device=None,
    *,
    heads: int | None = None,
    d_qk: int | None = None,
    d_v: int = 512,
    attn_sink: bool = False,
    variant: str | None = None,
) -> None:
    """Build the kernels a layer will run on ``device`` before its first call (the first call builds them otherwise).

    Every ``det`` stage of the device's architecture, plus -- under the ``fast`` variant (``variant``, default
    :data:`VARIANT`), when the layer's geometry (``heads`` local query heads, ``d_qk``, ``d_v``, ``attn_sink``) is one
    it serves -- the libraries of the FlashInfer program that geometry runs. Idempotent.
    """
    arch = check_device(device)
    _jit.prebuild(arch)
    _kernels._PREBUILT.add(arch)
    if _resolve_variant(variant) != "fast" or heads is None or d_qk is None:
        return
    if fast_unserved_reason(heads, d_qk, d_v, has_sink=attn_sink) is None:
        program = _fast_program(d_qk)
        _build_fast_program(program, arch)
        _FAST_SEEN.add((program, arch))


def _resolve_variant(variant: str | None) -> str:
    variant = VARIANT if variant is None else variant
    if variant not in _variant.VARIANTS:
        raise ValueError(f"variant must be one of {_variant.VARIANTS}, not {variant!r}")
    return variant


# --------------------------------------------------------------------------- #
# Top-k selection (the caller owns the policy: a replay-manager wrapper, a range-aware top-k, ...)
# --------------------------------------------------------------------------- #


def torch_topk(logits: torch.Tensor, topk: int) -> torch.Tensor:
    """``torch.topk`` over the last axis as int32 indices; ``-inf`` picks and the short-axis pad become ``-1``."""
    count = min(topk, logits.shape[-1])
    scores, indices = torch.topk(logits, count, dim=-1)
    indices = indices.to(torch.int32).masked_fill(scores == float("-inf"), -1)
    if count < topk:
        indices = torch.nn.functional.pad(indices, (0, topk - count), value=-1)
    return indices


def select_topk(logits: torch.Tensor, topk: int, topk_fn: TopkFn | None = None) -> torch.Tensor:
    """Select ``min(topk, num_keys)`` key indices per query row of ``logits[..., num_keys]`` with ``topk_fn``.

    Rows are flattened to ``[rows, num_keys]`` for ``topk_fn(logits, count)`` (:func:`torch_topk` by default) and
    the result is reshaped to ``[..., topk]``; short key axes are padded with ``-1``.
    """
    num_keys = logits.shape[-1]
    count = min(topk, num_keys)
    flat = logits.reshape(-1, num_keys)
    indices = (topk_fn or torch_topk)(flat, count).to(torch.int32)
    if count < topk:
        indices = torch.nn.functional.pad(indices, (0, topk - count), value=-1)
    return indices.reshape(*logits.shape[:-1], topk)


def extract_topk_scores(logits: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
    """Gather ``logits`` at ``topk_indices`` along the last axis; ``-1`` slots become ``-inf``."""
    valid = topk_indices >= 0
    safe = topk_indices.clamp(min=0).to(torch.int64)
    scores = torch.gather(logits, dim=-1, index=safe)
    return torch.where(valid, scores, torch.full_like(scores, float("-inf")))


# --------------------------------------------------------------------------- #
# Sparse attention: flat kernels
# --------------------------------------------------------------------------- #


def _sink_or_disabled(attn_sink: torch.Tensor | None, heads: int, device) -> torch.Tensor:
    if attn_sink is None:
        return torch.full((heads,), float("-inf"), dtype=torch.float32, device=device)
    if attn_sink.shape != (heads,):
        raise ValueError(f"attn_sink must be [heads={heads}], got {tuple(attn_sink.shape)}")
    return attn_sink.to(torch.float32).contiguous()


def _flat_attention_forward(q, kv, indices, sink, sm_scale, d_v, meta: RowMeta):
    """Run the flat forward; return ``(o_flat, lse_flat, o_layout)``.

    ``o_layout`` is the output in the model layout and is the tensor the wrapper hands back.  For batched
    layouts it is allocated here in layout shape and the kernel writes through a flat view of it, so the
    returned tensor is a fresh allocation rather than a view of an intermediate (callers such as the
    DeepSeek-V4 attention apply the inverse RoPE in place on the output, which autograd rejects for views
    produced inside a custom Function).
    """
    out = None
    o_layout = None
    if meta.layout != "thd":
        o_layout = torch.empty(meta.batch, meta.seq_len, q.shape[1], d_v, dtype=q.dtype, device=q.device)
        out = o_layout.view(q.shape[0], q.shape[1], d_v)
    o_flat, lse_flat = _kernels.dsa_attention_forward(q, kv, indices, sink, float(sm_scale), d_v=d_v, out=out)
    return o_flat, lse_flat, (o_layout if o_layout is not None else o_flat)


def _flat_attention_backward(q, kv, o, do, indices, lse, sink, sm_scale, d_v, workspace_bytes):
    num_rows, heads, d_qk = q.shape
    num_kv = kv.shape[0]
    topk = indices.shape[1]
    do = do.contiguous()
    geo = _kernels.attention_geometry(q, kv, indices, d_v)
    # delta = rowsum(O * dO) in FP32: one kernel, fixed order (bit-reproducible).
    delta = _kernels.dsa_attention_delta_launch(o.contiguous(), do)
    dq = torch.empty_like(q)
    dkv32 = torch.zeros(num_kv, d_qk, dtype=torch.float32, device=q.device)
    # FP32 partial rows per (query row, slot): one per head block, or one per row when the two head blocks of a row
    # run as a cluster and sum their contributions on chip (``geo.pair``)
    kv_partials = geo.kv_partials
    chunks = reduction.plan_row_chunks(num_rows, kv_partials * topk, d_qk, reduction.workspace_budget(workspace_bytes))
    rows_per_chunk = chunks[0][1] - chunks[0][0]
    partial = torch.empty(rows_per_chunk * kv_partials * topk, d_qk, dtype=torch.float32, device=q.device)
    # one global stable sort of (key, chunk) composites: per-chunk reducer segments without per-chunk sorts
    segments = reduction.ChunkedSegments(indices, kv_partials, rows_per_chunk, num_kv)
    for chunk, (start, end) in enumerate(chunks):
        rows = end - start
        part = partial[: rows * kv_partials * topk]
        _kernels.dsa_attention_backward_rows(
            q[start:end],
            kv,
            do[start:end],
            indices[start:end],
            lse[start:end],
            delta[start:end],
            dq[start:end],
            part,
            float(sm_scale),
            geo,
        )
        order_local, seg_start, seg_count = segments.segment(chunk)
        _kernels.dsa_segmented_reduce_launch(
            part, order_local, seg_start, seg_count, dkv32, split=ATTENTION_REDUCE_SPLIT
        )
    dkv = dkv32.to(kv.dtype)
    dsink = None
    if sink is not None:
        # d sink[h] = -sum_rows delta[row, h] * p_sink[row, h], p_sink = exp2(sink * log2 e - lse).
        dsink = -(delta * torch.exp2(sink.unsqueeze(0) * LOG2E - lse)).sum(dim=0)
    return dq, dkv, dsink


# --------------------------------------------------------------------------- #
# Sparse attention: public functional + autograd
# --------------------------------------------------------------------------- #


def _prepare_attention(q, kv, indices, layout):
    q_flat, kv_flat, idx_flat, meta = layouts.flatten_attention_inputs(q, kv, indices, layout)
    idx_flat = layouts.pad_topk(idx_flat, ATTENTION_TOPK_MULTIPLE)
    if q_flat.dtype != torch.bfloat16 or kv_flat.dtype != torch.bfloat16:
        raise TypeError("sparse attention expects BF16 q and kv")
    return q_flat, kv_flat, idx_flat, meta


def sparse_attention_forward(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    *,
    sm_scale: float | None = None,
    attn_sink: torch.Tensor | None = None,
    layout: str = "thd",
    d_v: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward only: returns ``(output, lse)`` in the input layout (``lse`` is base-2, sink included)."""
    q_flat, kv_flat, idx_flat, meta = _prepare_attention(q, kv, indices, layout)
    if sm_scale is None:
        sm_scale = q_flat.shape[-1] ** -0.5
    sink = _sink_or_disabled(attn_sink, q_flat.shape[1], q_flat.device)
    _o_flat, lse_flat, o_layout = _flat_attention_forward(q_flat, kv_flat, idx_flat, sink, sm_scale, d_v, meta)
    return o_layout, meta.rows_to_layout(lse_flat)


def sparse_attention_backward(
    q: torch.Tensor,
    kv: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    indices: torch.Tensor,
    lse: torch.Tensor,
    *,
    sm_scale: float | None = None,
    attn_sink: torch.Tensor | None = None,
    layout: str = "thd",
    d_v: int = 512,
    workspace_bytes: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Backward only: returns ``(dq, dkv, d_attn_sink)`` in the input layout (``d_attn_sink`` is ``None`` without a sink)."""
    q_flat, kv_flat, idx_flat, meta = _prepare_attention(q, kv, indices, layout)
    if sm_scale is None:
        sm_scale = q_flat.shape[-1] ** -0.5
    sink = None if attn_sink is None else _sink_or_disabled(attn_sink, q_flat.shape[1], q_flat.device)
    o_flat, _ = layouts.flatten_rows(o, layout, kind="q")
    do_flat, _ = layouts.flatten_rows(do, layout, kind="q")
    lse_flat, _ = layouts.flatten_rows(lse, layout, kind="q")
    dq, dkv, dsink = _flat_attention_backward(
        q_flat, kv_flat, o_flat, do_flat, idx_flat, lse_flat.float(), sink, sm_scale, d_v, workspace_bytes
    )
    dkv = meta.kv_rows_to_layout(dkv)
    if layout == "thd" and kv.dim() == 3:
        dkv = dkv.unsqueeze(1)
    return meta.rows_to_layout(dq), dkv, dsink


class SparseAttentionFunction(torch.autograd.Function):
    """Autograd wrapper around the unified sparse attention kernels."""

    @staticmethod
    def forward(ctx, q, kv, attn_sink, indices, sm_scale, layout, d_v, workspace_bytes):
        q_flat, kv_flat, idx_flat, meta = _prepare_attention(q, kv, indices, layout)
        if sm_scale is None:
            sm_scale = q_flat.shape[-1] ** -0.5
        has_sink = attn_sink is not None
        sink = _sink_or_disabled(attn_sink, q_flat.shape[1], q_flat.device)
        o_flat, lse_flat, o_layout = _flat_attention_forward(q_flat, kv_flat, idx_flat, sink, sm_scale, d_v, meta)
        if o_layout is not o_flat:
            # o_flat is a view of the returned layout tensor, created here in no-grad mode. DeepSeek-V4
            # applies the inverse RoPE in place on the output before the projection, so save a copy that
            # the in-place update cannot reach (the pinned TileLang function saves o.clone() for the same reason).
            o_flat = o_flat.clone()
        ctx.save_for_backward(q_flat, kv_flat, idx_flat, o_flat, lse_flat, sink)
        ctx.meta = meta
        ctx.sm_scale = float(sm_scale)
        ctx.d_v = int(d_v)
        ctx.workspace_bytes = workspace_bytes
        ctx.has_sink = has_sink
        ctx.kv_had_group_dim = layout == "thd" and kv.dim() == 3
        return o_layout

    @staticmethod
    def backward(ctx, grad_output):
        q_flat, kv_flat, idx_flat, o_flat, lse_flat, sink = ctx.saved_tensors
        meta: RowMeta = ctx.meta
        do_flat, _ = layouts.flatten_rows(grad_output, meta.layout, kind="q")
        dq, dkv, dsink = _flat_attention_backward(
            q_flat,
            kv_flat,
            o_flat,
            do_flat,
            idx_flat,
            lse_flat,
            sink if ctx.has_sink else None,
            ctx.sm_scale,
            ctx.d_v,
            ctx.workspace_bytes,
        )
        dkv = meta.kv_rows_to_layout(dkv)
        if ctx.kv_had_group_dim:
            dkv = dkv.unsqueeze(1)
        return meta.rows_to_layout(dq), dkv, dsink, None, None, None, None, None


def _aligned(tensor: torch.Tensor) -> torch.Tensor:
    """``tensor`` when its base is 16-byte aligned (the TMA descriptors of the fast kernels need it), else a copy."""
    return tensor if tensor.data_ptr() % 16 == 0 else tensor.clone()


def _fast_sparse_attention(q, kv, indices, sm_scale: float, layout: str) -> torch.Tensor:
    """The fast variant on the flat contract: the FlashInfer RoPE program for ``d_qk = 576`` (the latent and the RoPE
    tail as strided views of the packed rows, no copy), the NoPE program for ``d_qk = 512``."""
    q_flat, kv_flat, idx_flat, meta = layouts.flatten_attention_inputs(q, kv, indices, layout)
    q_flat, kv_flat = _aligned(q_flat), _aligned(kv_flat)
    _, cake_jit, backend = _fast_backend()
    program = _fast_program(q_flat.shape[-1])
    _fast_first_call(program, device_arch(q_flat.device))
    if program == cake_jit.PROGRAM:
        out = backend.dsa_sparse_attention(
            q_flat[..., :FAST_D_V],
            q_flat[..., FAST_D_V:],
            kv_flat[:, :FAST_D_V],
            kv_flat[:, FAST_D_V:],
            idx_flat,
            softmax_scale=sm_scale,
        )
    else:
        out = backend.dsa_sparse_attention_nope(q_flat, kv_flat, idx_flat, softmax_scale=sm_scale)
    return meta.rows_to_layout(out)


def sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    *,
    sm_scale: float | None = None,
    attn_sink: torch.Tensor | None = None,
    layout: str = "thd",
    d_v: int = 512,
    workspace_bytes: int | None = None,
    variant: str | None = None,
) -> torch.Tensor:
    """Sparse MQA/MLA attention over top-k key indices (autograd; gradients for ``q``, ``kv``, ``attn_sink``).

    Args:
        q: BF16 queries, ``[T, H, d_qk]`` (``thd``) or ``[B, S, H, d_qk]`` (``bshd``).
        kv: BF16 latent keys/values, ``[T_kv, d_qk]`` / ``[T_kv, 1, d_qk]`` (``thd``) or ``[B, S_kv, d_qk]``.
        indices: int32 key indices per query, ``-1`` for empty slots; ``[T, topk]`` /
            ``[T, 1, topk]`` (``thd``) or batch-local ``[B, S, topk]`` (``bshd``).
        sm_scale: softmax scale (default ``d_qk ** -0.5``).
        attn_sink: optional FP32 per-head sink logit ``[H]`` (DeepSeek-V4).
        d_v: number of leading latent channels that form the value (``512``).
        workspace_bytes: FP32 partial-buffer budget for the ``det`` backward's key-side
            partials (default ``--cake-dsa-bwd-workspace-bytes``, 2 GiB).
        variant: ``det`` or ``fast`` (default :data:`VARIANT`); a call ``fast`` does not
            serve (:func:`fast_unserved_reason`) runs ``det`` with a one-time warning per reason.

    Returns the BF16 attention output ``[..., H, d_v]`` in the input layout.
    """
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        raise TypeError("sparse attention expects BF16 q and kv")
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    if _resolve_variant(variant) == "fast":
        num_kv = kv.shape[0] if layout == "thd" else kv.shape[0] * kv.shape[1]
        reason = fast_unserved_reason(
            int(q.shape[-2]), int(q.shape[-1]), int(d_v), has_sink=attn_sink is not None, num_kv=int(num_kv)
        )
        if reason is None:
            return _fast_sparse_attention(q, kv, indices, float(sm_scale), layout)
        _variant.warn_det_fallback(FAMILY, reason)
    return SparseAttentionFunction.apply(q, kv, attn_sink, indices, sm_scale, layout, d_v, workspace_bytes)


# --------------------------------------------------------------------------- #
# Lightning indexer
# --------------------------------------------------------------------------- #


def indexer_logits(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    *,
    layout: str = "thd",
) -> torch.Tensor:
    """Head-summed ReLU indexer logits (no autograd; the indexer gradient flows through :func:`indexer_topk_scores`).

    ``index_q`` BF16 ``[T, H, D]`` / ``[B, S, H, D]`` / ``[S, B, H, D]``, ``index_k`` BF16
    ``[T_kv, D]`` / ``[B, S_kv, D]`` / ``[S_kv, B, D]``, ``weights`` FP32 ``[T, H]`` (or
    ``[T, H, 1]``) / ``[B, S, H]`` / ``[S, B, H]``.  ``cu_seqlen_ks``/``cu_seqlen_ke``
    are int32 per-query key bounds ``[S]`` shared across the batch.  Returns FP32
    logits ``[T, T_kv]`` (``thd``) or ``[B, S, S_kv]``; keys outside a query's bounds
    are ``-inf``.
    """
    q_flat, k_flat, w_flat, meta = layouts.flatten_indexer_inputs(index_q, index_k, weights, layout)
    batch, seq_len, seq_len_kv = meta.batch, meta.seq_len, meta.seq_len_kv
    if cu_seqlen_ks.shape != (seq_len,) or cu_seqlen_ke.shape != (seq_len,):
        raise ValueError(
            f"cu_seqlen_ks/ke must be [{seq_len}], got {tuple(cu_seqlen_ks.shape)} / {tuple(cu_seqlen_ke.shape)}"
        )
    if q_flat.dtype != torch.bfloat16 or k_flat.dtype != torch.bfloat16:
        raise TypeError("indexer expects BF16 index_q and index_k")
    ks = cu_seqlen_ks.to(torch.int32).contiguous()
    ke = cu_seqlen_ke.to(torch.int32).contiguous()
    logits = _kernels.dsa_indexer_logits(
        q_flat, k_flat, w_flat, ks, ke, batch=batch, seq_len=seq_len, seq_len_kv=seq_len_kv
    )
    return logits.squeeze(0) if layout == "thd" else logits


def _flat_indexer_backward(q, k, w, indices, grad_scores, workspace_bytes):
    num_rows, heads, dim = q.shape
    num_kv = k.shape[0]
    topk = indices.shape[1]
    dq = torch.empty_like(q)
    dw = torch.empty(num_rows, heads, dtype=torch.float32, device=q.device)
    chunks = reduction.plan_row_chunks(num_rows, topk, dim, reduction.workspace_budget(workspace_bytes))
    rows_per_chunk = chunks[0][1] - chunks[0][0]
    partial = torch.empty(rows_per_chunk * topk, dim, dtype=torch.float32, device=q.device)
    dk32 = None
    for start, end in chunks:
        part = partial[: (end - start) * topk]
        # The rows kernel is the first launch of the call: the host prepares the segment glue of
        # this chunk (and the dk32 fill) while it runs instead of leaving the GPU idle before it.
        _kernels.dsa_indexer_backward_rows(
            q[start:end],
            k,
            w[start:end],
            indices[start:end],
            grad_scores[start:end],
            dq[start:end],
            dw[start:end],
            part,
        )
        if dk32 is None:
            dk32 = torch.zeros(num_kv, dim, dtype=torch.float32, device=q.device)
        reduction.deterministic_scatter_add(part, indices[start:end].reshape(-1), dk32, split=INDEXER_REDUCE_SPLIT)
    return dq, dw, dk32


def indexer_backward(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    topk_indices: torch.Tensor,
    grad_scores: torch.Tensor,
    *,
    layout: str = "thd",
    workspace_bytes: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward only: ``(d_index_q BF16, d_weights FP32, d_index_k FP32)`` in the input layout.

    ``topk_indices`` and ``grad_scores`` are ``[T, topk]`` (``thd``) or batch-major
    ``[B, S, topk]`` for both batched layouts; ``-1`` slots contribute nothing.
    """
    q_flat, k_flat, w_flat, meta = layouts.flatten_indexer_inputs(index_q, index_k, weights, layout)
    idx_flat = layouts.pad_topk(
        layouts.flatten_topk_indices(topk_indices, meta), _kernels.indexer_key_block(q_flat.shape[1])
    )
    grad_flat = layouts.flatten_scores(grad_scores, meta)
    if grad_flat.shape[1] != idx_flat.shape[1]:
        grad_flat = torch.nn.functional.pad(grad_flat, (0, idx_flat.shape[1] - grad_flat.shape[1]), value=0.0)
    # Empty slots carry no gradient (their upstream scores are -inf, so grad may be NaN / inf there):
    # the rows kernel loads an exact 0.0 for idx == -1, so no host-side masking launch is needed.
    dq, dw, dk = _flat_indexer_backward(q_flat, k_flat, w_flat, idx_flat, grad_flat.contiguous(), workspace_bytes)
    dk = meta.kv_rows_to_layout(dk)
    if layout == "thd" and index_k.dim() == 3:
        dk = dk.unsqueeze(1)
    dw = meta.rows_to_layout(dw)
    if layout == "thd" and weights.dim() == 3:
        dw = dw.unsqueeze(-1)
    return meta.rows_to_layout(dq), dw, dk


class IndexerScoresFunction(torch.autograd.Function):
    """Top-k indexer scores with the unified deterministic backward."""

    @staticmethod
    def forward(ctx, index_q, index_k, weights, topk_indices, logits, layout, workspace_bytes):
        ctx.save_for_backward(index_q, index_k, weights, topk_indices)
        ctx.layout = layout
        ctx.workspace_bytes = workspace_bytes
        return extract_topk_scores(logits, topk_indices)

    @staticmethod
    def backward(ctx, grad_scores):
        index_q, index_k, weights, topk_indices = ctx.saved_tensors
        dq, dw, dk = indexer_backward(
            index_q,
            index_k,
            weights,
            topk_indices,
            grad_scores,
            layout=ctx.layout,
            workspace_bytes=ctx.workspace_bytes,
        )
        return dq, dk, dw.to(weights.dtype), None, None, None, None


def indexer_topk_scores(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    topk_indices: torch.Tensor,
    logits: torch.Tensor,
    *,
    layout: str = "thd",
    workspace_bytes: int | None = None,
) -> torch.Tensor:
    """Gather the selected indexer scores from ``logits`` (autograd to ``index_q``, ``index_k``, ``weights``).

    ``logits`` is ``[T, T_kv]`` or ``[B, S, S_kv]`` as returned by :func:`indexer_logits`;
    ``topk_indices`` is ``[T, topk]`` or batch-local ``[B, S, topk]``.  Empty slots yield
    ``-inf`` scores and receive no gradient.
    """
    return IndexerScoresFunction.apply(index_q, index_k, weights, topk_indices, logits, layout, workspace_bytes)


def lightning_indexer(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk: int,
    *,
    layout: str = "thd",
    topk_fn: TopkFn | None = None,
    topk_indices: torch.Tensor | None = None,
    workspace_bytes: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Indexer logits -> top-k selection -> differentiable selected scores.

    ``topk_fn`` is any ``fn(logits, topk)`` callable (a replay-manager wrapper, for
    instance; :func:`torch_topk` by default); ``topk_indices`` bypasses selection
    entirely (rollout replay).  Returns ``(scores, topk_indices)``: ``[T, topk]``
    (``thd``) or batch-major ``[B, S, topk]``.
    """
    logits = indexer_logits(index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, layout=layout)
    if topk_indices is None:
        topk_indices = select_topk(logits, topk, topk_fn)
    scores = indexer_topk_scores(
        index_q,
        index_k,
        weights,
        topk_indices,
        logits,
        layout=layout,
        workspace_bytes=workspace_bytes,
    )
    return scores, topk_indices
