"""
Copyright (c) 2026 by FlashInfer team.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Seeded inputs and the exact per-document FP64 reference of packed KDA chunked training (head dimension 128,
natural-log decay gate, sigmoid beta), ported from FlashInfer ``tests/test_helpers/cake_kda_det_train_reference.py``
at c5db8d3ccd3fa3842812280f7f208718dc0b8610, trimmed to the raw-gate helpers these tests use.

The reference is the token recurrence of one document, in FP64 with autograd, state ``S`` in ``R^{K x V}`` starting
from zero::

    S   = S * exp(g_t)[:, None]                   # per-key-dimension decay
    S   = S + (beta_t * k_t) (x) (v_t - k_t^T S)  # delta rule
    o_t = scale * q_t^T S

with the operands taken as exact FP64 values, checkpointed in 64-token blocks so the backward of a few thousand
tokens fits in memory. The raw gate (``use_gate_in_kernel=True``) is activated in the same FP64 graph as ``g = lb *
sigmoid(exp(A_log)[h] * (raw + dt_bias[h]))``, so autograd gives the gradients of the raw gate and of both
parameters too.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence

import torch
from torch.utils.checkpoint import checkpoint

D = 128
LOWER_BOUND = -5.0  # the gate's lower bound: g = -5 * sigmoid(...)
GRAD_NAMES = ("dq", "dk", "dv", "dg", "dbeta")  # autograd.grad order of the leaves (q, k, v, g, beta)
# the raw-gate form: the leaves (q, k, v, raw, beta, A_log, dt_bias); "dg" is the raw gate's gradient there
RAW_GRAD_NAMES = GRAD_NAMES + ("dA_log", "ddt_bias")


def cu_seqlens_of(lengths: Sequence[int]) -> list[int]:
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + int(n))
    return cu


def make_raw_inputs(
    lengths: Sequence[int], H: int, device: str | torch.device = "cuda", *, seed: int | None = None
) -> dict[str, torch.Tensor]:
    """The raw-gate operands of documents of ``lengths`` tokens with ``H`` heads: ``q, k`` BF16 ``[S, H, 128]``
    (L2-normalised in FP32 before the BF16 store), ``v`` BF16, ``g`` the raw gate BF16 ``[S, H, 128]``, ``beta`` BF16
    ``[S, H]`` (after the sigmoid), ``A_log`` FP32 ``[H]``, ``dt_bias`` FP32 ``[H * 128]``, ``cu_seqlens`` int32 and
    ``dout`` BF16 ``[S, H, 128]``. Device-independent: a CPU generator seeded from ``seed`` (default: a CRC of the
    shape)."""
    S = int(sum(lengths))
    if seed is None:
        seed = zlib.crc32(f"{list(lengths)}:{H}".encode())
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen, dtype=torch.float32)

    def l2norm_bf16(x: torch.Tensor) -> torch.Tensor:
        return (x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)).to(torch.bfloat16)

    q = l2norm_bf16(randn(S, H, D))
    k = l2norm_bf16(randn(S, H, D))
    v = (randn(S, H, D) * 0.5).to(torch.bfloat16)
    a_log = torch.log(torch.rand(H, generator=gen, dtype=torch.float32) + 1.0)  # decay rates in [1, 2)
    dt_bias = randn(H, D) * 0.5
    raw = randn(S, H, D)
    beta = torch.sigmoid(randn(S, H)).to(torch.bfloat16)
    dout = (randn(S, H, D) * 0.5).to(torch.bfloat16)
    out = {
        "q": q,
        "k": k,
        "v": v,
        "g": raw.to(torch.bfloat16),
        "beta": beta,
        "cu_seqlens": torch.tensor(cu_seqlens_of(lengths), dtype=torch.int32),
        "A_log": a_log.contiguous(),
        "dt_bias": dt_bias.flatten().contiguous(),
        "dout": dout,
    }
    return {name: t.to(device) for name, t in out.items()}


def raw_gate(
    raw: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float = LOWER_BOUND,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """The activated gate ``lower_bound * sigmoid(exp(A_log)[h] * (raw + dt_bias[h]))`` of a raw gate ``[..., H,
    128]`` in ``dtype``, differentiable in ``raw``, ``A_log`` and ``dt_bias`` (``[H * 128]`` or ``[H, 128]``)."""
    H = raw.shape[-2]
    x = raw.to(dtype)
    a = torch.exp(A_log.to(dtype)).reshape(H, 1)
    b = dt_bias.to(dtype).reshape(H, D)
    return lower_bound * torch.sigmoid(a * (x + b))


def _block(S0, q, k, v, g, beta, scale):
    # S0 [H, K, V]; q, k, v, g [T, H, D]; beta [T, H]  (all FP64)
    St = S0
    outs = []
    for t in range(q.shape[0]):
        St = St * torch.exp(g[t])[:, :, None]
        kt = k[t]
        pred = torch.einsum("hk,hkv->hv", kt, St)
        St = St + torch.einsum("hk,hv->hkv", beta[t][:, None] * kt, v[t] - pred)
        outs.append(scale * torch.einsum("hk,hkv->hv", q[t], St))
    return torch.stack(outs, 0), St


def kda_exact_forward(q, k, v, g, beta, cu_seqlens, scale, block=64, dtype=torch.float64):
    """``q, k, v, g [S, H, D]``, ``beta [S, H]``, ``cu_seqlens`` 1-D int (device or host): ``out [S, H, D]`` in
    ``dtype`` from the exact per-document recurrence."""
    S, H, K = q.shape
    V = v.shape[-1]
    o = torch.zeros(S, H, V, dtype=dtype, device=q.device)
    cu = [int(x) for x in cu_seqlens.tolist()]
    for d in range(len(cu) - 1):
        s0, s1 = cu[d], cu[d + 1]
        if s1 <= s0:
            continue
        St = torch.zeros(H, K, V, dtype=dtype, device=q.device)
        for b0 in range(s0, s1, block):
            b1 = min(b0 + block, s1)
            ob, St = checkpoint(
                _block, St, q[b0:b1], k[b0:b1], v[b0:b1], g[b0:b1], beta[b0:b1], scale, use_reentrant=False
            )
            o[b0:b1] = ob
    return o


def error_metrics(cand: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    """max |err|, RMS err, relative RMS (err RMS / ref RMS) and the reference RMS, computed in FP64."""
    c = cand.detach().to(torch.float64)
    r = ref.detach().to(torch.float64)
    err = c - r
    rms_ref = torch.sqrt(torch.mean(r * r)).item() if r.numel() else 0.0
    rms_err = torch.sqrt(torch.mean(err * err)).item() if err.numel() else 0.0
    if rms_ref > 0:
        rel_rms = rms_err / rms_ref
    else:
        rel_rms = float("inf") if rms_err > 0 else 0.0
    return {
        "max_abs": err.abs().max().item() if err.numel() else 0.0,
        "rms": rms_err,
        "rel_rms": rel_rms,
        "ref_rms": rms_ref,
    }


def elementwise_gate(cand: torch.Tensor, ref: torch.Tensor, atol: float, rtol: float) -> dict:
    """``|cand - ref| <= atol + rtol * |ref|`` elementwise in FP32; the mismatch and non-finite counts."""
    c = cand.detach().to(torch.float32)
    r = ref.detach().to(torch.float32)
    bad = (c - r).abs() > (atol + rtol * r.abs())
    nonfinite = int((~torch.isfinite(c)).sum().item())
    mismatch = int(bad.sum().item())
    return {
        "mismatch": mismatch,
        "numel": int(c.numel()),
        "nonfinite": nonfinite,
        "pass": mismatch == 0 and nonfinite == 0,
    }
