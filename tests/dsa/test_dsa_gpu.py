"""The DSA kernels on SM100a / SM103a / SM107a at the TP-local shapes miles runs.

* det sparse attention -- GLM-5 / DeepSeek-V3.2 ``thd`` MLA (``d_qk = 512 + 64``, no sink) at 8 / 16 / 32 / 64 heads,
  DeepSeek-V4 ``bshd`` MQA (``d = 512``, FP32 sink) at 16 / 64 heads, and the V4 window + compressed rows (128 + 512 /
  128 + 1024 slots with ``-1`` holes): output and gradients against an FP32 reference, bit-identical across passes.
* det lightning indexer -- logits and selected scores against an FP32 reference, bit-identical across passes.
* fast sparse attention -- the 64-head RoPE (DeepSeek-V3.2 TP2 / GLM-5 TP1) and NoPE (GLM-5.3) geometries against the
  same reference and against det: ``out`` and ``dq`` bit-identical across passes, ``dkv`` within the spread of its
  FP32 atomic accumulation; and the calls fast does not serve, which run det.
"""

import argparse
import logging
import math

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in {(10, 0), (10, 3), (10, 7)}:
    pytest.skip("the DSA kernels target SM100a / SM103a / SM107a", allow_module_level=True)

from miles_ext_kernels import _variant, cake_native  # noqa: E402
from miles_ext_kernels.dsa import configure, ops  # noqa: E402

DEVICE = "cuda"
TOL = 1e-2  # max |error| / max |reference|: BF16 operands and outputs, FP32 accumulation


@pytest.fixture(autouse=True)
def _det_by_default(monkeypatch):
    monkeypatch.setattr(ops, "VARIANT", "det")
    monkeypatch.setattr(_variant, "_warned", set())


def _rel_err(actual, expected):
    expected = expected.float()
    actual = actual.float()
    finite = torch.isfinite(expected)
    assert torch.equal(torch.isfinite(actual), finite), "finite/-inf pattern differs"
    scale = expected[finite].abs().max().clamp_min(1e-6)
    return ((actual[finite] - expected[finite]).abs().max() / scale).item()


def _generator(seed):
    return torch.Generator(device=DEVICE).manual_seed(seed)


def _causal_indices(rows, topk, num_keys, gen):
    """Row ``r`` selects ``min(topk, limit_r)`` distinct keys below ``limit_r = (r + 1) * num_keys // rows`` in random
    slot order; ``-1`` pads the rest."""
    limits = ((torch.arange(rows, device=DEVICE) + 1) * num_keys // rows).clamp(min=1, max=num_keys)
    keys = torch.arange(num_keys, device=DEVICE)
    score = torch.rand(rows, num_keys, device=DEVICE, generator=gen).masked_fill(keys >= limits[:, None], -1.0)
    count = min(topk, num_keys)
    picked = torch.topk(score, count, dim=-1)
    idx = torch.where(picked.values >= 0, picked.indices, -1).int()
    if count < topk:
        idx = torch.nn.functional.pad(idx, (0, topk - count), value=-1)
    return idx


def _window_compressed_indices(seq, window, compressed_topk, ratio, gen):
    """DeepSeek-V4 rows: ``window`` sliding-window keys (``-1`` before the sequence start) then ``compressed_topk``
    picks among the ``(r + 1) // ratio`` visible compressed keys (stored after the ``seq`` window keys), with ``-1``
    holes scattered over the compressed slots."""
    rows = torch.arange(seq, device=DEVICE)
    window_idx = rows[:, None] - (window - 1) + torch.arange(window, device=DEVICE)[None, :]
    window_idx = torch.where(window_idx >= 0, window_idx, -1)
    n_comp = seq // ratio
    visible = (rows + 1) // ratio
    score = torch.rand(seq, n_comp, device=DEVICE, generator=gen)
    score = score.masked_fill(torch.arange(n_comp, device=DEVICE)[None, :] >= visible[:, None], -1.0)
    picked = torch.topk(score, min(compressed_topk, n_comp), dim=-1)
    comp_idx = torch.where(picked.values >= 0, picked.indices + seq, -1)
    if comp_idx.shape[1] < compressed_topk:
        comp_idx = torch.nn.functional.pad(comp_idx, (0, compressed_topk - comp_idx.shape[1]), value=-1)
    comp_idx = comp_idx[:, torch.randperm(compressed_topk, device=DEVICE, generator=gen)]
    return torch.cat([window_idx, comp_idx], dim=-1).int(), seq + n_comp


# --------------------------------------------------------------------------- #
# FP32 reference on the flat contract (-inf sink disables, empty rows -> 0)
# --------------------------------------------------------------------------- #


def ref_sparse_attention(q, kv, indices, do, sm_scale, sink=None, d_v=512, rows_per_chunk=32):
    """FP32 forward and backward of sparse attention, ``q [T, H, D]``, ``kv [S, D]``, global ``indices [T, K]``.

    Returns ``(out, dq, dkv, dsink)``; computed in row chunks with the closed-form gradients
    (``dS = P (dP - rowsum(dO O))``), so long rows fit in memory.
    """
    q32, kv32, do32 = q.float(), kv.float(), do.float()
    num_rows, heads, dim = q.shape
    num_kv = kv.shape[0]
    out = torch.empty(num_rows, heads, d_v, device=q.device)
    dq = torch.empty(num_rows, heads, dim, device=q.device)
    dkv = torch.zeros(num_kv, dim, device=q.device)
    dsink = None if sink is None else torch.zeros(heads, device=q.device)
    sink32 = None if sink is None else sink.float().view(1, heads, 1)
    for start in range(0, num_rows, rows_per_chunk):
        rows = slice(start, min(num_rows, start + rows_per_chunk))
        idx = indices[rows].long()
        valid = (idx >= 0) & (idx < num_kv)
        safe = idx.clamp(0, num_kv - 1)
        keys = kv32[safe]  # [r, K, D]
        scores = torch.einsum("rhd,rkd->rhk", q32[rows], keys) * sm_scale
        scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
        row_max = scores.amax(dim=-1, keepdim=True)
        if sink32 is not None:
            row_max = torch.maximum(row_max, sink32)
        row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
        expo = torch.exp(scores - row_max)
        denom = expo.sum(dim=-1, keepdim=True)
        if sink32 is not None:
            denom = denom + torch.exp(sink32 - row_max)
        denom = denom.clamp_min(torch.finfo(torch.float32).tiny)
        probs = expo / denom
        o = torch.einsum("rhk,rkd->rhd", probs, keys[..., :d_v])
        out[rows] = o
        dp = torch.einsum("rhd,rkd->rhk", do32[rows], keys[..., :d_v])
        delta = (o * do32[rows]).sum(dim=-1, keepdim=True)
        ds = probs * (dp - delta)
        dq[rows] = torch.einsum("rhk,rkd->rhd", ds, keys) * sm_scale
        dkeys = torch.einsum("rhk,rhd->rkd", ds, q32[rows]) * sm_scale
        dkeys[..., :d_v] += torch.einsum("rhk,rhd->rkd", probs, do32[rows])
        dkeys = dkeys * valid[..., None]
        dkv.index_add_(0, safe.reshape(-1), dkeys.reshape(-1, dim))
        if dsink is not None:
            p_sink = torch.exp(sink32 - row_max) / denom
            dsink -= (p_sink * delta).sum(dim=(0, 2))
    return out, dq, dkv, dsink


def ref_indexer_logits(index_q, index_k, weights, ks, ke):
    sims = torch.relu(torch.einsum("thd,kd->thk", index_q.float(), index_k.float()))
    logits = torch.einsum("thk,th->tk", sims, weights.float())
    keys = torch.arange(index_k.shape[0], device=index_k.device)
    inside = (keys[None, :] >= ks[:, None]) & (keys[None, :] < ke[:, None])
    return logits.masked_fill(~inside, float("-inf"))


def ref_indexer_scores(index_q, index_k, weights, topk_indices):
    valid = topk_indices >= 0
    gathered = index_k.float()[topk_indices.clamp(min=0).long()]
    sims = torch.relu(torch.einsum("thd,tkd->thk", index_q.float(), gathered))
    return torch.einsum("thk,th->tk", sims, weights.float()).masked_fill(~valid, float("-inf"))


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #


def _run(q, kv, indices, do, *, sink=None, **kwargs):
    """One autograd pass: ``(out, dq, dkv, dsink)``."""
    q_ = q.clone().requires_grad_(True)
    kv_ = kv.clone().requires_grad_(True)
    s_ = None if sink is None else sink.clone().requires_grad_(True)
    out = ops.sparse_attention(q_, kv_, indices, attn_sink=s_, **kwargs)
    out.backward(do)
    return out.detach(), q_.grad, kv_.grad, None if s_ is None else s_.grad


def _assert_bitwise(a, b, names=("out", "dq", "dkv", "dsink")):
    for name, x, y in zip(names, a, b):
        if x is not None or y is not None:
            assert torch.equal(x, y), f"{name} is not bit-identical across passes"


def _assert_matches(got, ref, tol=TOL, names=("out", "dq", "dkv", "dsink")):
    for name, x, y in zip(names, got, ref):
        if y is not None:
            err = _rel_err(x, y)
            assert err < tol, f"{name}: rel err {err:.3e}"


def _glm5_inputs(rows, heads, topk, seed, d_qk=576, d_v=512):
    gen = _generator(seed)
    q = torch.randn(rows, heads, d_qk, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    kv = torch.randn(rows, 1, d_qk, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    indices = _causal_indices(rows, topk, rows, gen).unsqueeze(1)  # [T, 1, topk] as glm5.py passes it
    do = torch.randn(rows, heads, d_v, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    return q, kv, indices, do


def _flat_reference(q, kv, indices, do, sm_scale, sink=None):
    kv_flat = kv.reshape(-1, kv.shape[-1])
    return ref_sparse_attention(q, kv_flat, indices.reshape(q.shape[0], -1), do, sm_scale, sink)


# --------------------------------------------------------------------------- #
# det sparse attention
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("heads", "rows", "topk"),
    [(16, 512, 256), (64, 512, 256), (8, 2048, 2048), (32, 2048, 2048)],
    ids=["tp4-h16", "tp1-h64", "h8-topk2048", "h32-topk2048"],
)
def test_det_glm5_attention_matches_the_reference_and_is_bit_identical(heads, rows, topk):
    q, kv, indices, do = _glm5_inputs(rows, heads, topk, seed=1)
    sm_scale = 576**-0.5

    first = _run(q, kv, indices, do, sm_scale=sm_scale, layout="thd", variant="det")
    second = _run(q, kv, indices, do, sm_scale=sm_scale, layout="thd", variant="det")
    _assert_bitwise(first, second)

    out, dq, dkv, _ = first
    _assert_matches((out, dq, dkv.squeeze(1)), _flat_reference(q, kv, indices, do, sm_scale)[:3])


def _dsv4_inputs(batch, seq, heads, topk, seed):
    gen = _generator(seed)
    q = torch.randn(batch, seq, heads, 512, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    kv = torch.randn(batch, seq, 512, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    indices = torch.stack([_causal_indices(seq, topk, seq, gen) for _ in range(batch)])
    sink = torch.randn(heads, device=DEVICE, generator=gen)
    do = torch.randn(batch, seq, heads, 512, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    return q, kv, indices, sink, do


def _globalized(indices, seq_kv):
    batch = indices.shape[0]
    offsets = (torch.arange(batch, device=DEVICE) * seq_kv).view(batch, 1, 1)
    return torch.where(indices >= 0, indices + offsets, indices).reshape(-1, indices.shape[-1])


@pytest.mark.parametrize("heads", [16, 64], ids=["tp4-h16", "tp1-h64"])
def test_det_dsv4_attention_with_sink_matches_the_reference_and_is_bit_identical(heads):
    batch, seq, topk = 2, 256, 192  # 128 window + 64 compressed slots; padded to 256 inside
    q, kv, indices, sink, do = _dsv4_inputs(batch, seq, heads, topk, seed=4)
    sm_scale = 512**-0.5

    first = _run(q, kv, indices, do, sink=sink, sm_scale=sm_scale, layout="bshd", variant="det")
    second = _run(q, kv, indices, do, sink=sink, sm_scale=sm_scale, layout="bshd", variant="det")
    _assert_bitwise(first, second)

    out, dq, dkv, dsink = first
    ref = ref_sparse_attention(
        q.reshape(batch * seq, heads, 512),
        kv.reshape(batch * seq, 512),
        _globalized(indices, seq),
        do.reshape(batch * seq, heads, 512),
        sm_scale,
        sink,
    )
    _assert_matches(
        (out.reshape(batch * seq, heads, 512), dq.reshape(batch * seq, heads, 512), dkv.reshape(-1, 512), dsink), ref
    )


@pytest.mark.parametrize(
    ("heads", "batch", "seq", "compressed_topk"),
    [(8, 2, 2048, 512), (16, 1, 2048, 512), (64, 1, 4096, 1024)],
    ids=["h8-128+512", "h16-128+512", "h64-128+1024"],
)
def test_det_dsv4_window_and_compressed_rows_with_holes(heads, batch, seq, compressed_topk):
    gen = _generator(7)
    per_batch = [_window_compressed_indices(seq, 128, compressed_topk, 4, gen) for _ in range(batch)]
    indices = torch.stack([idx for idx, _ in per_batch])
    seq_kv = per_batch[0][1]
    assert (indices[:, :, 128:] == -1).any() and indices.shape[-1] == 128 + compressed_topk
    q = torch.randn(batch, seq, heads, 512, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    kv = torch.randn(batch, seq_kv, 512, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    sink = torch.randn(heads, device=DEVICE, generator=gen)
    do = torch.randn(batch, seq, heads, 512, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    sm_scale = 512**-0.5

    first = _run(q, kv, indices, do, sink=sink, sm_scale=sm_scale, layout="bshd", variant="det")
    second = _run(q, kv, indices, do, sink=sink, sm_scale=sm_scale, layout="bshd", variant="det")
    _assert_bitwise(first, second)

    out, dq, dkv, dsink = first
    rows = batch * seq
    ref = ref_sparse_attention(
        q.reshape(rows, heads, 512),
        kv.reshape(-1, 512),
        _globalized(indices, seq_kv),
        do.reshape(rows, heads, 512),
        sm_scale,
        sink,
    )
    _assert_matches((out.reshape(rows, heads, 512), dq.reshape(rows, heads, 512), dkv.reshape(-1, 512), dsink), ref)


def test_det_dsv4_output_allows_the_in_place_inverse_rope_before_backward():
    """DeepSeek-V4 applies the inverse RoPE in place on the attention output before the output projection."""
    q, kv, indices, sink, do = _dsv4_inputs(2, 128, 16, 64, seed=5)
    sm_scale = 512**-0.5
    q_ = q.clone().requires_grad_(True)
    out = ops.sparse_attention(q_, kv, indices, sm_scale=sm_scale, attn_sink=sink, layout="bshd", variant="det")
    assert out.shape == q.shape and out._base is None, "the output must be a fresh layout-shaped tensor, not a view"
    out[..., -64:].mul_(-1.0)  # in place on a slice, as apply_rotary_emb(o[..., -rd:], ..., inverse=True) does
    out.backward(do)
    q_ref = q.clone().requires_grad_(True)
    out_ref = ops.sparse_attention(q_ref, kv, indices, sm_scale=sm_scale, attn_sink=sink, layout="bshd", variant="det")
    scale = torch.ones_like(out_ref)
    scale[..., -64:] = -1.0
    (out_ref * scale).backward(do)
    assert torch.equal(q_.grad, q_ref.grad)


# --------------------------------------------------------------------------- #
# det lightning indexer
# --------------------------------------------------------------------------- #


def _glm5_indexer_inputs(rows, heads, seed):
    gen = _generator(seed)
    index_q = torch.randn(rows, heads, 128, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    index_k = torch.randn(rows, 128, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    weights = torch.randn(rows, heads, device=DEVICE, generator=gen)
    ks = torch.zeros(rows, device=DEVICE, dtype=torch.int32)
    ke = (torch.arange(rows, device=DEVICE) + 1).int()
    return index_q, index_k, weights, ks, ke


@pytest.mark.parametrize("heads", [32, 64])
def test_det_glm5_lightning_indexer_matches_the_reference_and_is_bit_identical(heads):
    rows, topk = 512, 128
    index_q, index_k, weights, ks, ke = _glm5_indexer_inputs(rows, heads, seed=2)

    logits = ops.indexer_logits(index_q, index_k, weights, ks, ke, layout="thd")
    assert torch.equal(logits, ops.indexer_logits(index_q, index_k, weights, ks, ke, layout="thd"))
    assert _rel_err(logits, ref_indexer_logits(index_q, index_k, weights, ks, ke)) < 1e-3

    def run():
        q_ = index_q.clone().requires_grad_(True)
        k_ = index_k.clone().requires_grad_(True)
        w_ = weights.clone().requires_grad_(True)
        scores, topk_indices = ops.lightning_indexer(q_, k_, w_, ks, ke, topk, layout="thd")
        grad = torch.randn(scores.shape, device=DEVICE, generator=_generator(3)).masked_fill(
            ~torch.isfinite(scores), 0.0
        )
        scores.masked_fill(~torch.isfinite(scores), 0.0).backward(grad)
        return scores.detach(), topk_indices, q_.grad, k_.grad, w_.grad, grad

    scores, topk_indices, dq, dk, dw, grad = run()
    again = run()
    for name, a, b in zip(("scores", "indices", "dq", "dk", "dw"), (scores, topk_indices, dq, dk, dw), again):
        assert torch.equal(a, b), f"{name} is not bit-identical across passes"

    q32 = index_q.clone().float().requires_grad_(True)
    k32 = index_k.clone().float().requires_grad_(True)
    w32 = weights.clone().requires_grad_(True)
    ref = ref_indexer_scores(q32, k32, w32, topk_indices)
    assert _rel_err(scores, ref) < 1e-3
    ref.masked_fill(~torch.isfinite(ref), 0.0).backward(grad)
    assert _rel_err(dq, q32.grad) < 2e-2  # ReLU mask flips at |q.k| ~ bf16 eps
    assert _rel_err(dk, k32.grad) < 1e-2
    assert _rel_err(dw, w32.grad) < 1e-3


@pytest.mark.parametrize("seq_len_kv", [512, 509], ids=["aligned", "unaligned"])
def test_det_indexer_logits_on_short_key_rows(seq_len_kv):
    index_q, index_k, weights, ks, ke = _glm5_indexer_inputs(512, 32, seed=6)
    index_k = index_k[:seq_len_kv].contiguous()
    ke = ke.clamp(max=seq_len_kv)

    logits = ops.indexer_logits(index_q, index_k, weights, ks, ke, layout="thd")
    assert logits.shape == (512, seq_len_kv)
    assert _rel_err(logits, ref_indexer_logits(index_q, index_k, weights, ks, ke)) < 1e-3


def test_det_dsv4_batched_indexer_logits_match_the_reference():
    gen = _generator(5)
    seq, batch, heads, ratio = 256, 2, 64, 4
    seq_kv = seq // ratio
    q = torch.randn(seq, batch, heads, 128, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    k = torch.randn(seq_kv, batch, 128, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    weights = torch.randn(seq, batch, heads, device=DEVICE, generator=gen)
    positions = torch.arange(seq, device=DEVICE, dtype=torch.int32)
    ks = torch.zeros(seq, device=DEVICE, dtype=torch.int32)
    ke = ((positions + 1) // ratio).to(torch.int32)

    logits = ops.indexer_logits(q, k, weights, ks, ke, layout="sbhd")
    assert logits.shape == (batch, seq, seq_kv)
    assert torch.equal(logits, ops.indexer_logits(q, k, weights, ks, ke, layout="sbhd"))
    for b in range(batch):
        assert _rel_err(logits[b], ref_indexer_logits(q[:, b], k[:, b], weights[:, b], ks, ke)) < 1e-3


# --------------------------------------------------------------------------- #
# fast sparse attention and its det fallback
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("d_qk", "sm_scale", "rows", "topk"),
    [(576, 576**-0.5, 2048, 2048), (512, 1.0 / 16.0, 2048, 2048), (576, 576**-0.5, 512, 200)],
    ids=["rope-topk2048", "nope-topk2048", "rope-short-topk"],
)
def test_fast_64_heads_matches_the_reference_and_det(d_qk, sm_scale, rows, topk):
    q, kv, indices, do = _glm5_inputs(rows, 64, topk, seed=11, d_qk=d_qk)

    first = _run(q, kv, indices, do, sm_scale=sm_scale, layout="thd", variant="fast")
    second = _run(q, kv, indices, do, sm_scale=sm_scale, layout="thd", variant="fast")
    # out and dq are written once per row; dkv sums FP32 atomics in a run-dependent order
    _assert_bitwise(first[:2], second[:2], names=("out", "dq"))
    assert _rel_err(second[2], first[2]) < TOL

    out, dq, dkv, _ = first
    assert dkv.shape == kv.shape and dkv.dtype == torch.bfloat16
    _assert_matches((out, dq, dkv.squeeze(1)), _flat_reference(q, kv, indices, do, sm_scale)[:3])
    det = _run(q, kv, indices, do, sm_scale=sm_scale, layout="thd", variant="det")
    _assert_matches((out, dq, dkv), det[:3], tol=2 * TOL)


def test_fast_runs_batched_rows():
    batch, seq, topk = 2, 512, 256
    gen = _generator(12)
    q = torch.randn(batch, seq, 64, 576, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    kv = torch.randn(batch, seq, 576, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    indices = torch.stack([_causal_indices(seq, topk, seq, gen) for _ in range(batch)])
    do = torch.randn(batch, seq, 64, 512, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    sm_scale = 576**-0.5

    out, dq, dkv, _ = _run(q, kv, indices, do, sm_scale=sm_scale, layout="bshd", variant="fast")
    assert out.shape == (batch, seq, 64, 512) and dq.shape == q.shape and dkv.shape == kv.shape
    rows = batch * seq
    ref = ref_sparse_attention(
        q.reshape(rows, 64, 576), kv.reshape(-1, 576), _globalized(indices, seq), do.reshape(rows, 64, 512), sm_scale
    )
    _assert_matches((out.reshape(rows, 64, 512), dq.reshape(rows, 64, 576), dkv.reshape(-1, 576)), ref[:3])


def test_calls_fast_does_not_serve_run_det_bit_identically_and_warn_once(caplog):
    configure(argparse.Namespace(cake_dsa_variant="fast"))
    assert ops.VARIANT == "fast"
    q, kv, indices, sink, do = _dsv4_inputs(1, 256, 64, 192, seed=13)
    sm_scale = 512**-0.5
    glm_q, glm_kv, glm_indices, glm_do = _glm5_inputs(256, 16, 128, seed=14)

    with caplog.at_level(logging.WARNING, logger="miles_ext_kernels"):
        sink_fast = _run(q, kv, indices, do, sink=sink, sm_scale=sm_scale, layout="bshd")
        heads_fast = _run(glm_q, glm_kv, glm_indices, glm_do, sm_scale=576**-0.5, layout="thd")
        _run(glm_q, glm_kv, glm_indices, glm_do, sm_scale=576**-0.5, layout="thd")
    _assert_bitwise(sink_fast, _run(q, kv, indices, do, sink=sink, sm_scale=sm_scale, layout="bshd", variant="det"))
    _assert_bitwise(
        heads_fast, _run(glm_q, glm_kv, glm_indices, glm_do, sm_scale=576**-0.5, layout="thd", variant="det")
    )
    assert [record.getMessage() for record in caplog.records] == [
        "miles_ext_kernels.dsa: the fast variant does not serve calls with an attention sink; running det",
        "miles_ext_kernels.dsa: the fast variant does not serve 16 local query heads (only 64); running det",
    ]


def test_prebuild_warms_both_variants_for_a_served_geometry():
    ops.prebuild(heads=64, d_qk=576, variant="fast")
    ops.prebuild(heads=64, d_qk=512, variant="fast")
    ops.prebuild(heads=16, d_qk=576, attn_sink=True, variant="fast")  # det only
    q, kv, indices, do = _glm5_inputs(256, 64, 128, seed=15)
    out, dq, dkv, _ = _run(q, kv, indices, do, sm_scale=576**-0.5, layout="thd", variant="fast")
    assert all(torch.isfinite(t.float()).all() for t in (out, dq, dkv))


def test_check_device_names_the_architecture():
    assert ops.check_device() == cake_native.ARCH_BY_CAPABILITY[torch.cuda.get_device_capability()]


def test_check_device_needs_the_architectures_toolkit(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    with pytest.raises(NotImplementedError, match="SM100a, SM103a or SM107a"):
        ops.check_device()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 7))
    monkeypatch.setattr(cake_native, "MIN_CUDA", {**cake_native.MIN_CUDA, "sm_107a": (99, 0)})
    with pytest.raises(RuntimeError, match=r"DSA training kernels for sm_107a needs CUDA 99\.0 or newer"):
        ops.check_device()
    assert math.isclose(ops.LOG2E, math.log2(math.e))
