"""Packed causal GQA attention (``miles_ext_kernels.gqa``) on SM100a / SM103a / SM107a.

Against an FP64 reference of the same BF16 inputs, on FlashInfer's suite layouts (unequal lengths, tails that are no
multiple of a tile, keys fewer than queries, empty segments, long keys) and on the tensor-parallel-local head pairs
miles runs (8/2, 6/1, 16/2, 8/1) over a packed micro-batch whose last segment is its alignment pad: the output, the
LSE and every gradient within FlashInfer's declared budget, rows that see no key exactly zero. Run to run, the output,
the LSE and dK / dV are bitwise and dQ is within one BF16 step of its FP32 reordering. Also: document isolation,
non-reentrant and reentrant checkpointing, no device synchronization in steady state, the rejected operands and
devices, and the prebuild hook covering every library a forward and backward load, without importing FlashInfer.
Needs a B200 / GB200, B300 / GB300 or R200 and skips elsewhere.
"""

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

from torch.utils.checkpoint import checkpoint  # noqa: E402

from miles_ext_kernels import cake_native  # noqa: E402
from miles_ext_kernels.gqa import ops  # noqa: E402

if tuple(torch.cuda.get_device_capability()) not in ops.SUPPORTED_CAPABILITIES:
    pytest.skip(
        f"the kernels run on {sorted(ops.SUPPORTED_CAPABILITIES)}, not {torch.cuda.get_device_capability()}",
        allow_module_level=True,
    )

REPO_ROOT = Path(__file__).resolve().parents[2]
D = 256
SCALE = D**-0.5  # the default, as miles runs it
MIB = 2**20

# FlashInfer's accuracy budget for this package: per layout and tensor, the relative L2 error against the FP64
# reference of the same BF16 inputs is at most 1.10 times FlashAttention CuTe's (its dedicated head-dim-256 backward)
# on B300 against the same reference.
FA4_REL_L2 = {"out": 2.3e-3, "lse": 1.5e-7, "dq": 2.6e-3, "dk": 2.6e-3, "dv": 2.4e-3}
BUDGET = 1.10
# dQ is reduced in FP32 with atomic adds, one partial per 128-key block, in a scheduling order, then rounded to BF16
# once: two runs of one input differ by one BF16 rounding step (DQ_RTOL) plus an absolute floor for the FP32
# reordering error where partials cancel, DQ_ATOL_FRACTION of the tensor's largest magnitude.
DQ_RTOL = 2**-7
DQ_ATOL_FRACTION = 2**-16
# Checkpointing: memory kept after a non-reentrant checkpointed forward, less the output storage.
CKPT_RETAINED_LIMIT_MIB = 2.0

# (name, q_lens, k_lens, H_q, H_kv)
FI_LAYOUTS = (
    ("tails", (77, 300, 1, 129), (77, 700, 1000, 129), 24, 4),
    ("keys_fewer_than_queries", (200, 150, 64), (77, 0, 63), 24, 4),
    ("empty_segments", (0, 100, 0, 37, 0), (0, 300, 50, 37, 0), 24, 4),
    ("zigzag_cp2", (250, 250, 129, 130), (250, 1000, 129, 517), 24, 4),
    ("long_keys", (256, 255), (4099, 2049), 24, 4),
    ("group_2", (300, 77), (513, 77), 8, 4),
)
# A miles micro-batch: three sequences, then the pad that aligns the packed tokens to a multiple of 128 as its own
# segment; self-attention, so the query and key boundaries are one.
MILES_SEGMENTS = (1000, 517, 2049, 18)
MILES_HEADS = ((8, 2), (6, 1), (16, 2), (8, 1))
MILES_LAYOUTS = tuple((f"miles_{h_q}_{h_kv}", MILES_SEGMENTS, MILES_SEGMENTS, h_q, h_kv) for h_q, h_kv in MILES_HEADS)
LAYOUTS = FI_LAYOUTS + MILES_LAYOUTS


def _ids(layouts):
    return [layout[0] for layout in layouts]


@pytest.fixture(scope="module")
def device() -> torch.device:
    dev = torch.device("cuda", torch.cuda.current_device())
    # build every library of the forward and backward before the first checked call
    assert len(ops.prebuild(dev)) == len(ops.STAGES)
    return dev


def _cu(lens) -> list[int]:
    cu = [0]
    for n in lens:
        cu.append(cu[-1] + int(n))
    return cu


def _inputs(q_lens, k_lens, h_q, h_kv, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    t_q, t_k = sum(q_lens), sum(k_lens)

    def randn(*shape):
        return torch.randn(*shape, generator=g, device=device, dtype=torch.float32).to(torch.bfloat16)

    return {
        "q": randn(t_q, h_q, D),
        "k": randn(t_k, h_kv, D),
        "v": randn(t_k, h_kv, D),
        "dout": randn(t_q, h_q, D),
        "cu_q": torch.tensor(_cu(q_lens), dtype=torch.int32, device=device),
        "cu_k": torch.tensor(_cu(k_lens), dtype=torch.int32, device=device),
        "cq": _cu(q_lens),
        "ck": _cu(k_lens),
    }


def _longest_keys(x) -> int:
    ck = x["ck"]
    return max(ck[i + 1] - ck[i] for i in range(len(ck) - 1))


def _reference(x):
    """FP64 out, lse and gradients of the same BF16 inputs (bottom-right causal per segment)."""
    q = x["q"].double().requires_grad_()
    k = x["k"].double().requires_grad_()
    v = x["v"].double().requires_grad_()
    group = q.shape[1] // k.shape[1]
    outs, lses = [], []
    for b in range(len(x["cq"]) - 1):
        q0, q1, k0, k1 = x["cq"][b], x["cq"][b + 1], x["ck"][b], x["ck"][b + 1]
        ql, kl = q1 - q0, k1 - k0
        ks = k[k0:k1].repeat_interleave(group, dim=1)
        vs = v[k0:k1].repeat_interleave(group, dim=1)
        s = torch.einsum("qhd,khd->hqk", q[q0:q1], ks) * SCALE
        seen = torch.arange(kl, device=q.device)[None, :] <= torch.arange(ql, device=q.device)[:, None] + (kl - ql)
        has_key = seen.any(dim=-1)
        # rows without a key take finite dummy scores (no NaN reaches the gradients) and a zero P
        s = s.masked_fill(~seen, -math.inf).masked_fill(~has_key[None, :, None], 0.0)
        lse = torch.logsumexp(s, dim=-1)
        p = torch.exp(s - lse[..., None]) * seen
        outs.append(torch.einsum("hqk,khd->qhd", p, vs))
        lses.append(lse.detach().masked_fill(~has_key[None, :], -math.inf))
    out = torch.cat(outs)
    out.backward(x["dout"].double())
    return {"out": out.detach(), "lse": torch.cat(lses, dim=1), "dq": q.grad, "dk": k.grad, "dv": v.grad}


def _run(x, *, max_seqlen_k=None):
    """Forward and backward through autograd, plus the forward's LSE: ``{"out", "lse", "dq", "dk", "dv"}``."""
    q, k, v = (x[t].clone().requires_grad_() for t in ("q", "k", "v"))
    out = ops.varlen_gqa_attention(q, k, v, x["cu_q"], x["cu_k"], softmax_scale=SCALE, max_seqlen_k=max_seqlen_k)
    out.backward(x["dout"])
    _, lse = ops.backend().gqa_attention_fwd(x["q"], x["k"], x["v"], x["cu_q"], x["cu_k"], softmax_scale=SCALE)
    return {"out": out.detach(), "lse": lse, "dq": q.grad, "dk": k.grad, "dv": v.grad}


def _rel_l2(got: torch.Tensor, ref: torch.Tensor) -> float:
    got, ref = got.double(), ref.double()
    finite = torch.isfinite(ref)
    num = torch.linalg.vector_norm((got - ref)[finite])
    den = torch.linalg.vector_norm(ref[finite])
    return float(num / den) if den > 0 else float(num)


def _assert_dq_close(got: torch.Tensor, ref: torch.Tensor) -> None:
    """Two dQ of one input (see DQ_RTOL / DQ_ATOL_FRACTION)."""
    floor = DQ_ATOL_FRACTION * float(ref.float().abs().max()) if ref.numel() else 0.0
    torch.testing.assert_close(got, ref, rtol=DQ_RTOL, atol=floor)


def _no_key_rows(x):
    """Mask ``[T_q]`` of the query rows that see no key (``i + k_len - q_len < 0``)."""
    rows = []
    for b in range(len(x["cq"]) - 1):
        ql, kl = x["cq"][b + 1] - x["cq"][b], x["ck"][b + 1] - x["ck"][b]
        rows += [i + kl - ql < 0 for i in range(ql)]
    return torch.tensor(rows, dtype=torch.bool, device=x["q"].device)


@pytest.mark.parametrize("layout", LAYOUTS, ids=_ids(LAYOUTS))
def test_matches_the_fp64_reference(layout, device):
    name, q_lens, k_lens, h_q, h_kv = layout
    x = _inputs(q_lens, k_lens, h_q, h_kv, device)
    ref = _reference(x)
    # miles hands over the longest segment; FlashInfer's layouts also run the T_k bound
    got = _run(x, max_seqlen_k=_longest_keys(x) if name.startswith("miles") else None)
    torch.cuda.synchronize()
    errors = {}
    for t in ("out", "lse", "dq", "dk", "dv"):
        assert torch.isfinite(got[t][torch.isfinite(ref[t])]).all(), f"{name}: non-finite {t}"
        errors[t] = _rel_l2(got[t], ref[t])
    print(f"REL_L2 {name} " + " ".join(f"{t}={e:.3e}" for t, e in errors.items()))
    for t, e in errors.items():
        assert e <= BUDGET * FA4_REL_L2[t], f"{name}: {t} rel-L2 {e:.3e} > {BUDGET} x {FA4_REL_L2[t]:.1e}"
    # rows that see no key: out 0, lse -inf, dq 0, exactly
    no_key = _no_key_rows(x)
    assert torch.equal(got["out"][no_key], torch.zeros_like(got["out"][no_key]))
    assert bool((got["lse"][:, no_key] == -math.inf).all())
    assert bool((got["lse"][:, ~no_key] > -math.inf).all())
    assert torch.equal(got["dq"][no_key], torch.zeros_like(got["dq"][no_key]))
    # keys no query sees (a segment without queries) get exactly zero gradients
    unseen = (ref["dk"].abs().amax(dim=(1, 2)) == 0) & (ref["dv"].abs().amax(dim=(1, 2)) == 0)
    assert torch.equal(got["dk"][unseen], torch.zeros_like(got["dk"][unseen]))
    assert torch.equal(got["dv"][unseen], torch.zeros_like(got["dv"][unseen]))


@pytest.mark.parametrize("layout", MILES_LAYOUTS, ids=_ids(MILES_LAYOUTS))
def test_run_to_run_everything_but_dq_is_bitwise(layout, device):
    _, q_lens, k_lens, h_q, h_kv = layout
    x = _inputs(q_lens, k_lens, h_q, h_kv, device, seed=5)
    first = _run(x, max_seqlen_k=_longest_keys(x))
    second = _run(x, max_seqlen_k=_longest_keys(x))
    torch.cuda.synchronize()
    for t in ("out", "lse", "dk", "dv"):
        assert torch.equal(first[t], second[t]), t
    _assert_dq_close(second["dq"], first["dq"])


def test_documents_are_isolated(device):
    q_lens, k_lens = (130, 77, 300, 41), (500, 77, 301, 1000)
    x = _inputs(q_lens, k_lens, 8, 2, device)
    base = _run(x)
    again = _run(x)
    y = dict(x)
    b = 2
    q0, q1, k0, k1 = x["cq"][b], x["cq"][b + 1], x["ck"][b], x["ck"][b + 1]
    for key, lo, hi in (("q", q0, q1), ("dout", q0, q1), ("k", k0, k1), ("v", k0, k1)):
        y[key] = x[key].clone()
        y[key][lo:hi] = -y[key][lo:hi] * 1.5
    pert = _run(y)
    torch.cuda.synchronize()
    q_other = torch.ones(sum(q_lens), dtype=torch.bool, device=device)
    q_other[q0:q1] = False
    k_other = torch.ones(sum(k_lens), dtype=torch.bool, device=device)
    k_other[k0:k1] = False
    assert not torch.equal(pert["out"][~q_other], base["out"][~q_other])
    assert torch.equal(pert["out"][q_other], base["out"][q_other])
    assert torch.equal(pert["lse"][:, q_other], base["lse"][:, q_other])
    assert torch.equal(pert["dk"][k_other], base["dk"][k_other])
    assert torch.equal(pert["dv"][k_other], base["dv"][k_other])
    # dQ is reduced with atomic adds: the other segments agree to the reduction-order rounding of two plain runs
    _assert_dq_close(again["dq"], base["dq"])
    _assert_dq_close(pert["dq"][q_other], base["dq"][q_other])


@pytest.mark.parametrize("reentrant", [False, True], ids=["non_reentrant", "reentrant"])
def test_checkpointing_matches_eager(reentrant, device):
    x = _inputs((1000, 3000, 128), (1000, 3000, 128), 8, 2, device)

    def fn(q, k, v):
        return ops.varlen_gqa_attention(q, k, v, x["cu_q"], x["cu_k"], softmax_scale=SCALE, max_seqlen_k=3000)

    def step(use_checkpoint):
        q, k, v = (x[t].clone().requires_grad_() for t in ("q", "k", "v"))
        out = checkpoint(fn, q, k, v, use_reentrant=reentrant) if use_checkpoint else fn(q, k, v)
        out.backward(x["dout"])
        torch.cuda.synchronize()
        return out.detach(), q.grad, k.grad, v.grad

    if not reentrant:
        q, k, v = (x[t].clone().requires_grad_() for t in ("q", "k", "v"))
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        out = checkpoint(fn, q, k, v, use_reentrant=False)
        torch.cuda.synchronize()
        kept = torch.cuda.memory_allocated() - before - out.untyped_storage().nbytes()
        assert kept / MIB <= CKPT_RETAINED_LIMIT_MIB, f"a checkpointed forward kept {kept / MIB:.2f} MiB"
        del out
    eager = step(False)
    ckpt = step(True)
    assert torch.equal(ckpt[0], eager[0])
    _assert_dq_close(ckpt[1], eager[1])
    assert torch.equal(ckpt[2], eager[2]) and torch.equal(ckpt[3], eager[3])


def test_steady_state_does_not_synchronize(device):
    xs = [_inputs(q, q, 8, 2, device, seed=i) for i, q in enumerate(((700, 77, 3), (300,)))]

    def step(x):
        q, k, v = (x[t].clone().requires_grad_() for t in ("q", "k", "v"))
        out = ops.varlen_gqa_attention(q, k, v, x["cu_q"], x["cu_k"], softmax_scale=SCALE, max_seqlen_k=700)
        out.backward(x["dout"])

    for x in xs:
        step(x)
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for x in xs + xs:
            step(x)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    torch.cuda.synchronize()


def test_rejects_unsupported_operands(device):
    x = _inputs((10,), (10,), 24, 4, device)
    q, k, v, cq, ck = x["q"], x["k"], x["v"], x["cu_q"], x["cu_k"]
    with pytest.raises(ValueError, match="even multiple"):
        ops.varlen_gqa_attention(q[:, :12], k[:, :4], v[:, :4], cq, ck)  # group 3
    with pytest.raises(ValueError, match="bfloat16"):
        ops.varlen_gqa_attention(q[..., :128], k[..., :128], v[..., :128], cq, ck)  # head dim 128
    with pytest.raises(ValueError, match="bfloat16"):
        ops.varlen_gqa_attention(q.float(), k.float(), v.float(), cq, ck)
    with pytest.raises(ValueError, match="int32"):
        ops.varlen_gqa_attention(q, k, v, cq.long(), ck.long())


def test_device_arch_names_this_device_and_rejects_others(monkeypatch, device):
    capability = tuple(torch.cuda.get_device_capability(device))
    assert ops.device_arch(device) == ops.SUPPORTED_CAPABILITIES[capability]
    for other in ((9, 0), (10, 1), (12, 0)):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None, other=other: other)
        with pytest.raises(RuntimeError, match="SM100a, SM103a or SM107a"):
            ops.device_arch()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 7))
    monkeypatch.setattr(cake_native, "MIN_CUDA", {**cake_native.MIN_CUDA, "sm_107a": (99, 0)})
    with pytest.raises(RuntimeError, match=r"attention kernels for sm_107a needs CUDA 99\.0 or newer"):
        ops.device_arch()


_LOADS = """
import sys

import torch

from miles_ext_kernels.gqa import ops

build = ops.backend().cake_build
loaded = []
load_library = build.load_library


def recording_load_library(name):
    loaded.append(name)
    return load_library(name)


build.load_library = recording_load_library
paths = ops.prebuild()
assert len(paths) == len(ops.STAGES) and all(path.is_file() for path in paths), paths
g = torch.Generator(device="cuda").manual_seed(0)
cu = torch.tensor([0, 300, 811], dtype=torch.int32, device="cuda")
q, k, v = (
    torch.randn(811, h, 256, generator=g, device="cuda").bfloat16().requires_grad_() for h in (8, 2, 2)
)
out = ops.varlen_gqa_attention(q, k, v, cu, cu, max_seqlen_k=511)
out.backward(torch.randn_like(out))
torch.cuda.synchronize()
expected = ops.libraries(ops.device_arch())
assert sorted(loaded) == sorted(expected), (loaded, expected)
flashinfer = sorted(n for n in sys.modules if n == "flashinfer" or n.startswith("flashinfer."))
assert flashinfer == [], flashinfer
print("loaded", len(loaded))
"""


def test_prebuild_covers_every_library_a_forward_and_backward_load_without_flashinfer(device):
    pythonpath = os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", _LOADS],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": pythonpath},
        capture_output=True,
        text=True,
        timeout=1800,
    )

    assert result.returncode == 0, result.stderr[-5000:]
    assert result.stdout.split() == ["loaded", str(len(ops.STAGES))]
