"""The FP32 router gate GEMM (``dense_gemm.router_gating``) against FP64: logits from the BF16 operands (K1), input and
weight gradients in FP32 (K2) rounded to BF16, the top-k routing, and bit-identical reruns, for the expert counts and
hidden sizes of miles' MoE models at an odd token count. Then the vendored package's own router cases and the prebuild
hook. Needs a CUDA device; the tests need SM100a / SM103a and skip elsewhere.
"""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

from miles_ext_kernels.dense_gemm import SUPPORTED_CAPABILITIES, ops, prebuild_router, router_gating  # noqa: E402

if torch.cuda.get_device_capability() not in SUPPORTED_CAPABILITIES:
    pytest.skip("the Cake dense projection GEMMs need SM100a / SM103a", allow_module_level=True)

SEED = 20260930
EXPERTS = (128, 256, 288, 384, 512)
HIDDEN = (2048, 4096, 5120, 6144, 7168)
TOKENS = 1001
TOP_K = 8
F32_REL_FRO_MAX = 5e-5
F32_MAX_ABS_OVER_REF_ABSMAX = 1e-3
ROUTER_REL_FRO_MAX = 2e-6
REPEATS = 10
# Top-k agreement is checked on the rows whose k-th and (k+1)-th FP64 logits are further apart than this (relative to
# the largest logit): closer ties may order either way under any FP32 accumulation.
TOP_K_TIE_MARGIN = 1e-4
# The error of an FP32 dot product before its BF16 rounding, relative to the sum of its absolute products: chunked FP32
# accumulation over K <= 7168 stays far below it.
FP32_DOT_REL = 2.0**-16


@pytest.fixture(scope="module", autouse=True)
def _programs():
    """Compile the programs of every case up front, each prebuild in parallel, instead of one by one on first use."""
    for experts in EXPERTS:
        for hidden in HIDDEN:
            prebuild_router(experts, hidden, token_counts=(TOKENS,))


def _inputs(experts: int, hidden: int, tokens: int, seed: int):
    """The router's input (as Megatron hands it, [s, b, h]), its weight and an output gradient of router scale."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(tokens, 1, hidden, generator=gen, device="cuda").to(torch.bfloat16)
    weight = (torch.randn(experts, hidden, generator=gen, device="cuda") * 0.02).to(torch.bfloat16)
    grad = torch.randn(tokens, 1, experts, generator=gen, device="cuda") * 1e-3
    return x, weight, grad


def _step(x, weight, grad):
    x = x.detach().clone().requires_grad_(True)
    weight = weight.detach().clone().requires_grad_(True)
    logits = router_gating(x, weight)
    logits.backward(grad)
    return logits.detach(), x.grad, weight.grad


def _bf16_spacing(expected: torch.Tensor) -> torch.Tensor:
    """The BF16 spacing at each element of ``expected``: 2 ** (exponent - 7)."""
    return torch.exp2(torch.floor(torch.log2(expected.abs().clamp_min(2.0**-126))) - 7)


def _assert_top_k_agrees(logits: torch.Tensor, expected: torch.Tensor) -> None:
    """The top-k experts of ``logits`` are those of the FP64 ``expected`` on every row without a near tie at k."""
    ranked = torch.topk(expected, TOP_K + 1, dim=1).values
    clear = (ranked[:, TOP_K - 1] - ranked[:, TOP_K]) > TOP_K_TIE_MARGIN * float(expected.abs().max())
    assert int(clear.sum()) >= 0.99 * expected.shape[0], "too many near ties to check the routing"
    top = torch.topk(logits.double(), TOP_K, dim=1).indices.sort(dim=1).values
    ref_top = torch.topk(expected, TOP_K, dim=1).indices.sort(dim=1).values
    assert torch.equal(top[clear], ref_top[clear])


@pytest.mark.parametrize("hidden", HIDDEN)
@pytest.mark.parametrize("experts", EXPERTS)
def test_router_matches_fp64_and_reruns_bit_identically(experts, hidden):
    x, weight, grad = _inputs(experts, hidden, TOKENS, seed=SEED + experts + hidden)
    first, second = _step(x, weight, grad), _step(x, weight, grad)
    for name, got, again in zip(("logits", "input gradient", "weight gradient"), first, second):
        assert torch.equal(got, again), f"{name} differs between two identical calls"
    logits, dx, dw = first
    x2, w, g = x.double().reshape(-1, hidden), weight.double(), grad.double().reshape(-1, experts)

    expected = x2 @ w.T
    assert logits.dtype == torch.float32 and logits.shape == (TOKENS, 1, experts)
    diff = (logits.reshape(-1, experts).double() - expected).abs()
    assert float(diff.norm() / expected.norm()) <= F32_REL_FRO_MAX
    assert float(diff.max() / expected.abs().max()) <= F32_MAX_ABS_OVER_REF_ABSMAX
    _assert_top_k_agrees(logits.reshape(-1, experts), expected)
    # the FP32 gradients rounded once to BF16: within one BF16 spacing of the exact value, plus the FP32 error of
    # elements that cancel to near zero
    gradients = (
        ("input gradient", dx.reshape(-1, hidden), g @ w, g.abs() @ w.abs()),
        ("weight gradient", dw, g.T @ x2, g.abs().T @ x2.abs()),
    )
    for name, got, want, magnitude in gradients:
        assert got.dtype == torch.bfloat16 and got.shape == want.shape, name
        excess = (got.double() - want).abs() - _bf16_spacing(want) - FP32_DOT_REL * magnitude
        assert (
            float(excess.max()) <= 0
        ), f"{name}: {int((excess > 0).sum())} elements off by more than one BF16 spacing"


def test_router_refuses_what_the_programs_do_not_take():
    x = torch.empty(4, 4096, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="BF16"):
        router_gating(x, torch.empty(256, 4096, device="cuda"))
    with pytest.raises(ValueError, match="multiples of 8"):
        router_gating(x, torch.empty(260, 4096, dtype=torch.bfloat16, device="cuda"))
    with pytest.raises(ValueError, match="weight \\[E, H\\]"):
        router_gating(x, torch.empty(256, 2048, dtype=torch.bfloat16, device="cuda"))


def test_prebuild_compiles_every_program_the_router_launches():
    backend, cake_jit, _ = ops._vendored()
    arch = SUPPORTED_CAPABILITIES[torch.cuda.get_device_capability()]
    paths = prebuild_router(256, 6144, token_counts=(TOKENS, 4097))
    assert paths and all(path.is_file() for path in paths)
    prebuilt = {path.stem.rsplit("_", 2)[0] for path in paths}  # <library>_<closure20>_<key16>.so

    backend.clear_convenience_caches()
    for tokens in (TOKENS, 4097):
        _step(*_inputs(256, 6144, tokens, seed=SEED))
    caches = (backend._K1_CACHE, backend._SWAP_CACHE, backend._K2_CACHE)
    launched = {
        cake_jit.library_name(entry.module_name, arch) for cache in caches for entry in cache.values() if entry
    }

    assert launched and launched <= prebuilt, sorted(launched - prebuilt)


# ---------------------------------------------------------------------------
# Cases of the vendored package's own tests (tests/experimental/test_cake_dense_projection_det_gemm.py at the
# vendored commit), on its prepared and convenience launches.
# ---------------------------------------------------------------------------


def _router_inputs(op, T, seed, K=6144, N=256):
    g = torch.Generator(device="cuda").manual_seed(seed)
    X = torch.randn(T, K, device="cuda", generator=g)
    W = torch.randn(N, K, device="cuda", generator=g) * 0.02
    G = torch.randn(T, N, device="cuda", generator=g) * 1e-3
    if op == "fwd":
        return dict(A=X, B=W.t(), out=torch.empty(T, N, device="cuda"), X=X, W=W, G=G)
    if op == "dgrad":
        return dict(A=G, B=W, out=torch.empty(T, K, device="cuda"), X=X, W=W, G=G)
    return dict(A=G.t(), B=X, out=torch.empty(N, K, device="cuda"), X=X, W=W, G=G)


@pytest.mark.parametrize("op, T", [("fwd", 257), ("dgrad", 257), ("wgrad", 1001), ("fwd", 129)])
def test_router_rows_match_fp64_reference_and_rerun_bit_exact(op, T):
    backend = ops._vendored()[0]
    v = _router_inputs(op, T, seed=SEED + 4)
    prepared = backend.prepare_router_fp32_det_gemm(v["A"], v["B"], v["out"])
    assert prepared.splits == backend.ROUTER_SPLITS[prepared.plan.layout]
    first = prepared.launch().clone()
    prepared.launch()
    torch.cuda.synchronize()
    assert torch.equal(first, v["out"])
    expected = torch.matmul(v["A"].double(), v["B"].double())
    rel_fro = float((v["out"].double() - expected).norm() / expected.norm())
    assert bool(torch.isfinite(v["out"]).all()) and rel_fro <= ROUTER_REL_FRO_MAX, rel_fro
    if op == "fwd":  # top-8 routing agreement with the FP64 reference on the forward rows
        ref_top = torch.topk(expected, 8, dim=1).indices.sort(dim=1).values
        top = torch.topk(v["out"].double(), 8, dim=1).indices.sort(dim=1).values
        assert int((top != ref_top).any(dim=1).sum()) == 0
    # the eager entry point of the row gives the same bits
    eager = dict(
        fwd=lambda: backend.router_det_forward(v["X"], v["W"]),
        dgrad=lambda: backend.router_det_dgrad(v["G"], v["W"]),
        wgrad=lambda: backend.router_det_wgrad(v["G"], v["X"]),
    )[op]
    backend.clear_convenience_caches()
    assert torch.equal(eager(), v["out"])
    assert torch.equal(eager(), v["out"])  # the second call runs the cached plan


@pytest.mark.parametrize("op, T", [("fwd", 257), ("dgrad", 257), ("wgrad", 1001)])
def test_repeated_router_launches_are_bitwise_identical(op, T):
    """The router's prepared launch (split, split-K GEMM, reduction) rerun ``REPEATS`` times, its output poisoned
    with NaN before every rerun, gives the first launch's bits each time."""
    backend = ops._vendored()[0]
    v = _router_inputs(op, T, seed=SEED + 8)
    prepared = backend.prepare_router_fp32_det_gemm(v["A"], v["B"], v["out"])
    first = prepared.launch().clone()
    torch.cuda.synchronize()
    assert bool(torch.isfinite(first).all())
    for _ in range(REPEATS):
        v["out"].fill_(float("nan"))
        prepared.launch()
        torch.cuda.synchronize()
        assert torch.equal(v["out"], first)
