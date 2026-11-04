"""The Cake MoE grouped GEMM (``moe_grouped_gemm.grouped_mm``) on the expert projections of miles' torchtitan MoE
models (expert-parallel local shapes): the forward, input and weight gradients against a per-group FP32 reference over
imbalanced groups with empty and one-row ones, bit-identical ``det`` reruns, ``fast`` giving ``det``'s forward and
input-gradient bits, its unordered weight-gradient tail, the fallback to ``det`` under torch's deterministic mode, and
the prebuild hook. Needs a CUDA device; the tests need SM100a / SM103a and skip elsewhere.
"""

import importlib
import logging

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

from miles_ext_kernels import _variant  # noqa: E402
from miles_ext_kernels.moe_grouped_gemm import SUPPORTED_CAPABILITIES, grouped_mm, ops, prebuild  # noqa: E402

if torch.cuda.get_device_capability() not in SUPPORTED_CAPABILITIES:
    pytest.skip("the Cake MoE grouped GEMM needs SM100a / SM103a", allow_module_level=True)

SEED = 20260930
# the vendored package's bounds of a BF16 output against its reference
ATOL = RTOL = 1e-2
# Group sizes, cycled over the experts: an empty group, a one-row group, sizes that are not tile multiples.
RAGGED = (300, 0, 1, 700, 4099, 33, 511, 257, 2048)
# (hidden size H, expert intermediate size I, local experts E): the routed experts of an MoE layer on one
# expert-parallel rank. Each gives two projections: w13 (gate and up, [E, 2I, H]) and w2 (down, [E, H, I]).
LAYERS = {
    "h2048-i768-e16": (2048, 768, 16),
    "h2048-i1536-e8": (2048, 1536, 8),
    "h4096-i2048-e9": (4096, 2048, 9),
}
CASES = [(layer, proj) for layer in LAYERS for proj in ("w13", "w2")]
# More than 4096 rows per group on average and K a multiple of 512: the weight gradient takes the 512-wide k tile,
# whose split tail the fast variant adds with unordered reductions (the vendored package's K512_E3 and one of its
# shapes whose tail splits on every SM count).
K512_SIZES = (16400, 0, 16666)
K512_SHAPE = (4864, 1024)  # (N, K)


@pytest.fixture(scope="module", autouse=True)
def _programs():
    """Compile both variants' programs up front, in parallel, instead of one by one on first use."""
    return {variant: prebuild(variant=variant) for variant in _variant.VARIANTS}


@pytest.fixture(autouse=True)
def _fp32_reference():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


def _sizes(groups: int) -> list[int]:
    return [RAGGED[e % len(RAGGED)] for e in range(groups)]


def _shape(layer: str, proj: str) -> tuple[list[int], int, int]:
    hidden, inter, groups = LAYERS[layer]
    n, k = (2 * inter, hidden) if proj == "w13" else (hidden, inter)
    return _sizes(groups), n, k


def _offs(sizes) -> torch.Tensor:
    return torch.tensor(sizes, dtype=torch.int32, device="cuda").cumsum(0, dtype=torch.int32)


def _inputs(sizes, n: int, k: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(sum(sizes), k, generator=gen, device="cuda") * k**-0.5).to(torch.bfloat16)
    w = (torch.randn(len(sizes), n, k, generator=gen, device="cuda") * k**-0.5).to(torch.bfloat16)
    g = torch.randn(sum(sizes), n, generator=gen, device="cuda").to(torch.bfloat16)
    return x, w, g, _offs(sizes)


def _bounds(sizes):
    ends = torch.tensor(sizes).cumsum(0).tolist()
    return list(zip([0, *ends[:-1]], ends))


def _reference(x, w, g, sizes):
    """Per-group FP32 products of the BF16 operands: ``(y, dx, dw)``."""
    y = torch.zeros(x.shape[0], w.shape[1], device="cuda")
    dx, dw = torch.zeros(x.shape, device="cuda"), torch.zeros(w.shape, device="cuda")
    for e, (start, end) in enumerate(_bounds(sizes)):
        if end > start:
            rows_x, rows_g, w_e = x[start:end].float(), g[start:end].float(), w[e].float()
            y[start:end] = rows_x @ w_e.T
            dx[start:end] = rows_g @ w_e
            dw[e] = rows_g.T @ rows_x
    return y, dx, dw


def _step(x, w, g, offs, variant):
    x = x.detach().clone().requires_grad_(True)
    w = w.detach().clone().requires_grad_(True)
    y = grouped_mm(x, w, offs, variant=variant)
    y.backward(g)
    torch.cuda.synchronize()
    return y.detach(), x.grad, w.grad


def _assert_close(name, got, want):
    assert got.dtype == torch.bfloat16 and got.shape == want.shape, name
    torch.testing.assert_close(got.float(), want, atol=ATOL, rtol=RTOL, msg=lambda m: f"{name}: {m}")


def _assert_empty_groups_zero(dw, sizes):
    for e, size in enumerate(sizes):
        if size == 0:
            assert not dw[e].any(), f"the weight gradient of empty group {e} must be exactly zero"


@pytest.mark.parametrize("layer, proj", CASES)
def test_det_matches_the_reference_and_reruns_bit_identically(layer, proj):
    sizes, n, k = _shape(layer, proj)
    x, w, g, offs = _inputs(sizes, n, k, seed=SEED + n + k)
    first, second = _step(x, w, g, offs, "det"), _step(x, w, g, offs, "det")

    for name, got, again, want in zip(("y", "dx", "dw"), first, second, _reference(x, w, g, sizes)):
        assert torch.equal(got, again), f"det {name} differs between two identical calls"
        _assert_close(name, got, want)
    _assert_empty_groups_zero(first[2], sizes)


@pytest.mark.parametrize("layer, proj", CASES)
def test_fast_gives_the_det_bits_where_its_reductions_are_ordered(layer, proj):
    """With fewer than 4096 rows per group on average the weight gradient keeps the 256-wide k tile and its ordered
    tail reduction: fast is bitwise det in all three products."""
    sizes, n, k = _shape(layer, proj)
    x, w, g, offs = _inputs(sizes, n, k, seed=SEED + n + k)
    det, fast = _step(x, w, g, offs, "det"), _step(x, w, g, offs, "fast")

    for name, got, want in zip(("y", "dx", "dw"), fast, det):
        assert torch.equal(got, want), f"fast {name} differs from det"


def _vendored_backend(variant: str):
    package = ops._det() if variant == "det" else ops._fast()
    return importlib.import_module(package[0].__name__.rpartition(".")[0] + ".cake_backend")


def test_the_fast_weight_gradient_adds_its_split_tail_unordered_within_the_bounds():
    sizes, (n, k) = list(K512_SIZES), K512_SHAPE
    x, w, g, offs = _inputs(sizes, n, k, seed=SEED + 512)
    launch = _vendored_backend("fast").prepare_grouped_gemm_wgrad(g, x, offs, deterministic=False)
    assert launch.plan["tile_k"] == 512 and launch.plan["unordered_tail"], launch.plan

    det, fast = _step(x, w, g, offs, "det"), _step(x, w, g, offs, "fast")
    reference = _reference(x, w, g, sizes)
    assert torch.equal(fast[0], det[0]) and torch.equal(fast[1], det[1]), "fast y / dx differ from det"
    for name, got, want in zip(("det dw", "fast dw"), (det[2], fast[2]), (reference[2], reference[2])):
        _assert_close(name, got, want)
    _assert_empty_groups_zero(fast[2], sizes)
    assert torch.equal(_step(x, w, g, offs, "det")[2], det[2]), "det dw differs between two identical calls"


def test_new_group_boundaries_on_the_same_shapes_are_read_on_the_device():
    sizes, n, k = _shape("h2048-i768-e16", "w2")
    x, w, g, _ = _inputs(sizes, n, k, seed=SEED)
    for variant in _variant.VARIANTS:
        for moved in (sizes, sizes[::-1], [sum(sizes)] + [0] * (len(sizes) - 1)):
            got = _step(x, w, g, _offs(moved), variant)
            for name, out, want in zip(("y", "dx", "dw"), got, _reference(x, w, g, moved)):
                _assert_close(f"{variant} {name} {moved[:3]}", out, want)


def test_torch_deterministic_mode_runs_det_and_says_so_once(monkeypatch, caplog):
    sizes, n, k = _shape("h2048-i1536-e8", "w13")
    x, w, g, offs = _inputs(sizes, n, k, seed=SEED + 1)
    det = _step(x, w, g, offs, "det")
    monkeypatch.setattr(_variant, "_warned", set())
    monkeypatch.setattr(ops, "VARIANT", "fast")  # the configured default the second call runs
    enabled, warn_only = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    torch.use_deterministic_algorithms(True, warn_only=True)  # warn only: the test's own reference uses cuBLAS
    try:
        with caplog.at_level(logging.WARNING, logger="miles_ext_kernels"):
            first, second = _step(x, w, g, offs, "fast"), _step(x, w, g, offs, None)
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)

    for got in (first, second):
        for name, out, want in zip(("y", "dx", "dw"), got, det):
            assert torch.equal(out, want), f"{name} under torch's deterministic mode differs from det"
    fallbacks = [r for r in caplog.records if "fast variant does not serve" in r.getMessage()]
    assert len(fallbacks) == 1, [r.getMessage() for r in caplog.records]


def test_calls_the_programs_do_not_serve_are_refused():
    x = torch.empty(64, 1024, dtype=torch.bfloat16, device="cuda")
    w = torch.empty(4, 512, 1024, dtype=torch.bfloat16, device="cuda")
    offs = _offs([16] * 4)
    with pytest.raises(ValueError, match="variant must be one of"):
        grouped_mm(x, w, offs, variant="fastest")
    with pytest.raises(ValueError, match="multiple of 256"):
        grouped_mm(x, torch.empty(4, 384, 1024, dtype=torch.bfloat16, device="cuda"), offs)
    with pytest.raises(ValueError, match="multiple of 256"):
        grouped_mm(x[:, :960], torch.empty(4, 512, 960, dtype=torch.bfloat16, device="cuda"), offs)
    with pytest.raises(ValueError, match="int32 cumulative row ends"):
        grouped_mm(x, w, offs.long())
    with pytest.raises(ValueError, match="int32 cumulative row ends"):
        grouped_mm(x, w, _offs([16] * 5))
    with pytest.raises(ValueError, match="BF16"):
        grouped_mm(x.float(), w, offs)
    with pytest.raises(ValueError, match="x \\[sum_m, K\\] and w \\[E, N, K\\]"):
        grouped_mm(x, w[..., :512], offs)


@pytest.mark.parametrize("variant", _variant.VARIANTS)
def test_prebuild_compiles_every_program_the_layers_launch(_programs, variant):
    """Every stage the vendored planner binds for the layers' three products is among the prebuilt libraries."""
    _, cake_jit, _ = ops._det() if variant == "det" else ops._fast()
    backend = _vendored_backend(variant)
    arch = SUPPORTED_CAPABILITIES[torch.cuda.get_device_capability()]
    prebuilt = {path.stem.rsplit("_", 2)[0] for path in _programs[variant]}  # <library>_<closure20>_<key16>.so
    assert prebuilt
    mode = {} if variant == "det" else {"deterministic": False}
    prepare = {
        op: getattr(backend, f"prepare_grouped_{'det_' if variant == 'det' else ''}gemm_{op}")
        for op in ("fwd", "dgrad", "wgrad")
    }

    shapes = [_shape(layer, proj) for layer, proj in CASES] + [(list(K512_SIZES), *K512_SHAPE)]
    launched = set()
    for sizes, n, k in shapes:
        x, w, g, offs = _inputs(sizes, n, k, seed=SEED)
        for launch in (
            prepare["fwd"](x, w, offs, **mode),
            prepare["dgrad"](g, w, offs, **mode),
            prepare["wgrad"](g, x, offs, **mode),
        ):
            for stage in launch.stages:
                binding = cake_jit.stage_binding(launch.record_name, stage, arch)
                launched.add(cake_jit.library_name(binding["program"], arch, binding["specializations"]))

    assert launched and launched <= prebuilt, sorted(launched - prebuilt)
