"""The Cake mHC kernels (``miles_ext_kernels.mhc``) on SM100a / SM103a / SM107a.

Ported from FlashInfer's backend suites of the three vendored packages (``tests/mhc/test_cake_mhc_det_train.py``,
``tests/experimental/test_cake_mhc_train.py``, ``tests/mhc/test_cake_mhc_det_train_head.py``) at every hidden size
the device serves (4096, 5120, 7168; 4096 only on SM107a, where the others are refused) and token counts 1, 257, 2049
and 16263, through the autograd entry points miles calls:

* ``det``: every output against a pure-torch FP64 reference of the function (autograd through the FP64 forward)
  under FlashInfer's bars -- BF16 ``U``, ``Y``, ``dB``, ``dX``: ``|out - ref| <= 1e-2 + 1e-2 |ref|`` on every element;
  FP32 ``dfn``, ``dscale``, ``dbase``: rel-L2 <= 1e-5, and ``dscale`` / ``dbase`` per element <= 1e-4 relative (floor:
  the vector's RMS) -- and a second run bitwise identical;
* ``fast``: ``U``, ``Y``, ``dB`` and ``dX`` bitwise the ``det`` outputs, ``dscale`` / ``dbase`` too outside the
  ``det`` group-row band (hidden 4096 above 14927 tokens, where ``det`` sums them in another fixed order: there under
  the FP32 bars), ``dfn`` under the FP32 bars and within FlashInfer's summation-order bound of the ``det`` ``dfn``
  (``F u 1.02 sum_t |dz'_t| |u_t|``, ``u = 2^-24``, ``F`` FlashInfer's factor of the token count: ``2 P`` for the
  ``P`` chunk partials both sides sum, ``2 max(P_lpt, P) + 10`` where the column slab runs its two-size ``lpt``
  partition), and a second run with identical ``U`` / ``Y`` / ``dB`` / ``dX`` / ``dscale`` / ``dbase``;
* the variant of a call: :data:`ops.VARIANT` set by ``configure``, the handle naming its package, and ``fast`` under
  torch's deterministic algorithms running ``det`` (bitwise) with one logged warning;
* the head: bit for bit torch's BF16 ``X.mean(dim=1)`` and its gradient, at hidden 4096, 5120 and 7168;
* :func:`ops.prebuild` covers every library the calls load.

Inputs are FlashInfer's unit regime: ``X``, ``B``, ``dY``, ``dU`` ~ N(0, 1) BF16, ``fn`` ~ 1e-2 N(0, 1),
``scale`` ~ 1 + 0.1 N, ``base`` ~ 0.1 N, drawn in FlashInfer's order from the device generator seeded 2032.
"""

import argparse
import logging

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

from miles_ext_kernels import _variant, cake_native  # noqa: E402
from miles_ext_kernels.mhc import arguments, ops  # noqa: E402

DEVICE = torch.device("cuda", torch.cuda.current_device())
CAPABILITY = tuple(torch.cuda.get_device_capability(DEVICE))
if CAPABILITY not in ops.SUPPORTED_CAPABILITIES:
    pytest.skip(
        f"the Cake mHC kernels run SM100a / SM103a / SM107a, not capability {CAPABILITY}", allow_module_level=True
    )
ARCH = cake_native.ARCH_BY_CAPABILITY[CAPABILITY]
# the hidden sizes the device serves: FlashInfer exports the packages for SM107a at hidden 4096 only
SERVED = (4096,) if ARCH == "sm_107a" else arguments.HIDDEN_SIZES

BF16, F32, F64 = torch.bfloat16, torch.float32, torch.float64
STREAMS, MAPPING_ROWS = 4, 24
SEED = 2032
BLOCK_TOKENS = 8192  # rows of the FP64 reference per block
BF16_ATOL = BF16_RTOL = 1e-2
FP32_REL_L2_MAX = 1e-5
FP32_ELEM_REL_MAX = 1e-4
FP32_UNIT = 2.0**-24
DFN_ABS_SLACK = 1.02
CASES = [(hidden, T) for hidden in SERVED for T in (1, 257, 2049, 16263)]
BF16_OUTPUTS = ("U", "Y", "dB", "dX")
PARAMETER_GRADS = ("dfn", "dscale", "dbase")


@pytest.fixture(autouse=True, scope="module")
def prebuilt():
    """Compile the libraries of both variants and the head first: no test pays for a build mid-call."""
    for hidden in SERVED:
        for variant in ("det", "fast"):
            ops.prebuild(DEVICE, hidden, variant=variant, head=True)
    yield


@pytest.fixture(autouse=True)
def _fast_unless_named(monkeypatch):
    monkeypatch.setattr(ops, "VARIANT", "fast")


def _make_inputs(T: int, hidden: int, seed: int = SEED) -> dict:
    g = torch.Generator(device=DEVICE).manual_seed(seed)

    def randn(*shape: int, dtype: torch.dtype = F32) -> torch.Tensor:
        return torch.randn(*shape, generator=g, device=DEVICE, dtype=F32).to(dtype)

    X = randn(T, STREAMS, hidden, dtype=BF16)
    fn = randn(MAPPING_ROWS, STREAMS * hidden) * 1e-2
    scale = 1.0 + 0.1 * randn(3)
    base = 0.1 * randn(MAPPING_ROWS)
    B = randn(T, hidden, dtype=BF16)
    dY = randn(T, STREAMS, hidden, dtype=BF16)
    dU = randn(T, hidden, dtype=BF16)
    return {"X": X, "fn": fn.contiguous(), "scale": scale, "base": base, "B": B, "dY": dY, "dU": dU}


def _run(i: dict, variant: str | None) -> dict:
    """One connection forward and backward through the entry points miles calls; ``B`` stands for the branch."""
    X, fn, scale, base, B = (i[k].detach().clone().requires_grad_(True) for k in ("X", "fn", "scale", "base", "B"))
    U, handle = ops.mhc_prepare(X, fn, scale, base, variant=variant)
    Y = ops.mhc_finish(X, B, handle)
    torch.autograd.backward([U, Y], [i["dU"], i["dY"]])
    return {
        "U": U.detach(),
        "Y": Y.detach(),
        "dB": B.grad,
        "dX": X.grad,
        "dfn": fn.grad,
        "dscale": scale.grad,
        "dbase": base.grad,
        "handle": type(handle),
    }


def _sinkhorn(logits: torch.Tensor) -> torch.Tensor:
    C = torch.softmax(logits, dim=-1) + ops.HC_EPS
    C = C / (C.sum(dim=1, keepdim=True) + ops.HC_EPS)
    for _ in range(ops.SINKHORN_ITERS - 1):
        C = C / (C.sum(dim=2, keepdim=True) + ops.HC_EPS)
        C = C / (C.sum(dim=1, keepdim=True) + ops.HC_EPS)
    return C


def _reference_block(X, B, dY, dU, fn, scale, base) -> dict:
    """FP64 forward of one row block, its backward by autograd, and ``sum_t |dz'_t| |u_t|`` for the order bound."""
    T = X.shape[0]
    with torch.enable_grad():
        Xd, fnd, sd, bd, Bd = (t.detach().to(F64).requires_grad_(True) for t in (X, fn, scale, base, B))
        u = Xd.reshape(T, -1)
        r = torch.rsqrt(u.square().mean(dim=-1, keepdim=True) + ops.RMS_EPS)
        z = (u * r) @ fnd.T
        pre = torch.sigmoid(z[:, 0:4] * sd[0] + bd[0:4]) + ops.HC_EPS
        post = 2 * torch.sigmoid(z[:, 4:8] * sd[1] + bd[4:8])
        C = _sinkhorn((z[:, 8:] * sd[2] + bd[8:]).reshape(T, STREAMS, STREAMS))
        U = (pre.unsqueeze(-1) * Xd).sum(dim=1)
        Y = post.unsqueeze(-1) * Bd.unsqueeze(1) + torch.einsum("tij,tid->tjd", C, Xd)
        dX, dfn, dscale, dbase, dB, dz = torch.autograd.grad(
            [U, Y], [Xd, fnd, sd, bd, Bd, z], grad_outputs=[dU.to(F64), dY.to(F64)]
        )
    with torch.no_grad():
        dfn_abs = (dz * r).abs().T @ u.abs()
    return {
        "U": U.detach(),
        "Y": Y.detach(),
        "dB": dB,
        "dX": dX,
        "dfn": dfn,
        "dscale": dscale,
        "dbase": dbase,
        "dfn_abs": dfn_abs,
    }


def _reference(i: dict) -> dict:
    T = i["X"].shape[0]
    blocks = [
        _reference_block(
            *(i[k][lo : lo + BLOCK_TOKENS] for k in ("X", "B", "dY", "dU")), i["fn"], i["scale"], i["base"]
        )
        for lo in range(0, T, BLOCK_TOKENS)
    ]
    out = {k: torch.cat([b[k] for b in blocks]) for k in BF16_OUTPUTS}
    for k in (*PARAMETER_GRADS, "dfn_abs"):
        out[k] = sum(b[k] for b in blocks[1:]) + blocks[0][k]
    return out


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view({BF16: torch.int16, F32: torch.int32}[t.dtype])


def _assert_bitwise(a: torch.Tensor, b: torch.Tensor, label: str) -> None:
    assert a.dtype == b.dtype and a.shape == b.shape, (label, a.dtype, b.dtype, tuple(a.shape), tuple(b.shape))
    differ = _bits(a) != _bits(b)
    assert not differ.any(), f"{label}: {int(differ.sum())} of {differ.numel()} elements differ"


def _assert_bf16(out: torch.Tensor, ref: torch.Tensor, label: str) -> None:
    assert out.dtype == BF16 and tuple(out.shape) == tuple(ref.shape), (label, out.dtype, tuple(out.shape))
    err = (out.to(F64) - ref).abs()
    bad = err > BF16_ATOL + BF16_RTOL * ref.abs()
    assert (
        torch.isfinite(out).all() and not bad.any()
    ), f"{label}: {int(bad.sum())} of {bad.numel()} elements outside atol = rtol = 1e-2 (max abs err {float(err.max()):.3e})"


def _assert_fp32(out: torch.Tensor, ref: torch.Tensor, label: str, *, per_element: bool) -> None:
    assert out.dtype == F32 and tuple(out.shape) == tuple(ref.shape), (label, out.dtype, tuple(out.shape))
    assert torch.isfinite(out).all(), f"{label}: non-finite values"
    err = (out.to(F64) - ref).abs()
    rel_l2 = float(err.norm() / ref.norm())
    assert rel_l2 <= FP32_REL_L2_MAX, f"{label}: rel-L2 {rel_l2:.3e} > {FP32_REL_L2_MAX:.0e}"
    if per_element:
        rms = ref.square().mean().sqrt()
        worst = float((err / torch.maximum(ref.abs(), rms)).max())
        assert worst <= FP32_ELEM_REL_MAX, f"{label}: per-element relative error {worst:.3e} > {FP32_ELEM_REL_MAX:.0e}"


def _det_group_rows(T: int, hidden: int) -> bool:
    """Whether ``det`` sums ``dscale`` / ``dbase`` over 8-token group rows at ``T`` (``fast`` keeps its per-CTA form)."""
    jit = ops._det().cake_jit
    return jit.bwd_group_rows(T, jit.sm_count(DEVICE.index), jit.arch_for_capability(CAPABILITY), hidden)


def _dfn_factor(T: int, hidden: int) -> float:
    """``F`` of the ``dfn`` summation-order bound at ``T`` (FlashInfer's ``_dfn_factor``): ``2 P`` where the column
    slab sums the partials of the ``P`` chunks ``det`` sums (at most ``P`` FP32 roundings on each side), ``2 max(P_lpt,
    P) + 10`` where it runs the two-size ``lpt`` partition (the partials are another partition's)."""
    fast, det = ops._fast().cake_jit, ops._det().cake_jit
    sm_count, arch = fast.sm_count(DEVICE.index), fast.arch_for_capability(CAPABILITY)
    parts = fast.host_plan(T, sm_count, arch, hidden)["parts"]
    assert parts == det.host_plan(T, sm_count, arch, hidden)["parts"], (T, hidden)
    assert 1 <= parts <= 32, (T, hidden, parts)
    if not fast.bwd_slab_atomic_program(T, sm_count, arch, hidden).startswith("lpt"):
        return 2.0 * parts
    lpt_parts = fast.slab_plan(T, sm_count, arch, hidden)["lpt_parts"]
    assert 1 <= lpt_parts <= 32, (T, hidden, lpt_parts)
    return 2.0 * max(lpt_parts, parts) + 10.0


@pytest.mark.parametrize(("hidden", "T"), CASES, ids=[f"H{hidden}-T{T}" for hidden, T in CASES])
def test_det_and_fast_match_the_fp64_reference_and_each_other(hidden: int, T: int) -> None:
    i = _make_inputs(T, hidden)
    det = [_run(i, "det") for _ in range(2)]
    fast = [_run(i, "fast") for _ in range(2)]
    ref = _reference(i)
    label = f"H={hidden} T={T}"

    assert det[0]["handle"] is ops._det()._MHCTrainHandle
    assert fast[0]["handle"] is ops._fast()._MHCTrainHandle
    for k in BF16_OUTPUTS:
        _assert_bf16(det[0][k], ref[k], f"det {k} {label}")
    for k in PARAMETER_GRADS:
        _assert_fp32(det[0][k], ref[k], f"det {k} {label}", per_element=k != "dfn")
    for k in (*BF16_OUTPUTS, *PARAMETER_GRADS):
        _assert_bitwise(det[1][k], det[0][k], f"det run to run {k} {label}")

    group_rows = _det_group_rows(T, hidden)
    for n, run in enumerate(fast):
        for k in BF16_OUTPUTS:
            _assert_bitwise(run[k], det[0][k], f"fast run {n} {k} vs det {label}")
        for k in ("dscale", "dbase"):
            if group_rows:
                _assert_fp32(run[k], ref[k], f"fast run {n} {k} {label}", per_element=True)
            else:
                _assert_bitwise(run[k], det[0][k], f"fast run {n} {k} vs det {label}")
        _assert_fp32(run["dfn"], ref["dfn"], f"fast run {n} dfn {label}", per_element=False)
        bound = _dfn_factor(T, hidden) * FP32_UNIT * DFN_ABS_SLACK * ref["dfn_abs"]
        excess = (run["dfn"].to(F64) - det[0]["dfn"].to(F64)).abs() - bound
        assert float(excess.max()) <= 0, f"fast run {n} dfn {label}: beyond the summation-order bound of det's dfn"
    for k in (*BF16_OUTPUTS, "dscale", "dbase"):
        _assert_bitwise(fast[1][k], fast[0][k], f"fast run to run {k} {label}")


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_configure_sets_the_variant_of_calls_that_name_none(variant: str) -> None:
    arguments.configure(argparse.Namespace(cake_mhc_variant=variant))
    i = _make_inputs(257, 4096)

    expected = (ops._det() if variant == "det" else ops._fast())._MHCTrainHandle
    assert _run(i, None)["handle"] is expected


def test_fast_under_torch_deterministic_algorithms_runs_det_and_says_so_once(monkeypatch, caplog) -> None:
    i = _make_inputs(257, 4096)
    det = _run(i, "det")
    monkeypatch.setattr(_variant, "_warned", set())
    torch.use_deterministic_algorithms(True)
    try:
        with caplog.at_level(logging.WARNING, logger="miles_ext_kernels"):
            runs = [_run(i, "fast"), _run(i, None)]
    finally:
        torch.use_deterministic_algorithms(False)

    for run in runs:
        assert run["handle"] is ops._det()._MHCTrainHandle
        for k in (*BF16_OUTPUTS, *PARAMETER_GRADS):
            _assert_bitwise(run[k], det[k], f"fallback {k}")
    warnings = [record for record in caplog.records if "fast variant does not serve" in record.getMessage()]
    assert len(warnings) == 1, [record.getMessage() for record in caplog.records]


def test_the_device_serves_its_hidden_sizes_and_refuses_the_others() -> None:
    assert ops.supported_hidden_sizes(DEVICE) == SERVED
    for hidden in arguments.HIDDEN_SIZES:
        assert ops.is_supported(DEVICE, hidden) is (hidden in SERVED)
        if hidden not in SERVED:
            for variant in ("det", "fast"):
                with pytest.raises(ValueError, match=rf"the hidden sizes \(4096,\) on {ARCH}, not {hidden}"):
                    ops.prebuild(DEVICE, hidden, variant=variant)


def test_prebuild_needs_the_architectures_toolkit(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    with pytest.raises(ValueError, match=r"compute capability"):
        ops.prebuild(DEVICE, 4096, variant="det")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 7))
    monkeypatch.setattr(cake_native, "MIN_CUDA", {**cake_native.MIN_CUDA, "sm_107a": (99, 0)})
    with pytest.raises(RuntimeError, match=r"mHC kernels for sm_107a needs CUDA 99\.0 or newer"):
        ops.prebuild(DEVICE, 4096, variant="det")


def test_finish_refuses_a_foreign_handle() -> None:
    i = _make_inputs(1, 4096)

    with pytest.raises(TypeError, match="handle must be the handle mhc_prepare returned"):
        ops.mhc_finish(i["X"], i["B"], object())


@pytest.mark.parametrize("hidden", SERVED)
@pytest.mark.parametrize("T", [1, 257, 16263])
def test_the_head_is_torch_mean_bit_for_bit(hidden: int, T: int) -> None:
    g = torch.Generator(device=DEVICE).manual_seed(SEED)
    X = torch.randn(T, STREAMS, hidden, generator=g, device=DEVICE).to(BF16)
    dY = torch.randn(T, hidden, generator=g, device=DEVICE).to(BF16)
    Xk, Xr = X.clone().requires_grad_(True), X.clone().requires_grad_(True)

    Y, Yr = ops.mhc_head(Xk), Xr.mean(dim=1)
    _assert_bitwise(Y, Yr, f"head Y H={hidden} T={T}")
    Y.backward(dY)
    Yr.backward(dY)
    _assert_bitwise(Xk.grad, Xr.grad, f"head dX H={hidden} T={T}")


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_prebuild_covers_the_libraries_the_calls_load(monkeypatch, variant: str) -> None:
    names = set(ops.prebuild(DEVICE, 4096, variant=variant))
    package = ops._det() if variant == "det" else ops._fast()
    for name in names:
        assert package.cake_build.library_path(name).is_file(), name

    loaded = []
    load_library = package.cake_build.load_library

    def recording(name):
        loaded.append(name)
        return load_library(name)

    monkeypatch.setattr(package.cake_build, "load_library", recording)
    package.cake_jit.load.cache_clear()
    for T in (3001, 9001):  # token counts no other test runs: fresh launch plans load their libraries again
        _run(_make_inputs(T, 4096), variant)

    assert loaded and set(loaded) <= names, sorted(set(loaded) - names)
