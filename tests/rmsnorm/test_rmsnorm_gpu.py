"""The Cake zero-centred input RMSNorm (``miles_ext_kernels.rmsnorm``) against an FP64 reference.

miles' GDN input layernorm: ``[T, 1, H]`` BF16 hidden states at H = 2048 / 4096 / 5120 (Qwen3.5 / 3.6 widths the
kernels serve) and small, irregular and 16k-32k token counts, ``weight`` the w of ``(1 + w)`` in BF16 or FP32. ``y``
and ``dx`` at FlashInfer's tolerance; the FP32 weight gradient within its budget (rel-L2 <= 1e-4, elementwise
atol = rtol = 1e-4), a BF16 one widened by its rounding; ``det`` bitwise run to run; ``fast`` with ``y`` bitwise
``det``'s, ``dx`` reproducible and within one BF16 ulp of ``det``'s; unsupported hidden sizes refused at
construction; ``fast`` runs ``det`` under torch's deterministic algorithms. Needs SM100a / SM103a / SM107a; skips
elsewhere.
"""

import argparse
import logging

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

from torch.utils.checkpoint import checkpoint  # noqa: E402

from miles_ext_kernels import _variant, cake_native  # noqa: E402
from miles_ext_kernels.rmsnorm import ZeroCenteredRMSNorm, arguments, ops  # noqa: E402

if tuple(torch.cuda.get_device_capability()) not in ops.SUPPORTED_CAPABILITIES:
    pytest.skip("the Cake RMSNorm kernels need SM100a / SM103a / SM107a", allow_module_level=True)

EPS = 1e-6
DEVICE = torch.device("cuda", torch.cuda.current_device())
HIDDEN_SIZES = (2048, 4096, 5120)
TOKENS = (1, 7, 1000, 16385, 32769)
# FlashInfer's budgets: y / dx within atol = rtol = 1e-2 of the reference scaled by its RMS; the FP32 dw within
# rel-L2 1e-4 and elementwise atol = rtol = 1e-4 of the FP64 sum.
ATOL = RTOL = 1e-2
DW_REL_L2 = 1e-4
DW_ATOL = DW_RTOL = 1e-4
REPEATS = 2


def make(T, H, *, seed=0):
    """``(x, w, dy)``: BF16 ``[T, 1, H]`` hidden states and gradient of N(0, 1), BF16 ``w`` of N(0, 0.01)."""
    g = torch.Generator(device=DEVICE).manual_seed(seed * 1_000_003 + T * 131 + H)
    x = torch.randn(T, 1, H, device=DEVICE, generator=g).bfloat16()
    dy = torch.randn(T, 1, H, device=DEVICE, generator=g).bfloat16()
    w = (0.1 * torch.randn(H, device=DEVICE, generator=g)).bfloat16()
    return x, w, dy


def reference(x, w, dy):
    """FP64 ``(y, dx, dw)`` of ``x * rstd * (1 + w)`` at the kernels' FP32 effective weight ``1 + w``."""
    xd, gd = x.double(), dy.double()
    weight = (1.0 + w.float()).double()
    rstd = torch.rsqrt(xd.square().mean(-1, keepdim=True) + EPS)
    y = xd * rstd * weight
    a = gd * weight
    dx = rstd * a - xd * rstd**3 * (a * xd).mean(-1, keepdim=True)
    dw = (gd * xd * rstd).reshape(-1, x.shape[-1]).sum(0)
    return y, dx, dw


def assert_close_at_rms(actual, expected, label):
    assert actual.dtype == torch.bfloat16, label
    a, b = actual.double(), expected.double()
    assert torch.isfinite(a).all(), f"{label} has non-finite values"
    scale = b.square().mean().sqrt().item() or 1.0
    torch.testing.assert_close(a / scale, b / scale, atol=ATOL, rtol=RTOL, msg=label)


def assert_dw(dw, dw_ref, label):
    """FP32: FlashInfer's budget; BF16: the FP32 budget widened by the BF16 rounding."""
    diff = (dw.double() - dw_ref).abs()
    if dw.dtype == torch.float32:
        rel_l2 = ((dw.double() - dw_ref).norm() / dw_ref.norm()).item()
        assert rel_l2 <= DW_REL_L2, f"{label}: dw rel-L2 {rel_l2:.3e}"
        bound = DW_ATOL + DW_RTOL * dw_ref.abs()
    else:
        bound = 2**-8 * dw_ref.abs() + (1 + 2**-8) * (DW_ATOL + DW_RTOL * dw_ref.abs())
    bad = diff > bound
    assert not bad.any(), f"{label}: {int(bad.sum())} dw elements outside the budget"


def run(variant, x, w, dy, *, params_dtype=torch.bfloat16, use_checkpoint=False):
    """``(y, dx, dw)`` of one forward + backward of a :class:`ZeroCenteredRMSNorm` holding ``w``."""
    norm = ZeroCenteredRMSNorm(x.shape[-1], EPS, params_dtype=params_dtype, variant=variant)
    with torch.no_grad():
        norm.weight.copy_(w)
    xi = x.detach().requires_grad_()
    y = checkpoint(norm, xi, use_reentrant=False) if use_checkpoint else norm(xi)
    y.backward(dy)
    assert y.dtype == torch.bfloat16 and y.shape == x.shape
    assert xi.grad.dtype == torch.bfloat16 and norm.weight.grad.dtype == params_dtype
    return y.detach(), xi.grad, norm.weight.grad


@pytest.mark.parametrize("params_dtype", [torch.bfloat16, torch.float32], ids=["bf16-weight", "fp32-weight"])
@pytest.mark.parametrize("T", TOKENS)
@pytest.mark.parametrize("H", HIDDEN_SIZES)
def test_det_and_fast_match_fp64(H, T, params_dtype):
    x, w, dy = make(T, H)
    w = w.to(params_dtype)
    y_ref, dx_ref, dw_ref = reference(x, w, dy)
    det = run("det", x, w, dy, params_dtype=params_dtype)
    assert_close_at_rms(det[0], y_ref, "det y")
    assert_close_at_rms(det[1], dx_ref, "det dx")
    assert_dw(det[2], dw_ref, "det dw")
    for _ in range(REPEATS):
        again = run("det", x, w, dy, params_dtype=params_dtype)
        for name, a, b in zip(("y", "dx", "dw"), det, again, strict=True):
            assert torch.equal(a, b), f"det {name} differs between runs"
    fast = run("fast", x, w, dy, params_dtype=params_dtype)
    assert torch.equal(fast[0], det[0]), "fast y differs from det"
    ulp = 2**-7 * torch.maximum(det[1].double().abs(), det[1].double().square().mean().sqrt())
    assert ((fast[1].double() - det[1].double()).abs() <= ulp).all(), "fast dx beyond one BF16 ulp of det"
    assert_dw(fast[2], dw_ref, "fast dw")
    for _ in range(REPEATS):
        assert torch.equal(run("fast", x, w, dy, params_dtype=params_dtype)[1], fast[1]), "fast dx differs"


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_strided_rows_match_contiguous_rows(variant):
    T, H = 1000, 2048
    x, w, dy = make(T, H)
    wide = torch.zeros(T, 1, H + 64, dtype=torch.bfloat16, device=DEVICE)
    wide[..., :H] = x
    strided = wide[..., :H]
    assert not strided.is_contiguous()
    plain, from_strided = run(variant, x, w, dy), run(variant, strided, w, dy)
    assert torch.equal(plain[0], from_strided[0]) and torch.equal(plain[1], from_strided[1])


def test_non_reentrant_checkpoint_matches_the_plain_call():
    x, w, dy = make(16385, 4096)
    plain = run("det", x, w, dy)
    checkpointed = run("det", x, w, dy, use_checkpoint=True)
    for name, a, b in zip(("y", "dx", "dw"), plain, checkpointed, strict=True):
        assert torch.equal(a, b), f"checkpointed {name} differs"


def test_the_module_has_the_te_parameter_surface():
    norm = ZeroCenteredRMSNorm(2048, EPS)
    assert list(norm.state_dict()) == ["weight"]
    assert norm.weight.dtype == torch.bfloat16 and norm.weight.device == DEVICE
    assert not norm.weight.any()
    x = make(7, 2048)[0]
    assert torch.equal(norm(x), ops.zero_centered_rmsnorm(x, torch.zeros(2048, device=DEVICE), EPS))


@pytest.mark.parametrize("hidden", [1024, 2560, 3072])
def test_unsupported_hidden_sizes_are_refused_at_construction(hidden):
    with pytest.raises(ValueError, match=f"not {hidden}"):
        ZeroCenteredRMSNorm(hidden, EPS)


def test_the_supported_hidden_sizes_include_the_gdn_widths():
    arch = cake_native.ARCH_BY_CAPABILITY[tuple(torch.cuda.get_device_capability())]
    assert set(HIDDEN_SIZES) <= set(ops.supported_hidden_sizes(arch))
    assert not {1024, 2560} & set(ops.supported_hidden_sizes(arch))


def test_the_device_needs_its_architectures_toolkit(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    with pytest.raises(ValueError, match=r"SM100a / SM103a / SM107a"):
        ops.check_supported(2048)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 7))
    monkeypatch.setattr(cake_native, "MIN_CUDA", {**cake_native.MIN_CUDA, "sm_107a": (99, 0)})
    with pytest.raises(RuntimeError, match=r"RMSNorm kernels for sm_107a needs CUDA 99\.0 or newer"):
        ops.check_supported(2048)


@pytest.mark.parametrize("variant", ["det", "fast"])
@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
def test_construction_builds_every_program_of_the_routes(variant, hidden):
    ZeroCenteredRMSNorm(hidden, EPS, variant=variant)
    det, fast = ops._packages()
    package = det if variant == "det" else fast
    jit = package.cake_jit
    arch = jit.arch_for_capability(torch.cuda.get_device_capability())
    names = {
        jit.library_name(module, arch)
        for kind in ("fwd", "bwd")
        for module in jit.route_modules(arch, hidden, kind, False, "f32")
    }
    missing = sorted(name for name in names if not package.cake_build.library_path(name).is_file())
    assert names and not missing, missing


def test_fast_runs_det_under_torch_deterministic_algorithms(monkeypatch, caplog):
    monkeypatch.setattr(_variant, "_warned", set())
    x, w, dy = make(16385, 2048)
    det = run("det", x, w, dy)
    previous = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        with caplog.at_level(logging.WARNING, logger="miles_ext_kernels"):
            fallback = [run("fast", x, w, dy) for _ in range(2)]
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
    for result in fallback:
        for name, a, b in zip(("y", "dx", "dw"), det, result, strict=True):
            assert torch.equal(a, b), f"fallback {name} differs from det"
    warnings = [r for r in caplog.records if "rmsnorm" in r.getMessage() and "running det" in r.getMessage()]
    assert len(warnings) == 1, [r.getMessage() for r in caplog.records]


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_configure_selects_the_variant_of_a_call(monkeypatch, variant):
    monkeypatch.setattr(ops, "VARIANT", ops.VARIANT)
    _, fast = ops._packages()
    original = fast.cake_rmsnorm
    calls = []

    def record(*args, **kwargs):
        calls.append(kwargs["deterministic"])
        return original(*args, **kwargs)

    monkeypatch.setattr(fast, "cake_rmsnorm", record)
    arguments.configure(argparse.Namespace(cake_rmsnorm_variant=variant))
    x = make(7, 2048)[0]

    ZeroCenteredRMSNorm(2048, EPS)(x)
    assert calls == [variant == "det"]
