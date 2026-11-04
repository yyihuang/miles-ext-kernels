"""The Cake SiLU-gated RMSNorm kernels (``miles_ext_kernels.gated_rmsnorm``) against an FP64 eager reference.

miles' GDN output norm: ``x`` the recurrence output ``[T, H, 128]``, the gate ``z`` a column slice of the fused input
projection, at H = 12 / 16 / 32 / 48 local value heads and small, irregular and ~32k token counts. ``y``, ``dx`` and
``dz`` within one BF16 ulp of FP64; the weight gradient within the deterministic package's bound; ``det`` bitwise run
to run; ``fast`` with ``y``, ``dx`` and ``dz`` bitwise ``det``'s; non-reentrant checkpointing and gradient
accumulation; ``fast`` runs ``det`` under torch's deterministic algorithms. Needs SM100a / SM103a; skips elsewhere.
"""

import argparse
import logging

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

from torch.utils.checkpoint import checkpoint  # noqa: E402

from miles_ext_kernels import _variant  # noqa: E402
from miles_ext_kernels.gated_rmsnorm import arguments, ops  # noqa: E402

if tuple(torch.cuda.get_device_capability()) not in ops.SUPPORTED_CAPABILITIES:
    pytest.skip("the Cake gated RMSNorm kernels need SM100a / SM103a", allow_module_level=True)

D = 128
EPS = 1e-6
DEVICE = torch.device("cuda", torch.cuda.current_device())
BF16_RTOL = 2.0**-7  # one BF16 ulp relative to the value
DW_MAG_TOL = 2.0**-20  # the FP32 accumulation term of the weight-gradient bound, relative to sum |terms|
REPEATS = 2
TOKENS = (1, 7, 1031, 32771)
HEADS = (12, 16, 32, 48)


def make(T, H, *, seed=0):
    """``(x, z, w, dy)``: ``x`` a contiguous ``[T, H, 128]`` recurrence output, ``z`` the gate columns of a wider
    projection (read in place), ``dy`` contiguous; all BF16 of N(0, 1), ``w`` of 1 + N(0, 0.25)."""
    g = torch.Generator(device=DEVICE).manual_seed(seed * 1_000_003 + T * 131 + H)

    def randn(*shape):
        return torch.randn(*shape, device=DEVICE, generator=g).bfloat16()

    x = randn(1, T, H, D).flatten(0, 1)
    projection = randn(1, T, 3 * H * D + 2 * H + 8)  # q/k/v + z + b + a + padding, as miles' fused GEMM
    z = projection[..., H * D : 2 * H * D].unflatten(-1, (H, D)).flatten(0, 1)
    dy = randn(T, H, D)
    w = (1.0 + 0.5 * torch.randn(D, device=DEVICE, generator=g)).bfloat16()
    return x, z, w, dy


def fp64_reference(dy, x, z, w, eps):
    """FP64 forward and closed-form backward from the BF16 inputs: ``(y, dx, dz, dw, sum |terms| per column)``."""
    xd, zd, wd, gd = x.double(), z.double(), w.double(), dy.double()
    rstd = 1.0 / torch.sqrt(xd.square().mean(dim=-1, keepdim=True) + eps)
    n = xd * rstd
    s = torch.sigmoid(zd)
    silu = zd * s
    y = n * wd * silu
    a = gd * silu
    dz = gd * n * wd * s * (1.0 + zd * (1.0 - s))
    dn = a * wd
    dx = rstd * dn - xd * rstd**3 * (dn * xd).mean(dim=-1, keepdim=True)
    terms = (a * n).reshape(-1, D)
    return y, dx, dz, terms.sum(dim=0), terms.abs().sum(dim=0)


def assert_bf16_close(got, want, *, what):
    """``got`` within one BF16 ulp of ``want`` plus a floor of 1/64 ulp of its largest magnitude (cancellation)."""
    want = want.double()
    floor = want.abs().max().item() * BF16_RTOL / 64
    err = (got.double() - want).abs()
    bad = err > want.abs() * BF16_RTOL + floor
    assert not bad.any(), (
        f"{what}: {int(bad.sum())} of {bad.numel()} beyond one BF16 ulp; worst {err.max().item():.3e}"
    )


def assert_dw_within(dw, want, mag, *, what):
    """``|dw - want| <= |want| * 2^-7 + sum|terms| * 2^-20`` per column."""
    err = (dw.double() - want).abs()
    bad = err > want.abs() * BF16_RTOL + mag * DW_MAG_TOL
    assert not bad.any(), f"{what}: {int(bad.sum())} of {bad.numel()} columns beyond the bound"


def run(variant, x, z, w, dy, *, use_checkpoint=False):
    """``(y, dx, dz, dw)`` of one autograd forward + backward of ``variant``."""
    xi, zi, wi = (t.detach().requires_grad_() for t in (x, z, w))
    if use_checkpoint:
        y = checkpoint(ops.rmsnorm_silu_gate, xi, zi, wi, EPS, variant=variant, use_reentrant=False)
    else:
        y = ops.rmsnorm_silu_gate(xi, zi, wi, EPS, variant=variant)
    y.backward(dy)
    assert y.dtype == torch.bfloat16 and y.shape == x.shape
    assert xi.grad.dtype == zi.grad.dtype == wi.grad.dtype == torch.bfloat16 and wi.grad.shape == (D,)
    return y.detach(), xi.grad, zi.grad, wi.grad


@pytest.mark.parametrize("H", HEADS)
@pytest.mark.parametrize("T", TOKENS)
def test_det_and_fast_match_fp64(T, H):
    x, z, w, dy = make(T, H)
    y64, dx64, dz64, dw64, mag = fp64_reference(dy, x, z, w, EPS)
    det = run("det", x, z, w, dy)
    assert_bf16_close(det[0], y64, what="det y")
    assert_bf16_close(det[1], dx64, what="det dx")
    assert_bf16_close(det[2], dz64, what="det dz")
    assert_dw_within(det[3], dw64, mag, what="det dw")
    for _ in range(REPEATS):
        again = run("det", x, z, w, dy)
        for name, a, b in zip(("y", "dx", "dz", "dw"), det, again, strict=True):
            assert torch.equal(a, b), f"det {name} differs between runs"
    for _ in range(REPEATS):
        fast = run("fast", x, z, w, dy)
        for name, a, b in zip(("y", "dx", "dz"), det, fast, strict=True):
            assert torch.equal(a, b), f"fast {name} differs from det"
        assert_dw_within(fast[3], dw64, mag, what="fast dw")


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_non_reentrant_checkpoint_matches_the_plain_call(variant):
    x, z, w, dy = make(1031, 32)
    plain = run(variant, x, z, w, dy)
    checkpointed = run(variant, x, z, w, dy, use_checkpoint=True)
    names = ("y", "dx", "dz", "dw") if variant == "det" else ("y", "dx", "dz")
    for name, a, b in zip(names, plain, checkpointed):
        assert torch.equal(a, b), f"{variant} checkpointed {name} differs"
    if variant == "fast":
        y64, dx64, dz64, dw64, mag = fp64_reference(dy, x, z, w, EPS)
        assert_dw_within(checkpointed[3], dw64, mag, what="fast checkpointed dw")


def test_gradients_accumulate_across_micro_batches():
    x, z, w, dy1 = make(1031, 16, seed=1)
    dy2 = make(1031, 16, seed=2)[3]
    xi, zi, wi = (t.detach().requires_grad_() for t in (x, z, w))
    for dy in (dy1, dy2):
        ops.rmsnorm_silu_gate(xi, zi, wi, EPS, variant="det").backward(dy)
    first, second = run("det", x, z, w, dy1), run("det", x, z, w, dy2)

    assert torch.equal(wi.grad, first[3] + second[3])
    assert torch.equal(xi.grad, first[1] + second[1])
    assert torch.equal(zi.grad, first[2] + second[2])


def test_fast_runs_det_under_torch_deterministic_algorithms(monkeypatch, caplog):
    monkeypatch.setattr(_variant, "_warned", set())
    x, z, w, dy = make(32771, 16)
    det = run("det", x, z, w, dy)
    previous = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        with caplog.at_level(logging.WARNING, logger="miles_ext_kernels"):
            fallback = [run("fast", x, z, w, dy) for _ in range(2)]
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
    for result in fallback:
        for name, a, b in zip(("y", "dx", "dz", "dw"), det, result, strict=True):
            assert torch.equal(a, b), f"fallback {name} differs from det"
    warnings = [r for r in caplog.records if "gated_rmsnorm" in r.getMessage() and "running det" in r.getMessage()]
    assert len(warnings) == 1, [r.getMessage() for r in caplog.records]


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_configure_selects_the_variant_of_a_call(monkeypatch, variant):
    monkeypatch.setattr(ops, "VARIANT", ops.VARIANT)
    calls = []
    for package_variant, name in (("det", "cake_rmsnorm_silu_gate_det"), ("fast", "cake_rmsnorm_silu_gate")):
        package = ops._package(package_variant)
        original = getattr(package, name)

        def record(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(package, name, record)
    arguments.configure(argparse.Namespace(cake_gated_rmsnorm_variant=variant))
    x, z, w, _ = make(7, 12)

    ops.rmsnorm_silu_gate(x, z, w, EPS)
    assert calls == ["cake_rmsnorm_silu_gate_det" if variant == "det" else "cake_rmsnorm_silu_gate"]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (
            dict(head_dim=128, activation="sigmoid", weight_dtype=torch.bfloat16),
            "gates its output norm with 'sigmoid'",
        ),
        (dict(head_dim=64, activation="silu", weight_dtype=torch.bfloat16), "rows of 128"),
        (dict(head_dim=128, activation="silu", weight_dtype=torch.bfloat16, has_bias=True), "no bias"),
        (dict(head_dim=128, activation="silu", weight_dtype=torch.float32), "BF16 weight"),
    ],
    ids=["sigmoid-gate", "head-dim-64", "bias", "fp32-weight"],
)
def test_unsupported_norms_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ops.check_supported(**kwargs)


def test_the_gdn_norm_is_supported():
    ops.check_supported(128, "silu", torch.bfloat16)


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_prebuild_builds_every_library(variant):
    ops.prebuild(variant=variant)
    package = ops._package(variant)
    jit = package.cake_jit
    arch = jit.arch_for_capability(torch.cuda.get_device_capability())
    names = {jit.library_name(module, arch) for module in jit.ROUTES[arch].values()}
    missing = sorted(name for name in names if not package.cake_build.library_path(name).is_file())
    assert names and not missing, missing
