"""The Cake short-conv kernels (``miles_ext_kernels.conv1d``) against a document-local FP64 reference.

Both variants at FlashInfer's edge cases, at miles' TP-local channel counts (Qwen3.5 / Qwen3.6 GDN, Kimi-K3 KDA,
GLM-5.3-Flash) on ~32k packed tokens with documents shorter than the kernel, with FP32 and BF16 taps, and at
FlashInfer's S x C > 2^31 case. ``y`` and ``dx`` within one BF16 ulp of the reference; ``dweight`` within the
deterministic package's budgets; ``det`` bitwise run to run; ``fast`` with ``y`` and ``dx`` bitwise ``det``'s and
``dweight`` within budget of ``det``; ``fast`` runs ``det`` under torch's deterministic algorithms. Needs SM100a /
SM103a; skips elsewhere.
"""

import argparse
import logging

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

from miles_ext_kernels import _variant  # noqa: E402
from miles_ext_kernels.conv1d import arguments, ops  # noqa: E402

if tuple(torch.cuda.get_device_capability()) not in ops.SUPPORTED_CAPABILITIES:
    pytest.skip("the Cake short-conv kernels need SM100a / SM103a", allow_module_level=True)

WIDTH = 4
F32, BF16 = torch.float32, torch.bfloat16
DEVICE = torch.device("cuda", torch.cuda.current_device())
# FlashInfer's contract: y / dx within one BF16 ulp of the reference; dweight within atol 5e-2 + rtol 1e-4 of the FP64
# sum under the bf16 SiLU' policy and atol 1e-3 + rtol 2e-6 under the fp32 policy (a BF16 dweight widened by its own
# rounding).
Y_ATOL, Y_RTOL = 1e-3, 2**-7
DW_ATOL, DW_RTOL = 5e-2, 1e-4
DW_FP32G_ATOL, DW_FP32G_RTOL = 1e-3, 2e-6
BLOCK = 8192  # the reference's token block
COMPARE_CHUNK = 2**26
PERF_FRACTIONS = (0.37, 0.29, 0.21)
SEED = 2033
REPEATS = 2

# FlashInfer's edge cases (S, C, documents): documents of 1-3 tokens and empty ones, sub-tile boundaries, the perf
# packing, multi-tile channel counts and the fallback channel splits.
CASES = (
    (5, 136, (2, 0, 3)),
    (63, 384, (1, 2, 3, 57)),
    (1000, 512, "perf"),
    (4097, 1280, (1000, 0, 3, 3094)),
    (4097, 2560, (1000, 0, 3, 3094)),
    (1000, 1600, "perf"),
)
# miles' per-rank conv channels: Qwen3.5 / 3.6 GDN (2560 ... 10240), Kimi-K3 KDA (12288 / TP), GLM-5.3-Flash (24576).
MILES_CHANNELS = (2560, 3072, 4096, 8192, 10240, 24576)
# ~32k packed tokens: documents shorter than the kernel (1, 2, 3 tokens), an empty one and two long ones.
MILES_DOCS = (1, 2, 3, 0, 9000, 3, 23762)
LARGE_CASE = (262401, 8192, "perf")  # S x C > 2^31
LARGE_CASE_FREE_BYTES = 40 * 2**30


def doc_lengths(S, docs):
    if docs is None:
        return (S,)
    if docs == "perf":
        lens = []
        for fraction in PERF_FRACTIONS:
            n = int(fraction * S)
            lens.append(n - 1 if n % 64 == 0 else n)
        return (*lens, S - sum(lens))
    assert sum(docs) == S
    return tuple(docs)


def _inputs(S, C, docs, *, taps, key=0):
    """``(x, w, cu, dy)``: BF16 ``x`` / ``dy`` of N(0, 1), taps of N(0, 0.25) in ``taps``."""
    generator = torch.Generator(device=DEVICE).manual_seed((SEED * 1_000_003 + S * 7919 + C * 31 + key) % (2**31 - 1))
    x = torch.randn(1, S, C, device=DEVICE, generator=generator).bfloat16()
    w = (0.5 * torch.randn(C, WIDTH, device=DEVICE, generator=generator)).to(taps)
    dy = torch.randn(1, S, C, device=DEVICE, generator=generator).bfloat16()
    lens = doc_lengths(S, docs)
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device=DEVICE)
    return x, w, cu, dy


def _shift(v, ids, d):
    """``out[t] = v[t - d]`` (``d > 0``) or ``v[t + |d|]`` (``d < 0``) within token ``t``'s document, else 0."""
    out = torch.zeros_like(v)
    if d == 0:
        return v.clone()
    n = abs(d)
    if n >= v.shape[0]:
        return out
    if d > 0:
        out[n:] = torch.where((ids[n:] == ids[:-n]).unsqueeze(1), v[:-n], 0.0)
    else:
        out[:-n] = torch.where((ids[:-n] == ids[n:]).unsqueeze(1), v[n:], 0.0)
    return out


def _preactivation(x, ids, w):
    pre = torch.zeros_like(x)
    for k in range(WIDTH):
        pre = pre + w[:, k].to(x.dtype) * _shift(x, ids, WIDTH - 1 - k)
    return pre


def reference(x, w, cu, dy, policy="bf16"):
    """``(y, dx, dweight)``: BF16 ``y`` / ``dx`` of the FP32 (fp32 policy: FP64) math, FP64 ``dweight``.

    The bf16 policy evaluates SiLU' at the BF16-rounded FP32 preactivation, as the kernels do (FlashInfer's
    reference); the fp32 policy evaluates everything in FP64 and rounds once."""
    x, dy = x[0], dy[0]
    S, C = x.shape
    ids = torch.bucketize(torch.arange(S, device=x.device), cu[1:].long(), right=True)
    y = torch.empty_like(x)
    dx = torch.empty_like(x)
    dw = torch.zeros(C, WIDTH, dtype=torch.float64, device=x.device)
    for a in range(0, S, BLOCK):
        b = min(a + BLOCK, S)
        lo, hi = max(a - (WIDTH - 1), 0), min(b + (WIDTH - 1), S)
        win = ids[lo:hi]
        rows = slice(a - lo, b - lo)
        pre32 = _preactivation(x[lo:hi].float(), win, w)
        y[a:b] = (pre32 * torch.sigmoid(pre32)).bfloat16()[rows]
        if policy == "fp32":
            xf = x[lo:hi].double()
            p = _preactivation(xf, win, w)
        else:
            xf = x[lo:hi].float()
            p = pre32.bfloat16().float()
        s = torch.sigmoid(p)
        dpre = dy[lo:hi].to(xf.dtype) * (s * (1.0 + p * (1.0 - s)))
        dxw = torch.zeros_like(dpre)
        for k in range(WIDTH):
            dxw = dxw + w[:, k].to(xf.dtype) * _shift(dpre, win, -(WIDTH - 1 - k))
        dx[a:b] = dxw[rows].bfloat16()
        for k in range(WIDTH):
            dw[:, k] += (dpre[rows].double() * _shift(xf, win, WIDTH - 1 - k)[rows].double()).sum(dim=0)
    return y.unsqueeze(0), dx.unsqueeze(0), dw


def _grad_tolerance(policy, dtype):
    atol, rtol = (DW_ATOL, DW_RTOL) if policy == "bf16" else (DW_FP32G_ATOL, DW_FP32G_RTOL)
    if dtype == BF16:
        return atol * (1 + 2**-8), 2**-7
    return atol, rtol


def assert_within(actual, expected, *, atol, rtol, label):
    """No element outside ``atol + rtol * |expected|``."""
    assert tuple(actual.shape) == tuple(expected.shape), label
    flat_actual, flat_expected = actual.reshape(-1), expected.reshape(-1)
    count, first = 0, None
    for start in range(0, flat_actual.numel(), COMPARE_CHUNK):
        a = flat_actual[start : start + COMPARE_CHUNK].double()
        b = flat_expected[start : start + COMPARE_CHUNK].double()
        assert torch.isfinite(a).all(), f"{label} has non-finite values"
        outside = (a - b).abs() > atol + rtol * b.abs()
        chunk = int(outside.sum())
        if chunk and first is None:
            index = int(outside.nonzero()[0])
            first = (start + index, a[index].item(), b[index].item())
        count += chunk
    assert (
        not count
    ), f"{label}: {count} of {flat_actual.numel()} elements outside atol={atol} rtol={rtol}; first {first}"


def run(variant, x, w, cu, dy, policy="bf16"):
    """``(y, dx, dweight)`` of one autograd forward + backward of ``variant``."""
    xi = x.detach().requires_grad_()  # new leaves sharing the inputs' storage
    wi = w.detach().requires_grad_()
    y = ops.causal_conv1d(xi, wi, cu, variant=variant, silu_grad_preactivation=policy)
    y.backward(dy)
    assert y.dtype == BF16 and y.shape == x.shape
    assert xi.grad.dtype == BF16 and wi.grad.dtype == w.dtype and wi.grad.shape == w.shape
    return y.detach(), xi.grad, wi.grad


def check(x, w, cu, dy, *, policy="bf16", label):
    """``det`` against the reference and bitwise run to run; ``fast`` with ``y`` / ``dx`` bitwise ``det``'s and
    ``dweight`` within budget of the reference and of ``det``."""
    y_ref, dx_ref, dw_ref = reference(x, w, cu, dy, policy)
    atol, rtol = _grad_tolerance(policy, w.dtype)
    det = run("det", x, w, cu, dy, policy)
    assert_within(det[0], y_ref, atol=Y_ATOL, rtol=Y_RTOL, label=f"det y {label}")
    assert_within(det[1], dx_ref, atol=Y_ATOL, rtol=Y_RTOL, label=f"det dx {label}")
    assert_within(det[2].float(), dw_ref.float(), atol=atol, rtol=rtol, label=f"det dweight {label}")
    for _ in range(REPEATS):
        again = run("det", x, w, cu, dy, policy)
        for name, a, b in zip(("y", "dx", "dweight"), det, again, strict=True):
            assert torch.equal(a, b), f"det {name} differs between runs ({label})"
    fast = run("fast", x, w, cu, dy, policy)
    assert torch.equal(fast[0], det[0]), f"fast y differs from det ({label})"
    assert torch.equal(fast[1], det[1]), f"fast dx differs from det ({label})"
    assert_within(fast[2].float(), dw_ref.float(), atol=atol, rtol=rtol, label=f"fast dweight {label}")
    assert_within(fast[2].float(), det[2].float(), atol=atol, rtol=rtol, label=f"fast dweight vs det {label}")


def _case_id(case):
    S, C, docs = case
    return f"S{S}-C{C}-{docs if isinstance(docs, str) else len(doc_lengths(S, docs))}docs"


@pytest.mark.parametrize("taps", [F32, BF16], ids=["fp32-taps", "bf16-taps"])
@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_edge_cases_match_the_reference(case, taps):
    S, C, docs = case
    check(*_inputs(S, C, docs, taps=taps), label=f"{case} {taps}")


@pytest.mark.parametrize("taps", [F32, BF16], ids=["fp32-taps", "bf16-taps"])
@pytest.mark.parametrize("channels", MILES_CHANNELS)
def test_miles_shapes_match_the_reference(channels, taps):
    S = sum(MILES_DOCS)
    check(*_inputs(S, channels, MILES_DOCS, taps=taps), label=f"S={S} C={channels} {taps}")


@pytest.mark.parametrize("taps", [F32, BF16], ids=["fp32-taps", "bf16-taps"])
def test_the_fp32_silu_policy_matches_the_reference(taps):
    check(*_inputs(4097, 1280, (1000, 0, 3, 3094), taps=taps), policy="fp32", label=f"fp32 policy {taps}")


def test_more_than_2_to_the_31_elements():
    S, C, docs = LARGE_CASE
    free, _ = torch.cuda.mem_get_info(DEVICE)
    if free < LARGE_CASE_FREE_BYTES:
        pytest.skip(f"needs {LARGE_CASE_FREE_BYTES >> 30} GiB free, {free >> 30} GiB available")
    check(*_inputs(S, C, docs, taps=F32), label=str(LARGE_CASE))


def test_a_strided_input_matches_the_contiguous_one():
    # miles' GDN q/k/v is a column slice of the fused projection
    S, C = 1000, 1536
    x, w, cu, dy = _inputs(S, C, "perf", taps=BF16)
    wide = torch.randn(1, S, C + 520, device=DEVICE).bfloat16()
    wide[..., :C] = x
    strided = wide[..., :C]
    assert not strided.is_contiguous()
    for variant in ("det", "fast"):
        assert torch.equal(ops.causal_conv1d(strided, w, cu, variant=variant), run(variant, x, w, cu, dy)[0])


def test_no_cu_seqlens_is_one_document():
    x, w, cu, dy = _inputs(4097, 1024, None, taps=F32)
    for variant in ("det", "fast"):
        assert torch.equal(ops.causal_conv1d(x, w, variant=variant), ops.causal_conv1d(x, w, cu, variant=variant))


def test_a_batch_of_several_sequences_is_rejected():
    x = torch.zeros(2, 64, 256, dtype=BF16, device=DEVICE)
    w = torch.zeros(256, WIDTH, device=DEVICE)
    with pytest.raises(ValueError, match="one packed batch"):
        ops.causal_conv1d(x, w)


def test_fast_runs_det_under_torch_deterministic_algorithms(monkeypatch, caplog):
    monkeypatch.setattr(_variant, "_warned", set())
    x, w, cu, dy = _inputs(4097, 2560, (1000, 0, 3, 3094), taps=BF16)
    det = run("det", x, w, cu, dy)
    previous = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        with caplog.at_level(logging.WARNING, logger="miles_ext_kernels"):
            fallback = [run("fast", x, w, cu, dy) for _ in range(2)]
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
    for result in fallback:
        for name, a, b in zip(("y", "dx", "dweight"), det, result, strict=True):
            assert torch.equal(a, b), f"fallback {name} differs from det"
    warnings = [r for r in caplog.records if "conv1d" in r.getMessage() and "running det" in r.getMessage()]
    assert len(warnings) == 1, [r.getMessage() for r in caplog.records]


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_configure_selects_the_variant_of_a_call(monkeypatch, variant):
    monkeypatch.setattr(ops, "VARIANT", ops.VARIANT)
    monkeypatch.setattr(ops, "SILU_GRAD_PREACTIVATION", ops.SILU_GRAD_PREACTIVATION)
    calls = []
    for name in ("cake_causal_conv1d_det", "cake_causal_conv1d"):
        package = ops._package("det" if name.endswith("_det") else "fast")
        original = getattr(package, name)

        def record(*args, _name=name, _original=original, **kwargs):
            calls.append((_name, kwargs.get("silu_grad_preactivation")))
            return _original(*args, **kwargs)

        monkeypatch.setattr(package, name, record)
    arguments.configure(argparse.Namespace(cake_conv1d_variant=variant, cake_conv1d_silu_grad_preactivation="fp32"))
    x, w, cu, _ = _inputs(1000, 512, "perf", taps=F32)

    ops.causal_conv1d(x, w, cu)
    assert calls == [("cake_causal_conv1d_det" if variant == "det" else "cake_causal_conv1d", "fp32")]


@pytest.mark.parametrize(
    "channels, kernel_size, activation, match",
    [
        (4096, 3, "silu", "kernel size 4"),
        (4096, 4, "sigmoid", "applies SiLU"),
        (4100, 4, "silu", "multiple of 8"),
    ],
)
def test_unsupported_convolutions_are_rejected(channels, kernel_size, activation, match):
    with pytest.raises(ValueError, match=match):
        ops.check_supported(channels, kernel_size, activation)


def test_miles_convolutions_are_supported():
    for channels in MILES_CHANNELS:
        ops.check_supported(channels, 4, "silu")


@pytest.mark.parametrize("variant", ["det", "fast"])
@pytest.mark.parametrize("taps", [F32, BF16], ids=["fp32-taps", "bf16-taps"])
@pytest.mark.parametrize("policy", ["bf16", "fp32"])
def test_prebuild_builds_every_program_a_call_selects(variant, taps, policy):
    ops.prebuild(taps, variant=variant, silu_grad_preactivation=policy)
    package = ops._package(variant)
    jit = package.cake_jit
    arch = jit.arch_for_capability(torch.cuda.get_device_capability())
    bf16 = taps == BF16
    routes = (
        jit.forward_route(bias=False, bf16_taps=bf16),
        jit.backward_route(bias=False, fp32_grad=policy == "fp32", bf16_taps_or_dweight=bf16),
    )
    missing = []
    for route in routes:
        for C in (*MILES_CHANNELS, 136, 1600):
            for S in (5, sum(MILES_DOCS), LARGE_CASE[0]):
                for module in jit.selected_modules(arch, route, C=C, S=S):
                    names = [jit.library_name(module, arch)]
                    if jit.MODULES[module].get("reduce"):
                        names.append(jit.library_name(jit.MODULES[module]["reduce"], arch))
                    missing += [n for n in names if not package.cake_build.library_path(n).is_file()]
    assert not missing, sorted(set(missing))
