"""The Cake LM-head log-probabilities (``det`` and ``fast``) against a PyTorch reference, at the hidden / vocabulary
sizes of the models miles trains with ``--log-probs-backend cake`` (TP 1, so the whole vocabulary per GPU). Needs
Blackwell (SM100a / SM103a); skips elsewhere.

Accuracy gate (FlashInfer's own, from its chunked LM-head tests): against a high-precision oracle, each error of a
variant is at most 1.05x the error of the unchunked PyTorch path at the same rounding boundaries (BF16 ``z = X @ W^T``,
FP32 ``log_softmax(z / tau)`` gathered at the labels, autograd ``dX`` / ``dW`` in BF16) plus a small floor. The oracle
evaluates the same BF16 input values in FP32 with TF32 off and no intermediate rounding: its own error (about 1e-6
relative) is three orders below the BF16 errors gated (about 1e-3), and FP64 GEMMs are too slow on GB300 for this
matrix.

Bitwise properties: ``det`` repeats bit for bit; ``fast`` gives ``det``'s ``logp`` and, without zero-gradient rows,
``det``'s ``dW``; a no-grad forward gives the training forward's ``logp``; a frozen input leaves the other gradient's
bits unchanged (``det``); a caller-given valid-row count gives the counted call's bits.

Every kernel library is built before the tests (``ops.prebuild``); set ``TVM_FFI_CACHE_DIR`` to keep them.
"""

import contextlib
import logging
import os
from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

from miles_ext_kernels import _variant  # noqa: E402
from miles_ext_kernels.lm_head import ops  # noqa: E402

if torch.cuda.get_device_capability() not in ops.SUPPORTED_CAPABILITIES:
    pytest.skip("the Cake LM-head kernels need SM100a / SM103a", allow_module_level=True)

CUDA = torch.device("cuda")
IGNORE_INDEX = -100
SEED = 20261104
VARIANTS = ("det", "fast")
# (H, V) of the models the cake log-prob path serves (H a multiple of 512, V of 256)
SHAPES = [(4096, 129280), (5120, 248320), (4096, 154880), (6144, 154880), (7168, 129280)]
# one row, one chunk short of full, two chunks with a one-row tail, four chunks with a ragged tail
ROWS = [1, 4095, 4097, 16231]
TEMPERATURES = [None, 0.7]
# (H, V, T) of the CUDA 13.0 fault reports on the sm_100a dW / dX GEMMs, about a third of the labels ignored
ISSUE_SHAPES = [
    (6144, 154880, 16384),
    (4096, 154880, 16384),
    (6144, 154880, 16231),
    (4096, 154880, 16231),
    (6144, 154880, 4096),
    (5120, 248320, 4096),
    (5120, 248320, 16231),
    (5120, 248320, 16384),
]
GATE_MARGIN = 1.05
GATE_FLOOR = dict(loss_rel=1e-6, logp_max_abs=1e-5, dX_rel_l2=1e-3, dW_rel_l2=1e-3)
_ORACLE_ROWS = 1024  # rows per oracle step: one [rows, V] FP32 logits block
_NORM_ROWS = 8192  # rows per step of an error norm


@pytest.fixture(scope="module", autouse=True)
def built():
    """Every library of both variants (with the temperature forms) built up front, in parallel."""
    workers = min(16, len(os.sched_getaffinity(0)))
    for variant in VARIANTS:
        ops.prebuild(variant=variant, temperature=True, max_workers=workers)
        assert ops.prebuild(variant=variant, temperature=True) == []  # all cached now


@pytest.fixture(autouse=True)
def _fresh_bindings():
    """Each test starts without remembered bindings (their workspaces) from earlier shapes."""
    yield
    for variant in VARIANTS:
        entry = ops._package(variant).entry
        for module in (entry, entry._main):
            module.BINDING_CACHE.clear()
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
_WEIGHTS: dict[tuple[int, int], torch.Tensor] = {}


def _weight(H: int, V: int) -> torch.Tensor:
    """``W [V, H]`` BF16 ~ N(0, 1 / H), one per shape for the module."""
    if (H, V) not in _WEIGHTS:
        gen = torch.Generator(device=CUDA)
        gen.manual_seed(SEED + 7 * H + V)
        W = torch.empty((V, H), dtype=torch.bfloat16, device=CUDA)
        for r0 in range(0, V, 32768):
            r1 = min(V, r0 + 32768)
            W[r0:r1] = (torch.randn((r1 - r0, H), generator=gen, device=CUDA) * H**-0.5).to(torch.bfloat16)
        _WEIGHTS[(H, V)] = W
    return _WEIGHTS[(H, V)]


@dataclass
class Problem:
    X: torch.Tensor  # BF16 [T, H]
    W: torch.Tensor  # BF16 [V, H]
    labels: torch.Tensor  # int64 [T], -100 ignored
    dlogp: torch.Tensor  # FP32 [T] incoming gradient, 0 on ignored rows
    temperature: float | None

    @property
    def valid(self) -> torch.Tensor:
        return self.labels >= 0

    def tau_rows(self) -> torch.Tensor | None:
        if self.temperature is None:
            return None
        return torch.full((self.X.shape[0],), self.temperature, dtype=torch.float32, device=CUDA)


def _problem(H, V, T, *, temperature=None, ignore_frac=0.05, zero_dlogp_frac=0.0, seed=SEED) -> Problem:
    gen = torch.Generator(device=CUDA)
    gen.manual_seed(seed + T)
    X = (torch.randn((T, H), generator=gen, device=CUDA) * 0.5).to(torch.bfloat16)
    labels = torch.randint(0, V, (T,), generator=gen, device=CUDA)
    ignored = round(ignore_frac * T)
    if ignored:
        labels[torch.randperm(T, generator=gen, device=CUDA)[:ignored]] = IGNORE_INDEX
    dlogp = torch.randn((T,), generator=gen, device=CUDA) * 0.05
    if zero_dlogp_frac:
        dlogp[torch.rand((T,), generator=gen, device=CUDA) < zero_dlogp_frac] = 0.0
    dlogp = torch.where(labels >= 0, dlogp, torch.zeros_like(dlogp))
    return Problem(X, _weight(H, V), labels, dlogp.contiguous(), temperature)


# --------------------------------------------------------------------------- #
# references
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _fp32_gemms():
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


@contextlib.contextmanager
def _unsplit_gemms():
    """Each BF16 GEMM of the PyTorch path as one FP32 accumulation over K (cuBLASLt without split-K), the same form
    on every GPU model."""
    matmul = torch.backends.cuda.matmul
    if not hasattr(matmul, "allow_bf16_reduced_precision_reduction_split_k"):
        yield
        return
    library = torch.backends.cuda.preferred_blas_library()
    reduction = (matmul.allow_bf16_reduced_precision_reduction, matmul.allow_bf16_reduced_precision_reduction_split_k)
    torch.backends.cuda.preferred_blas_library("cublaslt")
    matmul.allow_bf16_reduced_precision_reduction = (False, False)
    try:
        yield
    finally:
        matmul.allow_bf16_reduced_precision_reduction = reduction
        torch.backends.cuda.preferred_blas_library(library)


@torch.no_grad()
def _oracle(p: Problem) -> dict:
    """``logp`` / ``dX`` / ``dW`` and the downstream loss ``sum(logp * dlogp)`` from the BF16 input values in FP32,
    without intermediate rounding, ``_ORACLE_ROWS`` rows at a time."""
    T, H = p.X.shape
    W32 = p.W.float()
    tau = p.tau_rows()
    logp = torch.zeros((T,), dtype=torch.float32, device=CUDA)
    dX = torch.empty((T, H), dtype=torch.float32, device=CUDA)
    dW = torch.zeros_like(W32)
    with _fp32_gemms():
        for r0 in range(0, T, _ORACLE_ROWS):
            r1 = min(T, r0 + _ORACLE_ROWS)
            x = p.X[r0:r1].float()
            z = x @ W32.t()
            if tau is not None:
                z /= tau[r0:r1, None]
            lse = torch.logsumexp(z, dim=-1)
            valid = p.valid[r0:r1]
            index = torch.where(valid, p.labels[r0:r1], torch.zeros_like(p.labels[r0:r1]))
            logp[r0:r1] = torch.where(valid, z.gather(1, index[:, None]).squeeze(1) - lse, torch.zeros_like(lse))
            d = p.dlogp[r0:r1] if tau is None else p.dlogp[r0:r1] / tau[r0:r1]
            d = torch.where(valid, d, torch.zeros_like(d))
            dz = z.sub_(lse[:, None]).exp_().neg_()  # -softmax, in place of the logits
            dz.scatter_add_(1, index[:, None], torch.ones((r1 - r0, 1), dtype=dz.dtype, device=CUDA))
            dz.mul_(d[:, None])  # ignored rows: d = 0
            dX[r0:r1] = dz @ W32
            dW.addmm_(dz.t(), x)
            del z, dz
    return dict(logp=logp, dX=dX, dW=dW, loss=_loss(logp, p))


def _baseline(p: Problem) -> dict:
    """The unchunked PyTorch path at the kernels' rounding boundaries."""
    X = p.X.detach().requires_grad_(True)
    W = p.W.detach().requires_grad_(True)
    tau = p.tau_rows()
    with torch.enable_grad(), _unsplit_gemms():
        z = (X @ W.t()).float()
        if tau is not None:
            z = z / tau[:, None]
        lse = torch.logsumexp(z, dim=-1)
        index = torch.where(p.valid, p.labels, torch.zeros_like(p.labels))
        logp = torch.where(p.valid, z.gather(1, index[:, None]).squeeze(1) - lse, torch.zeros_like(lse))
        del z
        logp.backward(p.dlogp)
    return dict(logp=logp.detach(), dX=X.grad, dW=W.grad, loss=_loss(logp.detach(), p))


def _loss(logp: torch.Tensor, p: Problem) -> torch.Tensor:
    return (logp.double() * p.dlogp.double())[p.valid].sum()


def _run(p: Problem, variant, *, train_x=True, train_w=True, grad=True, **kwargs) -> dict:
    """One forward (and, with ``grad`` and a trainable input, backward with ``p.dlogp``) through ``ops.log_probs``."""
    kwargs.setdefault("temperature", p.temperature)
    X = p.X.detach().requires_grad_(train_x)
    W = p.W.detach().requires_grad_(train_w)
    with torch.set_grad_enabled(grad):
        logp = ops.log_probs(X, W, p.labels, variant=variant, **kwargs)
    assert logp.dtype == torch.float32 and tuple(logp.shape) == (p.X.shape[0],)
    if grad and (train_x or train_w):
        logp.backward(p.dlogp)
    torch.cuda.synchronize()
    return dict(
        logp=logp.detach(), dX=X.grad, dW=W.grad, loss=_loss(logp.detach(), p), requires_grad=logp.requires_grad
    )


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _rel_l2(a: torch.Tensor, ref: torch.Tensor) -> float:
    """``||a - ref|| / ||ref||`` in FP64 sums, ``_NORM_ROWS`` rows at a time."""
    num = den = 0.0
    for r0 in range(0, a.shape[0], _NORM_ROWS):
        x, y = a[r0 : r0 + _NORM_ROWS].float(), ref[r0 : r0 + _NORM_ROWS].float()
        num += float((x - y).double().square().sum())
        den += float(y.double().square().sum())
    return num**0.5 / max(den**0.5, 1e-30)


def _finite(t) -> bool:
    return t is None or bool(torch.isfinite(t).all())


def _errors(result: dict, oracle: dict, p: Problem) -> dict:
    valid = p.valid
    errors = dict(loss_rel=abs(float(result["loss"]) - float(oracle["loss"])) / max(abs(float(oracle["loss"])), 1e-30))
    diff = (result["logp"].double() - oracle["logp"].double()).abs()
    errors["logp_max_abs"] = float(diff[valid].max()) if bool(valid.any()) else 0.0
    errors["logp_ignored_zero"] = bool((result["logp"][~valid] == 0).all())
    for name in ("dX", "dW"):
        if result[name] is not None:
            errors[f"{name}_rel_l2"] = _rel_l2(result[name], oracle[name])
    if result["dX"] is not None:
        errors["dX_ignored_zero"] = bool((result["dX"][~valid] == 0).all())
    errors["finite"] = all(_finite(result[name]) for name in ("logp", "dX", "dW"))
    return errors


def _assert_gate(result: dict, oracle: dict, baseline: dict, p: Problem, what: str) -> dict:
    errors = _errors(result, oracle, p)
    assert errors["finite"], f"{what}: a non-finite output"
    for key, floor in GATE_FLOOR.items():
        if key in errors:
            assert (
                errors[key] <= GATE_MARGIN * baseline[key] + floor
            ), f"{what} {key}: {errors[key]:.3e} vs the unchunked PyTorch path's {baseline[key]:.3e}"
    assert errors["logp_ignored_zero"], f"{what}: logp of an ignored row is not 0"
    if "dX_ignored_zero" in errors:
        assert errors["dX_ignored_zero"], f"{what}: dX of an ignored row is not 0"
    return errors


def _references(p: Problem) -> tuple[dict, dict]:
    """The oracle and the PyTorch path's errors against it (the PyTorch outputs are dropped)."""
    oracle = _oracle(p)
    baseline = _errors(_baseline(p), oracle, p)
    torch.cuda.empty_cache()
    return oracle, baseline


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view({torch.bfloat16: torch.int16, torch.float32: torch.int32}[t.dtype])


def _same_bits(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(_bits(a), _bits(b))


def _assert_same_bits(a: dict, b: dict, keys, what: str) -> None:
    for key in keys:
        assert _same_bits(a[key], b[key]), f"{what}: {key} bits differ"


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("temperature", TEMPERATURES, ids=lambda t: f"tau{t}")
@pytest.mark.parametrize("T", ROWS, ids=lambda t: f"T{t}")
@pytest.mark.parametrize(("H", "V"), SHAPES)
def test_log_probs_match_the_reference(H, V, T, temperature):
    """Both variants pass the gate; ``det`` repeats bit for bit; ``fast`` has ``det``'s ``logp`` and ``dW`` bits; the
    no-grad forward has the training forward's ``logp`` bits."""
    p = _problem(H, V, T, temperature=temperature)
    oracle, baseline = _references(p)

    det = _run(p, "det")
    _assert_gate(det, oracle, baseline, p, "det")
    _assert_same_bits(_run(p, "det"), det, ("logp", "dX", "dW"), "det repeat")
    _assert_same_bits(_run(p, "det", train_x=False, train_w=False, grad=False), det, ("logp",), "det no-grad")

    fast = _run(p, "fast")
    _assert_gate(fast, oracle, baseline, p, "fast")
    _assert_same_bits(fast, det, ("logp", "dW"), "fast vs det")
    _assert_same_bits(_run(p, "fast", train_x=False, train_w=False, grad=False), det, ("logp",), "fast no-grad")


@pytest.mark.parametrize(("H", "V", "T"), ISSUE_SHAPES)
def test_log_probs_at_the_cuda_13_0_fault_shapes(H, V, T):
    """The shapes of the CUDA 13.0 dW / dX GEMM fault reports, a third of the rows ignored (the compacted path):
    three ``det`` runs with the same bits within the gate, and ``fast`` within the gate."""
    p = _problem(H, V, T, ignore_frac=0.33)
    oracle, baseline = _references(p)

    det = _run(p, "det")
    _assert_gate(det, oracle, baseline, p, "det")
    for repeat in range(2):
        _assert_same_bits(_run(p, "det"), det, ("logp", "dX", "dW"), f"det repeat {repeat}")
    _assert_gate(_run(p, "fast"), oracle, baseline, p, "fast")


@pytest.mark.parametrize("temperature", TEMPERATURES, ids=lambda t: f"tau{t}")
@pytest.mark.parametrize("variant", VARIANTS)
def test_an_empty_call_returns_no_rows_and_a_zero_weight_gradient(variant, temperature):
    """A rank that scores no row stays in the autograd graph: empty ``logp`` and ``dX``, a zero ``dW``."""
    H, V = SHAPES[0]
    p = _problem(H, V, 0, temperature=temperature)

    for count in (None, 0):
        out = _run(p, variant, num_valid_rows=count)
        assert out["requires_grad"]
        assert out["dX"] is None or (tuple(out["dX"].shape) == (0, H) and out["dX"].dtype == torch.bfloat16)
        # a weight gradient every rank forms, so the data-parallel gradient reduction sees this rank's (zero) share
        assert tuple(out["dW"].shape) == (V, H) and out["dW"].dtype == torch.bfloat16
        assert not bool((out["dW"] != 0).any())


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_frozen_input_gets_no_gradient_and_leaves_the_other_one(variant):
    """``X`` or ``W`` frozen: its gradient is not formed, the other one passes the gate (and, for ``det``, keeps the
    bits of the call with both trainable); with both frozen the output needs no gradient."""
    H, V = SHAPES[0]
    p = _problem(H, V, 4097, temperature=0.7)
    oracle, baseline = _references(p)
    both = _run(p, variant)
    x_only = _run(p, variant, train_w=False)
    w_only = _run(p, variant, train_x=False)
    neither = _run(p, variant, train_x=False, train_w=False)

    assert x_only["dW"] is None and w_only["dX"] is None and neither["dX"] is None and neither["dW"] is None
    assert not neither["requires_grad"]
    for name, result in (("X only", x_only), ("W only", w_only), ("neither", neither)):
        _assert_gate(result, oracle, baseline, p, f"{variant} {name}")
        _assert_same_bits(result, both, ("logp",), f"{variant} {name}")
    _assert_same_bits(w_only, both, ("dW",), f"{variant} W only")
    if variant == "det":
        _assert_same_bits(x_only, both, ("dX",), "det X only")


@pytest.mark.parametrize("ignore_frac", [0.0, 0.05], ids=["all-valid", "some-ignored"])
@pytest.mark.parametrize("T", [4097, 16231], ids=lambda t: f"T{t}")
def test_a_given_valid_row_count_gives_the_counted_bits(T, ignore_frac):
    """miles knows the count (every scored row has a target): the call with ``num_valid_rows`` gives the bits of
    the call that counts on the device (``fast``: ``logp`` and ``dW``)."""
    H, V = SHAPES[2]
    p = _problem(H, V, T, temperature=0.7, ignore_frac=ignore_frac)
    count = int(p.valid.sum())

    for variant, keys in (("det", ("logp", "dX", "dW")), ("fast", ("logp", "dW"))):
        counted = _run(p, variant)
        _assert_same_bits(_run(p, variant, num_valid_rows=count), counted, keys, f"{variant} num_valid_rows")


@pytest.mark.parametrize("variant", VARIANTS)
def test_zero_gradient_rows_get_a_zero_hidden_gradient(variant):
    """PPO-clipped and masked tokens reach the kernels as rows with a zero incoming gradient (``fast`` skips their
    GEMM work above one chunk): within the gate, with exact zeros in their ``dX`` rows."""
    H, V = SHAPES[0]
    p = _problem(H, V, 16231, temperature=0.7, zero_dlogp_frac=0.3)
    oracle, baseline = _references(p)

    out = _run(p, variant)
    _assert_gate(out, oracle, baseline, p, variant)
    zero = p.valid & (p.dlogp == 0)
    assert int(zero.sum()) > 0
    assert not bool((out["dX"][zero] != 0).any())


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_unit_temperature_is_no_temperature(variant):
    H, V = SHAPES[0]
    p = _problem(H, V, 4097)
    plain = _run(p, variant)

    keys = ("logp", "dX", "dW") if variant == "det" else ("logp", "dW")
    for unit in (1.0, 1):
        _assert_same_bits(_run(p, variant, temperature=unit), plain, keys, f"{variant} temperature {unit}")


def test_fast_runs_det_under_deterministic_algorithms(monkeypatch, caplog):
    """``torch.use_deterministic_algorithms(True)``: a ``fast`` call runs ``det`` (its bits) and says so once."""
    monkeypatch.setattr(_variant, "_warned", set())
    H, V = SHAPES[0]
    p = _problem(H, V, 4097, temperature=0.7)
    det = _run(p, "det")

    previous = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        with caplog.at_level(logging.WARNING, logger="miles_ext_kernels"):
            runs = [_run(p, "fast") for _ in range(2)]
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
    for out in runs:
        _assert_same_bits(out, det, ("logp", "dX", "dW"), "fast under deterministic algorithms")
    assert [record.getMessage() for record in caplog.records if record.name == "miles_ext_kernels"] == [
        "miles_ext_kernels.lm_head: the fast variant does not serve torch.use_deterministic_algorithms(True); "
        "running det"
    ]


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_configured_variant_and_chunk_size_are_the_defaults(monkeypatch, variant):
    H, V = SHAPES[0]
    p = _problem(H, V, 4097)
    monkeypatch.setattr(ops, "VARIANT", variant)
    monkeypatch.setattr(ops, "CHUNK_SIZE", 1024)

    configured = _run(p, None)
    explicit = _run(p, variant, chunk_size=1024)
    _assert_same_bits(configured, explicit, ("logp", "dX", "dW") if variant == "det" else ("logp", "dW"), variant)


def test_an_unsupported_device_is_a_clear_error(monkeypatch):
    H, V = SHAPES[0]
    p = _problem(H, V, 8)
    with pytest.raises(RuntimeError, match="run on a CUDA device"):
        ops.log_probs(p.X.cpu(), p.W[:256].cpu(), p.labels.cpu())

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    with pytest.raises(RuntimeError, match=r"need compute capability 10\.0 .* or 10\.3 .*; cuda:\d+ has 9\.0"):
        ops.log_probs(p.X, p.W, p.labels)
    with pytest.raises(RuntimeError, match="need compute capability"):
        ops.prebuild()


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize(("H", "V", "message"), [(4352, 1024, "512"), (4096, 1000, "multiple of 256")])
def test_a_geometry_the_kernels_do_not_serve_is_a_clear_error(variant, H, V, message):
    X = torch.zeros((8, H), dtype=torch.bfloat16, device=CUDA)
    W = torch.zeros((V, H), dtype=torch.bfloat16, device=CUDA)
    labels = torch.zeros((8,), dtype=torch.int64, device=CUDA)
    with pytest.raises(ValueError, match=message):
        ops.log_probs(X, W, labels, variant=variant)


def test_an_unknown_variant_is_rejected():
    H, V = SHAPES[0]
    p = _problem(H, V, 8)
    with pytest.raises(ValueError, match="variant must be one of"):
        ops.log_probs(p.X, p.W, p.labels, variant="fastest")
