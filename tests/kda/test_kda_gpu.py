"""The deterministic KDA kernels as miles calls them (raw gate, in-kernel q / k L2 norm, ``[1, S, H, 128]`` operands)
against the exact FP64 recurrence, bit-identical repeats and backward routes, and flash-linear-attention.

Shapes: ragged packs of one and several documents whose lengths are not multiples of 64, at the per-rank head counts
miles runs (Kimi-K3's 96 KDA heads at tensor parallelism 2 / 4 / 8: 48 / 24 / 12; GLM-5.3-Flash: 64) and the
segmented route's 1 and 4 heads. The first test of the module compiles every library of those head counts
(``TVM_FFI_CACHE_DIR``, once per toolkit and machine). Needs an SM100a / SM103a GPU, torch and tvm-ffi; the fla
parity rows also need flash-linear-attention, and skip without it. ``-rP`` prints the fla parity tables.
"""

import os

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)
if tuple(torch.cuda.get_device_capability()) not in ((10, 0), (10, 3)):
    pytest.skip("the Cake KDA kernels are validated on SM100a / SM103a", allow_module_level=True)
pytest.importorskip("tvm_ffi")

from miles_ext_kernels import cake_native  # noqa: E402
from miles_ext_kernels.kda import ops  # noqa: E402
from miles_ext_kernels.kda._vendor.cake_kda_det_train import cake_backend, cake_build, cake_jit  # noqa: E402

from ._reference import LOWER_BOUND, RAW_GRAD_NAMES, D, elementwise_gate, error_metrics, make_raw_inputs  # noqa: E402
from ._reference_l2n import kda_exact_raw_l2n_with_grads, raw_qk  # noqa: E402

# The package's declared accuracy gate: every result elementwise within ATOL + RTOL * |ref| of the exact FP64
# recurrence of the same inputs.
ATOL = RTOL = 1e-2
SHAPES = [
    ((1000,), 1),
    ((63, 1, 65, 300), 1),
    ((1000, 37, 300), 4),
    ((777,), 12),
    ((300, 129, 500), 12),
    ((1100, 257), 24),
    ((513, 70, 900), 48),
    ((2000,), 64),
    ((1000, 37, 300, 63), 64),
]
HEADS = sorted({heads for _, heads in SHAPES})
NAMES = ("out",) + RAW_GRAD_NAMES  # out, dq, dk, dv, dg (the raw gate's), dbeta, dA_log, ddt_bias
BWD_NAMES = ("dq", "dk", "dv", "dbeta", "dg", "dA_log", "ddt_bias")  # the order chunk_kda_bwd returns them in
# The miles call's keywords besides the operands.
MILES_MODES = {"use_gate_in_kernel": True, "use_qk_l2norm_in_kernel": True}


def _shape_id(shape) -> str:
    lengths, heads = shape
    return "s" + "_".join(str(n) for n in lengths) + f"-h{heads}"


@pytest.fixture(scope="module", autouse=True)
def device_kind():
    """``(arch, sm_count)`` of the current device, after compiling every library of the module's head counts."""
    index = torch.cuda.current_device()
    arch, sm_count = ops.device_arch(index), cake_jit.sm_count(index)
    names = sorted({name for heads in HEADS for name in ops.route_libraries(heads, arch, sm_count)})
    cake_native.prebuild(cake_build, names, max_workers=min(32, len(os.sched_getaffinity(0))))
    for heads in HEADS:
        ops.prepare(heads, index)
    return arch, sm_count


def _bits(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().contiguous()
    return x.view({torch.bfloat16: torch.int16, torch.float32: torch.int32}[x.dtype])


def _assert_same_bits(got: dict, want: dict, tag) -> None:
    for name in want:
        assert got[name].shape == want[name].shape and got[name].dtype == want[name].dtype, (tag, name)
        assert torch.equal(_bits(got[name]), _bits(want[name])), (tag, name)


def _inputs(lengths, heads: int) -> dict:
    """The raw-gate operands with raw (not normalised) q / k."""
    inputs = make_raw_inputs(lengths, heads, "cuda")
    inputs["q"], inputs["k"] = raw_qk(lengths, heads, "cuda")
    return inputs


def _cake(q, k, v, g, beta, A_log, dt_bias, cu_seqlens):
    """miles' call: ``[1, S, H, 128]`` operands, the raw gate with lower bound -5, the in-kernel q / k L2 norm."""
    out = ops.chunk_kda(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        g=g.unsqueeze(0),
        beta=beta.unsqueeze(0),
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=LOWER_BOUND,
        cu_seqlens=cu_seqlens,
        **MILES_MODES,
    )
    assert out.shape == (1, *v.shape) and out.dtype == torch.bfloat16
    return out.squeeze(0)


def _run(inputs: dict, *, packed: bool = False, checkpoint: str | None = None, beta=None, kernel=_cake) -> dict:
    """``out`` and the gradients (``NAMES``) of one autograd step of ``kernel``.

    ``packed``: q / k / v are column slices of one ``[S, 3 * H * 128]`` leaf, the GLM-5.3-Flash layout (q / k
    strided views, ``v`` made contiguous). ``checkpoint``: under ``torch.utils.checkpoint`` (``"reentrant"`` as
    Megatron's recompute, or ``"non_reentrant"``). ``beta`` replaces the inputs' BF16 beta.
    """
    S, H, _ = inputs["v"].shape
    width = H * D
    cu_seqlens = inputs["cu_seqlens"]
    if packed:
        qkv = torch.cat([inputs[n].reshape(S, width) for n in ("q", "k", "v")], 1).requires_grad_(True)
        leaves = [qkv]

        def split(qkv):
            q, k, v = (qkv[:, i * width : (i + 1) * width].view(S, H, D) for i in range(3))
            return q, k, v.contiguous()

    else:
        leaves = [inputs[n].clone().requires_grad_(True) for n in ("q", "k", "v")]

        def split(q, k, v):
            return q, k, v

    beta = inputs["beta"] if beta is None else beta
    params = [x.clone().requires_grad_(True) for x in (inputs["g"], beta, inputs["A_log"], inputs["dt_bias"])]

    def step(*tensors):
        return kernel(*split(*tensors[: len(leaves)]), *tensors[len(leaves) :], cu_seqlens)

    if checkpoint is None:
        out = step(*leaves, *params)
    else:
        out = torch.utils.checkpoint.checkpoint(step, *leaves, *params, use_reentrant=checkpoint == "reentrant")
    out.backward(inputs["dout"])
    if packed:
        dq, dk, dv = (qkv.grad[:, i * width : (i + 1) * width].reshape(S, H, D) for i in range(3))
    else:
        dq, dk, dv = (leaf.grad for leaf in leaves)
    g, beta, A_log, dt_bias = (param.grad for param in params)
    return {
        "out": out.detach(),
        "dq": dq,
        "dk": dk,
        "dv": dv,
        "dg": g,
        "dbeta": beta,
        "dA_log": A_log,
        "ddt_bias": dt_bias,
    }


def _reference(inputs: dict, beta=None) -> dict:
    out, grads = kda_exact_raw_l2n_with_grads(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"] if beta is None else beta,
        inputs["A_log"],
        inputs["dt_bias"],
        inputs["dout"],
        inputs["cu_seqlens"],
        ops.DEFAULT_SCALE,
        LOWER_BOUND,
    )
    return {"out": out, **grads}


def test_head_counts_outside_1_to_64_and_batches_are_rejected():
    for heads in (0, 65, 96):  # 96: Kimi-K3's KDA layers without tensor parallelism
        with pytest.raises(ValueError, match="serve 1 to 64 heads per call"):
            ops.prepare(heads)
    x = torch.zeros(1, 3, 65, D, dtype=torch.bfloat16, device="cuda")
    cu_seqlens = torch.tensor([0, 3], dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="serve 1 to 64 heads per call"):
        ops.chunk_kda(x, x, x, x, x[..., 0], cu_seqlens)
    inputs = _inputs((5, 5), 2)
    two = [inputs[n].reshape(2, 5, *inputs[n].shape[1:]) for n in ("q", "k", "v", "g", "beta")]
    with pytest.raises(ValueError, match="batch axis must be 1"):
        ops.chunk_kda(
            *two,
            cu_seqlens=inputs["cu_seqlens"],
            A_log=inputs["A_log"],
            dt_bias=inputs["dt_bias"],
            lower_bound=LOWER_BOUND,
            **MILES_MODES,
        )


def test_the_device_needs_its_architectures_toolkit(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (9, 0))
    with pytest.raises(RuntimeError, match="compute capability 10.0 or 10.3"):
        ops.device_arch()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (10, 7))
    monkeypatch.setattr(cake_native, "get_cuda_version", lambda: cake_native.CudaVersion(13, 0, "13.0.88"))
    with pytest.raises(RuntimeError, match="sm_107a .* build with CUDA 13.4 or newer"):
        ops.device_arch()
    monkeypatch.setattr(cake_native, "get_cuda_version", lambda: cake_native.CudaVersion(13, 4, "13.4.92"))
    assert ops.device_arch() == "sm_107a"


def test_prepare_compiled_every_route_library(device_kind):
    arch, sm_count = device_kind
    for heads in HEADS:
        names = ops.route_libraries(heads, arch, sm_count)
        assert names and all(name.endswith(f"_{arch}") for name in names)
        missing = [name for name in names if not cake_build.jit_spec(name).get_library_path().is_file()]
        assert missing == [], (heads, missing)


@pytest.mark.parametrize("heads", [1, 2, 4, 8, 12, 16, 24, 32, 48, 64])
def test_the_route_libraries_cover_every_launch_of_the_autograd_node(device_kind, heads):
    arch, sm_count = device_kind
    names = set(ops.route_libraries(heads, arch, sm_count))
    planned = 0
    for num_docs in (1, 2, 3, 4, 5, 8, 9, 16, 31, 32, 33, 64, 100, 255, 256, 257, 600, 2048, 4096, 8193, 65536):
        for tokens in (num_docs, 64 * num_docs + 4000, 400_000):
            try:
                plan = cake_backend.plan(tokens, num_docs, heads, sm_count, arch, gate="raw", l2n="fused")
            except ValueError:  # beyond the kernels' int32 offsets: no such pack runs
                continue
            planned += 1
            launched = {
                cake_jit.library_name(record, arch)
                for path in ("fwd", plan.light_path)
                for record in plan.records[path].values()
                if record is not None
            }
            assert launched <= names, (num_docs, tokens, sorted(launched - names))
    assert planned


@pytest.mark.parametrize("shape", SHAPES, ids=_shape_id)
def test_matches_the_exact_fp64_reference(shape):
    lengths, heads = shape
    inputs = _inputs(lengths, heads)
    got = _run(inputs, packed=True)
    want = _reference(inputs)
    failures = []
    for name in NAMES:
        assert got[name].shape == want[name].shape, name
        gate = elementwise_gate(got[name], want[name], ATOL, RTOL)
        if not gate["pass"]:
            failures.append((name, gate, error_metrics(got[name], want[name])))
    assert not failures, failures


@pytest.mark.parametrize("shape", SHAPES, ids=_shape_id)
def test_repeated_steps_are_bitwise_identical(shape):
    lengths, heads = shape
    inputs = _inputs(lengths, heads)
    first = _run(inputs, packed=True)
    for repeat in range(2):
        _assert_same_bits(_run(inputs, packed=True), first, ("repeat", repeat))


@pytest.mark.parametrize("shape", [((1000, 37, 300), 4), ((1100, 257), 24), ((1000, 37, 300, 63), 64)], ids=_shape_id)
def test_packed_qk_views_give_the_bits_of_contiguous_copies(shape):
    lengths, heads = shape
    inputs = _inputs(lengths, heads)
    _assert_same_bits(_run(inputs, packed=True), _run(inputs, packed=False), "packed")


def _states_shape(device_kind) -> tuple[tuple[int, ...], int]:
    """One document at one head long enough that the forward saves the segmented route's segment states."""
    arch, sm_count = device_kind
    segments = cake_backend.plan(1, 1, 1, sm_count, arch).NSEG
    return (cake_backend.SEG_SAVE_MIN_TOKENS_PER_SEG * segments + 37,), 1


@pytest.mark.parametrize("case", ["serial-h12", "segmented-h4", "segmented-states-h1"])
def test_autograd_saved_set_replay_and_checkpoint_give_the_same_bits(device_kind, case):
    arch, sm_count = device_kind
    lengths, heads = {
        "serial-h12": ((300, 129, 500), 12),
        "segmented-h4": ((1000, 37, 300), 4),
        "segmented-states-h1": _states_shape(device_kind),
    }[case]
    plan = cake_backend.plan(sum(lengths), len(lengths), heads, sm_count, arch, gate="raw", l2n="fused")
    assert (plan.recur, plan.light_path) == {
        "serial-h12": ("serial", "light"),
        "segmented-h4": ("seg", "light"),
        "segmented-states-h1": ("seg", "light_states"),
    }[case]
    inputs = _inputs(lengths, heads)
    reference = _run(inputs)
    for checkpoint in ("reentrant", "non_reentrant"):
        _assert_same_bits(_run(inputs, checkpoint=checkpoint), reference, checkpoint)

    operands = [inputs[n].unsqueeze(0) for n in ("q", "k", "v", "g", "beta")]
    modes = dict(A_log=inputs["A_log"], dt_bias=inputs["dt_bias"], lower_bound=LOWER_BOUND, **MILES_MODES)
    out, saved = ops.chunk_kda_fwd(*operands, inputs["cu_seqlens"], **modes)
    assert (saved.hseg is not None) == (plan.light_path == "light_states")
    _assert_same_bits({"out": out.squeeze(0)}, {"out": reference["out"]}, "explicit forward")
    dout = inputs["dout"].unsqueeze(0)
    for tag, set_ in (("saved set", saved), ("replay", None)):
        grads = ops.chunk_kda_bwd(*operands, inputs["cu_seqlens"], dout, saved=set_, **modes)
        got = {
            name: grad.squeeze(0) if name not in ("dA_log", "ddt_bias") else grad
            for name, grad in zip(BWD_NAMES, grads)
        }
        _assert_same_bits(got, {name: reference[name] for name in BWD_NAMES}, tag)


@pytest.mark.parametrize("shape", [((300, 129, 500), 12), ((1000, 37, 300, 63), 64)], ids=_shape_id)
def test_fla_parity_with_fp32_beta(monkeypatch, shape):
    """The Cake path takes ``beta`` in BF16; fla takes the FP32 ``sigmoid(beta_logits)`` miles gives it. Both against
    the exact FP64 recurrence of fla's FP32 beta: the Cake error stays within twice fla's or 1 % (relative RMS), and
    the table reports the cost of the BF16 beta alone (the FP64 recurrence of the BF16 beta against it)."""
    monkeypatch.setenv("FLA_TILELANG", "0")  # what miles sets on Blackwell
    fla_chunk_kda = pytest.importorskip("fla.ops.kda").chunk_kda
    lengths, heads = shape
    inputs = _inputs(lengths, heads)
    generator = torch.Generator().manual_seed(4194)
    logits = torch.randn(sum(lengths), heads, generator=generator).to(torch.bfloat16).float().cuda()
    beta32 = logits.sigmoid()
    beta16 = beta32.to(torch.bfloat16)

    def fla(q, k, v, g, beta, A_log, dt_bias, cu_seqlens):
        out, _ = fla_chunk_kda(
            q=q.unsqueeze(0),
            k=k.unsqueeze(0),
            v=v.unsqueeze(0),
            g=g.unsqueeze(0),
            beta=beta.unsqueeze(0),
            A_log=A_log,
            dt_bias=dt_bias,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=LOWER_BOUND,
            transpose_state_layout=True,
            cu_seqlens=cu_seqlens,
        )
        return out.squeeze(0)

    cake = _run(inputs, beta=beta16)
    theirs = _run(inputs, beta=beta32, kernel=fla)
    exact32 = _reference(inputs, beta=beta32)
    exact16 = _reference(inputs, beta=beta16)
    rows, failures = [], []
    for name in NAMES:
        cake_error = error_metrics(cake[name], exact32[name])["rel_rms"]
        fla_error = error_metrics(theirs[name], exact32[name])["rel_rms"]
        pair = error_metrics(cake[name], theirs[name])["rel_rms"]
        cast = error_metrics(exact16[name], exact32[name])["rel_rms"]
        rows.append(f"{name:>9} {cake_error:12.3e} {fla_error:12.3e} {pair:12.3e} {cast:12.3e}")
        if not cake_error <= max(2 * fla_error, 1e-2):
            failures.append((name, cake_error, fla_error))
    print(f"\n{_shape_id(shape)}: relative RMS error against the FP64 recurrence of the FP32 beta")
    print(f"{'tensor':>9} {'cake bf16':>12} {'fla fp32':>12} {'cake-fla':>12} {'bf16 beta':>12}")
    print("\n".join(rows))
    assert not failures, failures
