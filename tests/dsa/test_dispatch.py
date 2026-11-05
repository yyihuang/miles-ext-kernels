"""Host-side dispatch of the DSA kernels without a GPU: which det stage a shape selects, which calls the fast variant
serves and how the others fall back to det, and the top-k helpers (needs torch, not CUDA)."""

import logging

import pytest

torch = pytest.importorskip("torch")

from miles_ext_kernels import _variant  # noqa: E402
from miles_ext_kernels.dsa import _jit, _kernels, ops  # noqa: E402

TC_STAGES = {f"dsa_attention_fwd_tc_h{h}_d512t{t}" for h in (8, 16, 32, 64) for t in (64, 0)}
TC_BWD_STAGES = {f"dsa_attention_bwd_tc_h{h}_d512t{t}" for h in (8, 16, 32, 64) for t in (64, 0)}
INDEXER_TC_STAGES = {f"dsa_indexer_logits_tc_h{h}" for h in (8, 16, 32, 64)}
INDEXER_TC_BWD_STAGES = {f"dsa_indexer_bwd_tc_h{h}" for h in (32, 64)}


def test_the_indexer_backward_selects_the_tcgen05_stage_for_32_and_64_heads(monkeypatch):
    monkeypatch.setattr(_kernels, "INDEXER_BWD_TC", True)
    assert _kernels.INDEXER_TC_BWD_HEADS == (32, 64)
    assert _kernels.INDEXER_TC_BWD_MIN_TOPK == {32: 1024, 64: 0}
    for arch in ("sm_100a", "sm_103a", "sm_107a"):
        assert _kernels.indexer_tc_bwd_stage(32, 2048, arch) == "dsa_indexer_bwd_tc_h32"
        assert _kernels.indexer_tc_bwd_stage(32, 1024, arch) == "dsa_indexer_bwd_tc_h32"
        assert _kernels.indexer_tc_bwd_stage(64, 512, arch) == "dsa_indexer_bwd_tc_h64"
        assert _kernels.indexer_tc_bwd_stage(64, 128, arch) == "dsa_indexer_bwd_tc_h64"
        for heads in (32, 64):
            assert _kernels.indexer_tc_bwd_stage(heads, 2048, arch) in INDEXER_TC_BWD_STAGES
        assert _kernels.indexer_tc_bwd_stage(32, 512, arch) is None
        assert _kernels.indexer_tc_bwd_stage(32, 992, arch) is None
        for heads in (1, 8, 16, 17, 33, 48, 65, 128):
            assert _kernels.indexer_tc_bwd_stage(heads, 2048, arch) is None
    assert _kernels.indexer_tc_bwd_stage(32, 2048, "sm_90a") is None
    monkeypatch.setattr(_kernels, "INDEXER_BWD_TC", False)
    assert _kernels.indexer_tc_bwd_stage(32, 2048, "sm_100a") is None


def test_the_attention_backward_selects_the_tcgen05_stages_for_8_16_32_and_64_heads(monkeypatch):
    monkeypatch.setattr(_kernels, "ATTENTION_BWD_TC", True)
    for arch in ("sm_100a", "sm_103a", "sm_107a"):
        for heads in (8, 16, 32, 64):
            for tail in (64, 0):
                stage = _kernels.attention_tc_bwd_stage(heads, tail, arch)
                assert stage == f"dsa_attention_bwd_tc_h{heads}_d512t{tail}" and stage in TC_BWD_STAGES
        for heads in (1, 4, 12, 24, 40, 48, 65, 128):
            assert _kernels.attention_tc_bwd_stage(heads, 0, arch) is None
    assert _kernels.attention_tc_bwd_stage(64, 0, "sm_90a") is None
    monkeypatch.setattr(_kernels, "ATTENTION_BWD_TC", False)
    assert _kernels.attention_tc_bwd_stage(64, 0, "sm_100a") is None
    assert _kernels.AttentionGeometry(64, 512, 0, 32, 2, 640, False, True).kv_partials == 1
    assert _kernels.AttentionGeometry(64, 512, 0, 32, 2, 640, True).kv_partials == 1
    assert _kernels.AttentionGeometry(64, 512, 0, 32, 2, 640, False).kv_partials == 2
    assert _kernels.AttentionGeometry(16, 512, 64, 16, 1, 2048, False, True).d_qk == 576


def test_the_attention_forward_and_indexer_logits_select_the_tcgen05_stages(monkeypatch):
    monkeypatch.setattr(_kernels, "ATTENTION_TC", True)
    monkeypatch.setattr(_kernels, "INDEXER_TC", True)
    for arch in ("sm_100a", "sm_103a", "sm_107a"):
        for heads in (8, 16, 32, 64):
            for tail in (64, 0):
                assert _kernels.attention_tc_stage(heads, tail, arch) in TC_STAGES
            assert _kernels.indexer_tc_stage(heads, arch) == f"dsa_indexer_logits_tc_h{heads}"
            assert _kernels.indexer_tc_stage(heads, arch) in INDEXER_TC_STAGES
            assert _kernels.INDEXER_TC_QUERY_COLUMNS % heads == 0
        for heads in (1, 4, 17, 48, 65, 128):
            assert _kernels.attention_tc_stage(heads, 0, arch) is None
            assert _kernels.indexer_tc_stage(heads, arch) is None
    monkeypatch.setattr(_kernels, "ATTENTION_TC", False)
    monkeypatch.setattr(_kernels, "INDEXER_TC", False)
    assert _kernels.attention_tc_stage(64, 0, "sm_100a") is None
    assert _kernels.indexer_tc_stage(32, "sm_100a") is None
    assert _kernels.attention_block_h(64) == 32 and _kernels.attention_head_blocks(64) == 2


def test_sm107a_compiles_every_sm103a_record_for_compute_107a():
    assert _jit.SUPPORTED_CAPABILITIES[(10, 7)] == "sm_107a"
    assert _jit.EXPORT_ARCH == {"sm_107a": "sm_103a"}
    assert "compute_107a" in _jit._ARCH_FLAGS["sm_107a"]
    for stage, record in _jit.MODULES.items():
        assert _jit._record(stage, "sm_107a") is _jit._record(stage, "sm_103a") is record, stage
    with pytest.raises(NotImplementedError, match="not exported for sm_90a"):
        _jit._record("dsa_indexer_clean", "sm_90a")


@pytest.mark.parametrize("d_qk", [576, 512], ids=["rope", "nope"])
def test_fast_serves_64_heads_without_a_sink_at_d_v_512(d_qk):
    assert ops.fast_unserved_reason(64, d_qk, 512, has_sink=False, num_kv=4096) is None


@pytest.mark.parametrize(
    ("heads", "d_qk", "d_v", "has_sink", "num_kv", "reason"),
    [
        (64, 512, 512, True, 64, "calls with an attention sink"),
        (16, 576, 512, False, 64, "16 local query heads (only 64)"),
        (8, 512, 512, True, 64, "calls with an attention sink"),
        (128, 576, 512, False, 64, "128 local query heads (only 64)"),
        (64, 576, 448, False, 64, "d_qk 576 with d_v 448"),
        (64, 640, 512, False, 64, "d_qk 640 with d_v 512"),
        (64, 576, 512, False, 0, "an empty key set"),
    ],
)
def test_fast_names_why_it_does_not_serve_a_call(heads, d_qk, d_v, has_sink, num_kv, reason):
    assert ops.fast_unserved_reason(heads, d_qk, d_v, has_sink=has_sink, num_kv=num_kv).startswith(reason)


@pytest.fixture
def routes(monkeypatch):
    """Record which variant ``ops.sparse_attention`` runs, without kernels (CPU tensors)."""
    taken = []

    def fast(q, kv, indices, sm_scale, layout):
        taken.append(("fast", layout, sm_scale))
        return "fast"

    class Det:
        @staticmethod
        def apply(q, kv, attn_sink, indices, sm_scale, layout, d_v, workspace_bytes):
            taken.append(("det", layout, sm_scale))
            return "det"

    monkeypatch.setattr(ops, "_fast_sparse_attention", fast)
    monkeypatch.setattr(ops, "SparseAttentionFunction", Det)
    monkeypatch.setattr(_variant, "_warned", set())
    monkeypatch.setattr(ops, "VARIANT", "det")
    return taken


def _glm5(heads, d_qk=576, rows=4):
    q = torch.zeros(rows, heads, d_qk, dtype=torch.bfloat16)
    kv = torch.zeros(rows, 1, d_qk, dtype=torch.bfloat16)
    return q, kv, torch.zeros(rows, 1, 64, dtype=torch.int32)


def test_the_module_variant_selects_the_route_and_a_call_may_name_its_own(routes):
    q, kv, idx = _glm5(64)
    assert ops.sparse_attention(q, kv, idx) == "det"
    ops.VARIANT = "fast"
    assert ops.sparse_attention(q, kv, idx) == "fast"
    assert ops.sparse_attention(q, kv, idx, variant="det") == "det"
    assert [route for route, _, _ in routes] == ["det", "fast", "det"]
    # the default scale is d_qk ** -0.5 on both routes
    assert {scale for _, _, scale in routes} == {576**-0.5}


def test_fast_runs_batched_rows_through_the_flat_contract(routes):
    q = torch.zeros(2, 4, 64, 512, dtype=torch.bfloat16)
    kv = torch.zeros(2, 5, 512, dtype=torch.bfloat16)
    assert (
        ops.sparse_attention(q, kv, torch.zeros(2, 4, 64, dtype=torch.int32), layout="bshd", variant="fast") == "fast"
    )
    assert routes == [("fast", "bshd", 512**-0.5)]


def test_a_call_fast_does_not_serve_runs_det_and_warns_once_per_reason(routes, caplog):
    sink = torch.zeros(64)
    with caplog.at_level(logging.WARNING, logger="miles_ext_kernels"):
        for _ in range(3):
            assert ops.sparse_attention(*_glm5(64), attn_sink=sink, variant="fast") == "det"
            assert ops.sparse_attention(*_glm5(16), variant="fast") == "det"
    messages = [record.getMessage() for record in caplog.records]
    assert messages == [
        "miles_ext_kernels.dsa: the fast variant does not serve calls with an attention sink; running det",
        "miles_ext_kernels.dsa: the fast variant does not serve 16 local query heads (only 64); running det",
    ]
    assert [route for route, _, _ in routes] == ["det"] * 6


def test_non_bf16_inputs_and_unknown_variants_are_rejected_on_every_route(routes):
    q, kv, idx = _glm5(64)
    for variant in ("det", "fast"):
        with pytest.raises(TypeError, match="BF16"):
            ops.sparse_attention(q.float(), kv, idx, variant=variant)
    with pytest.raises(ValueError, match="variant must be one of"):
        ops.sparse_attention(q, kv, idx, variant="nd")
    assert routes == []


def test_torch_topk_marks_unselectable_and_missing_picks_minus_one():
    logits = torch.tensor([[0.5, float("-inf"), 2.0], [float("-inf")] * 3])
    assert ops.torch_topk(logits, 2).tolist() == [[2, 0], [-1, -1]]
    assert ops.select_topk(logits, 4).tolist() == [[2, 0, -1, -1], [-1, -1, -1, -1]]


def test_select_topk_hands_the_flat_rows_to_the_caller_policy():
    seen = []

    def topk_fn(rows, count):
        seen.append((tuple(rows.shape), count))
        return torch.zeros(rows.shape[0], count, dtype=torch.int64)

    picked = ops.select_topk(torch.zeros(2, 3, 5), 8, topk_fn)
    assert seen == [((6, 5), 5)]
    assert picked.shape == (2, 3, 8) and picked.dtype == torch.int32
    assert (picked[..., 5:] == -1).all()
