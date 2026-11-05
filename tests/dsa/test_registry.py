"""Packaging of the DSA kernels: torch-free imports, the registry of the generated det stages and the sources it names,
and the host modules' independence from the environment and from miles (torch-free)."""

import ast
import fnmatch
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DSA = REPO_ROOT / "miles_ext_kernels" / "dsa"
ARCHES = {"sm_100a", "sm_103a"}  # exported; SM107a compiles the SM103a records (_jit.EXPORT_ARCH)

TC_STAGES = {f"dsa_attention_fwd_tc_h{h}_d512t{t}" for h in (8, 16, 32, 64) for t in (64, 0)}
TC_BWD_STAGES = {f"dsa_attention_bwd_tc_h{h}_d512t{t}" for h in (8, 16, 32, 64) for t in (64, 0)}
INDEXER_TC_STAGES = {f"dsa_indexer_logits_tc_h{h}" for h in (8, 16, 32, 64)}
INDEXER_TC_BWD_STAGES = {f"dsa_indexer_bwd_tc_h{h}" for h in (32, 64)}
STAGES = {
    *(f"dsa_attention_{kind}_h{h}_d512t{t}" for kind in ("fwd", "bwd") for h in (16, 32) for t in (64, 0)),
    *(f"dsa_attention_bwd_h32x2_d512t{t}" for t in (64, 0)),
    *(f"dsa_indexer_logits_h{h}" for h in (8, 16, 32, 64)),
    *(f"dsa_indexer_bwd_h{h}" for h in (16, 32, 64)),
    "dsa_indexer_clean",
    "dsa_segmented_reduce",
    "dsa_segmented_reduce_split4",
    "dsa_attention_delta",
    *TC_STAGES,
    *TC_BWD_STAGES,
    *INDEXER_TC_STAGES,
    *INDEXER_TC_BWD_STAGES,
}

_NO_TORCH_ON_IMPORT = """
import argparse
import sys

sys.modules["torch"] = None  # any import of torch or tvm_ffi raises ImportError
sys.modules["tvm_ffi"] = None

import miles_ext_kernels.dsa
from miles_ext_kernels.dsa import arguments, configure

parser = argparse.ArgumentParser()
arguments.add_arguments(parser)
args = parser.parse_args(["--cake-dsa-variant", "det"])
args.dsa_attention_backend = "cake"
arguments.validate_args(args)
loaded = sorted(n for n in sys.modules if n.startswith("miles_ext_kernels.dsa."))
assert loaded == ["miles_ext_kernels.dsa.arguments"], loaded
"""


def _registry() -> dict:
    return json.loads((DSA / "registry.json").read_text())


def _arg_names(record: dict) -> list[str]:
    return [name for kind, name in record["arg_plan"] if kind != "grid"]


def test_importing_the_package_and_validating_its_arguments_does_not_import_torch():
    pythonpath = os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", _NO_TORCH_ON_IMPORT],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": pythonpath},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_the_registry_holds_every_stage_for_both_architectures_from_sources_on_disk():
    registry = _registry()

    assert set(registry) == STAGES
    for stage, record in registry.items():
        assert set(record["arches"]) == ARCHES, stage
        assert record["kernel_symbol"] == f"kernel_{record['name']}", stage
        assert record["ffi_entry"] == "run", stage
        assert [kind for kind, _ in record["arg_plan"]][-3:] == ["grid"] * 3, stage
        for relative in record["sources"]:
            assert (DSA / "csrc" / relative).is_file(), (stage, relative)
    assert len({record["name"] for record in registry.values()}) == len(registry)


def test_every_file_under_csrc_is_a_registered_source_or_the_host_shim():
    sources = {relative for record in _registry().values() for relative in record["sources"]}
    files = {str(path.relative_to(DSA / "csrc")) for path in (DSA / "csrc").rglob("*") if path.is_file()}

    assert files - sources == {"cake_host_shim.h"}
    assert sources <= files


def test_the_tcgen05_stages_keep_their_launch_contracts():
    registry = _registry()
    tc_args = (
        "q_latent q_rope kv_latent k_rope out o_lo lse indices topk_length sink num_queries num_kv topk idx_stride "
        "indices_offset k_rope_stride k_rope_offset has_topk_length derive_length scale_log2 heads"
    ).split()
    tc_bwd_args = (
        "q_latent q_rope dout dq_latent dq_rope kv_latent k_rope lse delta out o_lo indices topk_length dkv_f32 dkr_f32 "
        "dkv_stride dkr_stride dkr_col0 dkv_dst_map dkv_has_map num_queries num_kv topk idx_stride indices_offset "
        "has_topk_length token_base token_step scale_log2 sm_scale pass_lo pass_hi dq_mode dq_partial key_scratch "
        "pass_counts"
    ).split()
    indexer_tc_args = "index_q index_k weights cu_ks cu_ke logits seq_len seq_len_kv".split()
    indexer_tc_bwd_args = (
        "index_q index_k weights topk_indices grad_scores d_index_q d_weights partial heads topk".split()
    )
    for stages, args, block in (
        (TC_STAGES, tc_args, [384, 1, 1]),
        (TC_BWD_STAGES, tc_bwd_args, [640, 1, 1]),
        (INDEXER_TC_STAGES, indexer_tc_args, [320, 1, 1]),
        (INDEXER_TC_BWD_STAGES, indexer_tc_bwd_args, [256, 1, 1]),
    ):
        for stage in sorted(stages):
            record = registry[stage]
            assert _arg_names(record) == args, stage
            assert "workspace" not in {kind for kind, _ in record["arg_plan"]}, stage
            assert record["block"] == block and record["tma_workspace_bytes"] == 0, stage
            assert record["compile_flags"] == [], stage
    assert _arg_names(registry["dsa_indexer_bwd_h32"]) == indexer_tc_bwd_args
    for heads in (8, 16, 32, 64):
        assert registry[f"dsa_indexer_logits_h{heads}"]["block"] == [256, 1, 1]


def test_the_package_data_ships_the_registry_and_every_kernel_source():
    tomllib = pytest.importorskip("tomllib")
    setuptools = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["tool"]["setuptools"]
    assert any(
        fnmatch.fnmatch("miles_ext_kernels.dsa", pattern) for pattern in setuptools["packages"]["find"]["include"]
    )

    shipped = set()
    for pattern in setuptools["package-data"]["*"]:
        shipped.update(path.relative_to(DSA) for path in DSA.glob(pattern) if path.is_file())
    csrc = {path.relative_to(DSA) for path in (DSA / "csrc").rglob("*") if path.is_file()}

    assert sorted(map(str, ({Path("registry.json")} | csrc) - shipped)) == []


def _host_modules() -> list[Path]:
    return sorted(path for path in DSA.glob("*.py"))


@pytest.mark.parametrize("path", _host_modules(), ids=lambda path: path.name)
def test_the_host_modules_read_no_environment_and_import_no_miles(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert not (
                isinstance(node.value, ast.Name) and node.value.id == "os" and node.attr in ("environ", "getenv")
            )
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] not in ("miles", "flashinfer") for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            assert (node.module or "").split(".")[0] not in ("miles", "flashinfer"), node.module
