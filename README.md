# miles-ext-kernels

Training kernels for [miles](https://github.com/radixark/miles), consumed as the `3rdparty/miles-ext-kernels`
submodule and enabled by miles' `--use-miles-ext-kernels`.

## Contract with miles

| Owned by miles | Owned by this repo |
|---|---|
| `--use-miles-ext-kernels` | kernel sources, JIT loader, host/ops glue |
| backend selection (`--linear-attention-backend cake`, ...) and the `cake` choice | every per-kernel argument (no `MILES_*` env knobs) |
| the dispatch seam: `if backend == "cake": from miles_ext_kernels.<kernel> import ...` | kernel tests (CPU and GPU) |
| `CAKE_BACKEND_ARGS` in `miles/utils/arguments.py`: `cake` without `--use-miles-ext-kernels` is rejected | |

Under `--use-miles-ext-kernels`, miles calls `miles_ext_kernels.add_arguments(parser)` before parsing and
`miles_ext_kernels.validate_args(args)` from `miles_validate_args`. Without the flag nothing here is imported.

## Kernels

| package | miles backend switch | arguments | GPUs | tests |
|---|---|---|---|---|
| `gdn` (chunked gated delta rule, forward + backward: `det` is the bit-deterministic render, the only variant with context parallelism; `fast` is the vendored Cake non-deterministic package `cake_gdn_train`, fixed-order, so it repeats bit for bit and `--deterministic-mode` allows an explicit `fast`; a `cp_context` call runs `det`) | `--linear-attention-backend cake` | `--cake-gdn-variant {det,fast}`, `--cake-gdn-recompute-state` | SM100a / SM103a / SM107a | `tests/gdn` (CPU; `test_gdn_wiring.py`, `test_fla_parity.py` and `test_gdn_fast_gpu.py` need CUDA) |
| `kda` (deterministic packed KDA chunked forward + backward: raw forget gate and q/k L2 norm in the kernels, 1-64 heads) | `--kda-backend cake` | none | SM100a / SM103a / SM107a | `tests/kda` (CPU; `test_kda_gpu.py` needs CUDA, its fla parity rows need fla) |
| `dsa` (DSA sparse-MLA attention forward + backward, lightning-indexer scores; variants det and fast) | `--dsa-attention-backend cake` | `--cake-dsa-variant {det,fast}`, `--cake-dsa-bwd-workspace-bytes`, `--cake-dsa-lazy-build` | SM100a / SM103a / SM107a | `tests/dsa` (CPU: `test_arguments.py`, `test_registry.py`; `test_dispatch.py` needs torch; `test_dsa_gpu.py` needs SM100a / SM103a / SM107a) |
| `mhc` (Cake four-stream mHC hyper connection: prepare, finish, stream-mean head; det and fast) | `--mhc-backend cake` | `--cake-mhc-variant {det,fast}` | SM100a / SM103a / SM107a (SM107a: hidden 4096 only) | `tests/mhc` (CPU: `test_arguments.py`, `test_vendored_routes.py`; `test_mhc_gpu.py` needs SM100a / SM103a / SM107a) |
| `gqa` (packed causal GQA attention, head dim 256, forward + backward; dQ is summed with FP32 atomics, so it is not bit-reproducible) | `--full-attention-backend cake` | none (the cake backend is rejected under `--deterministic-mode`) | SM100a / SM103a / SM107a | `tests/gqa` (CPU; `test_gqa_gpu.py` needs CUDA) |
| `conv1d` (packed causal depthwise conv1d + SiLU, kernel size 4: the GDN / KDA short convs; det + fast) | `--linear-attention-conv-backend cake` | `--cake-conv1d-variant {det,fast}`, `--cake-conv1d-silu-grad-preactivation {bf16,fp32}` | SM100a / SM103a / SM107a | `tests/conv1d` (CPU; `test_conv1d_gpu.py` needs CUDA) |
| `gated_rmsnorm` (RMSNorm(x) * w * SiLU(z) over rows of 128: the GDN output norm; det + fast) | `--linear-attention-output-norm-backend cake` | `--cake-gated-rmsnorm-variant {det,fast}` | SM100a / SM103a / SM107a | `tests/gated_rmsnorm` (CPU; `test_gated_rmsnorm_gpu.py` needs CUDA) |
| `rmsnorm` (zero-centred RMSNorm x * rstd * (1 + w): the GDN input layernorm, hidden 256/512/1536/2048/4096/5120/6144; det + fast) | `--linear-attention-input-norm cake` | `--cake-rmsnorm-variant {det,fast}` | SM100a / SM103a / SM107a | `tests/rmsnorm` (CPU; `test_rmsnorm_gpu.py` needs CUDA) |
| `dense_gemm` (deterministic batched per-head projection GEMM and FP32 router gate GEMM, forward + backward) | `--batched-projection-gemm-backend cake`, `--moe-router-gemm-backend cake` (Megatron) | none | SM100a / SM103a / SM107a | `tests/dense_gemm` (CPU; `test_batched_projection_gpu.py` and `test_router_gpu.py` need CUDA) |
| `moe_grouped_gemm` (ragged BF16 MoE grouped GEMM, `det` and `fast`, forward + backward) | `--moe-grouped-gemm-backend cake` (torchtitan) | `--cake-moe-grouped-gemm-variant {det,fast}` | SM100a / SM103a / SM107a | `tests/moe_grouped_gemm` (CPU; `test_moe_grouped_gemm_gpu.py` needs CUDA) |

A package that ships two variants takes `--cake-<kernel>-variant {det,fast}`: `det` is bit-deterministic, `fast` is
faster. Unset, it runs `fast`, or `det` under miles' `--deterministic-mode` (`miles_ext_kernels/_variant.py`). The
`fast` variants of `dsa`, `mhc`, `conv1d`, `gated_rmsnorm`, `rmsnorm` and `moe_grouped_gemm` reduce in a
run-dependent order, so they are not bit-reproducible and `--deterministic-mode` rejects an explicit `fast`. The `gdn`
`fast` runs in a fixed order: it repeats bit for bit run to run (its bits differ from `det`), so `--deterministic-mode`
allows it. `kda` and `dense_gemm` ship `det` only; `gqa` ships one variant, not bit-reproducible, which
`--deterministic-mode` rejects. A call that `fast` does not serve runs `det` and logs it once.

A package applies its parsed arguments in `configure(args)`; miles calls it where it builds a layer on the `cake`
backend (`miles_ext_kernels.gdn.configure`). miles keeps the values of the arguments registered here in one typed
field of its configuration and passes `configure` a namespace holding just those values.

## Vendored Cake packages

Most kernels are FlashInfer's Cake packages, vendored byte-identical under `miles_ext_kernels/<kernel>/_vendor/` by
`tools/vendor_cake.py` (never edited in place: re-vendor instead). Each `_vendor/VENDORED.json` records the provenance:
the FlashInfer repository and commit, and per package its FlashInfer path, Cake revision and the git blob of every
file. `tests/cake_native/test_vendored.py` checks every vendored tree against it; its `VENDORED_FAMILIES` names the
families that vendor, so the commit that vendors a family adds it there.

`miles_ext_kernels.cake_native` builds a package's libraries from its `build.json` with nvcc and loads them with
tvm-ffi (`apache-tvm-ffi>=0.1.11,<0.2`); FlashInfer itself is not needed. Libraries are cached under
`${TVM_FFI_CACHE_DIR:-~/.cache/tvm-ffi}/miles_ext_kernels/`, never in the package directory.

SM107a needs a CUDA 13.4 or newer nvcc (`cake_native.MIN_CUDA`): a CUDA 13.0 toolkit builds the SM100a and SM103a
kernels only, and an SM107a build under it fails with an error naming the toolkit it needs. The generated kernels
built as torch extensions follow the same rule with torch's nvcc (`CUDA_HOME`).

One rank builds each library under a file lock next to it; the others wait, log every 30 s and fail after
30 minutes. On a filesystem whose locks are cluster-wide, such as Lustre, a rank killed on another node can leave
its lock held for minutes. For multi-node jobs, prefer a node-local or per-job `TVM_FFI_CACHE_DIR`; each such
directory builds the libraries again.

## Adding a kernel

1. PR here: the package under `miles_ext_kernels/<kernel>/`, its arguments in `add_arguments` / `validate_args`
   (registered in `_KERNEL_PACKAGES`), and its tests.
2. PR in miles: bump the submodule, add `cake` to the backend's choices, add the dispatch seam, and add the
   backend argument to `CAKE_BACKEND_ARGS`. GPU CI runs there.

## Development

```bash
pip install -e .
pre-commit run --all-files
pytest tests
python tools/vendor_cake.py --check
```
