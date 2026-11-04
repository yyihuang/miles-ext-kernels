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

## Vendored Cake packages

Most kernels are FlashInfer's Cake packages, vendored byte-identical under `miles_ext_kernels/<kernel>/_vendor/` by
`tools/vendor_cake.py` (never edited in place: re-vendor instead). Each `_vendor/VENDORED.json` records the provenance:
the FlashInfer repository and commit, and per package its FlashInfer path, Cake revision and the git blob of every
file. `tests/cake_native/test_vendored.py` checks every vendored tree against it; its `VENDORED_FAMILIES` names the
families that vendor, so the commit that vendors a family adds it there.

`miles_ext_kernels.cake_native` builds a package's libraries from its `build.json` with nvcc and loads them with
tvm-ffi (`apache-tvm-ffi>=0.1.11,<0.2`); FlashInfer itself is not needed. Libraries are cached under
`${TVM_FFI_CACHE_DIR:-~/.cache/tvm-ffi}/miles_ext_kernels/`, never in the package directory.

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
