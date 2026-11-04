"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Deterministic dense projection GEMMs for Blackwell (SM100a / SM103a), on the Cake programs of the vendored FlashInfer
package ``dense_projection_det_gemm``:

- :func:`batched_projection`: the per-head projections of the MLA / DSA attention layers (the q-absorb and value
  einsums, DeepSeek-V4's output-group projection) as one batched BF16 GEMM over strided views, with its input and
  weight gradients (the K1 programs: FP32 accumulation, fixed split-K and stream-K fixup order).
- :func:`router_gating`: the MoE router gate GEMM in FP32 as Megatron's ``TopKRouter.gating`` computes it with
  ``--moe-router-dtype fp32``: BF16 operands and FP32 logits forward (K1), FP32 input and weight gradients through
  the split-BF16x3 emulation (K2).

Both are bit-reproducible run to run; there is no non-deterministic variant (the package's ``fast`` sibling runs
the same programs at miles' shapes). miles selects them with ``--batched-projection-gemm-backend cake`` and
``--moe-router-gemm-backend cake`` under ``--use-miles-ext-kernels``. Importing this package does not import torch:
the entry points load on first access.
"""

from importlib import import_module

# public name -> the submodule defining it, imported on first access
_LAZY = {
    "SUPPORTED_CAPABILITIES": ".ops",
    "WEIGHT_LAYOUTS": ".ops",
    "batched_projection": ".ops",
    "check_projection_dims": ".ops",
    "prebuild_batched_projection": ".ops",
    "prebuild_router": ".ops",
    "router_gating": ".ops",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str):
    try:
        module = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted({*globals(), *__all__})
