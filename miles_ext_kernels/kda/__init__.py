"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Deterministic packed KDA (Kimi Delta Attention) chunked training kernels for SM100a / SM103a / SM107a.

FlashInfer's Cake package ``cake_kda_det_train``, vendored under ``_vendor/``: forward and backward of the chunked
KDA recurrence (chunk 64, K = V = 128, 1 to 64 heads, per-document zero initial state) over one packed batch of
documents described by ``cu_seqlens``, with the forget gate activated inside the kernels (``use_gate_in_kernel``,
``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))``) and the q / k L2 normalisation inside the kernels
(``use_qk_l2norm_in_kernel``). The kernels reduce in a fixed order without atomics, so repeated passes on the same
inputs give bit-identical outputs and gradients. There is no non-deterministic variant.

miles selects it with ``--kda-backend cake`` under ``--use-miles-ext-kernels``. A layer calls :func:`prepare` with
its head count when it is built and :func:`chunk_kda` in its forward. Importing this package does not import torch:
the entry points load on first access.
"""

from importlib import import_module

# public name -> the submodule defining it, imported on first access
_LAZY = {
    "DEFAULT_SCALE": ".ops",
    "MAX_HEADS": ".ops",
    "chunk_kda": ".ops",
    "chunk_kda_bwd": ".ops",
    "chunk_kda_fwd": ".ops",
    "configure": ".arguments",
    "prepare": ".ops",
}

__all__ = ["DEFAULT_SCALE", "MAX_HEADS", "chunk_kda", "chunk_kda_bwd", "chunk_kda_fwd", "configure", "prepare"]


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
