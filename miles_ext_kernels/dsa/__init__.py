"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

DSA (DeepSeek Sparse Attention) training kernels for Blackwell (SM100a / SM103a).

One path for the GLM-5 / DeepSeek-V3.2 / GLM-5.3 (``thd``, absorbed MLA, no sink) and DeepSeek-V4 / V4.1 (``bshd``,
MQA, batched, compressed KV, FP32 attention sink) lightning indexer and sparse attention, forward and backward:

* ``det`` -- generated register-MMA and tcgen05 kernels without floating-point atomics: key-side gradients are reduced
  from per-slot FP32 partials in an order fixed by the indices, so repeated passes on identical inputs are
  bit-identical. Every head count of 8 to 64, with or without a sink; the indexer scores are always ``det``.
* ``fast`` -- FlashInfer's Cake DSA training kernels for 64 local query heads without a sink: the same forward
  numerics class, a bit-reproducible ``dq`` and a ``dkv`` accumulated with FP32 atomics.

miles selects them with ``--miles-dsa-kernel-backend cake`` under ``--use-miles-ext-kernels`` and hands the arguments
of :mod:`.arguments` (``--cake-dsa-variant`` among them) to :func:`configure`. Importing this package does not import
torch: the entry points load on first access. See :mod:`.ops` for the operators.
"""

from importlib import import_module

# public name -> the submodule defining it, imported on first access
_LAZY = {
    "SUPPORTED_CAPABILITIES": "._jit",
    "check_device": ".ops",
    "configure": ".arguments",
    "device_arch": "._jit",
    "extract_topk_scores": ".ops",
    "fast_unserved_reason": ".ops",
    "indexer_backward": ".ops",
    "indexer_logits": ".ops",
    "indexer_topk_scores": ".ops",
    "lightning_indexer": ".ops",
    "prebuild": ".ops",
    "select_topk": ".ops",
    "sparse_attention": ".ops",
    "sparse_attention_backward": ".ops",
    "sparse_attention_forward": ".ops",
    "torch_topk": ".ops",
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
