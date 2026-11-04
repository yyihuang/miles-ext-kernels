"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Packed causal GQA attention training kernels, head dimension 256, for Blackwell (SM100a / SM103a).

The Cake kernels of FlashInfer's ``flashinfer.gqa_train``, vendored under ``_vendor/cake_gqa_train``: forward and
backward over packed variable-length segments, BF16 ``q [T, H_q, 256]`` and ``k`` / ``v [T, H_kv, 256]`` with
``H_q / H_kv`` even, causal with FlashAttention's bottom-right alignment per segment. The output and the dK / dV are
bit-identical run to run; dQ is summed with FP32 atomic adds, so it is not, and deterministic mode rejects the kernels.

miles selects them with ``--full-attention-backend cake`` under ``--use-miles-ext-kernels`` for the full-attention
layers of the Qwen3.5 / Qwen3-Next models. Importing this package does not import torch: the entry points load on
first access.
"""

from importlib import import_module

# public name -> the submodule defining it, imported on first access
_LAZY = {
    "SUPPORTED_CAPABILITIES": ".ops",
    "device_arch": ".ops",
    "prebuild": ".ops",
    "varlen_gqa_attention": ".ops",
}

__all__ = ["SUPPORTED_CAPABILITIES", "device_arch", "prebuild", "varlen_gqa_attention"]


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
