"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Chunked LM-head selected-token log-probabilities for Blackwell (SM100a / SM103a): the Cake kernels of FlashInfer's
chunked LM-head training packages, vendored under ``_vendor/``.

:func:`log_probs` takes the hidden rows ``X [T, H]``, the output weight ``W [V, H]`` (both BF16) and the target ids
``labels [T]`` and returns the FP32 ``logp [T]`` of ``softmax(X @ W^T / temperature)`` at the labels, differentiable
in ``X`` and ``W``. No ``[T, V]`` logits tensor exists beyond one token chunk: the backward recomputes each chunk's
logits. Two variants: ``det`` is bit-reproducible run to run; ``fast`` sums the K slices of the hidden-state
gradient in an unordered FP32 reduction and skips the GEMM work of rows whose incoming gradient is zero.

miles selects it with ``--log-probs-backend cake`` under ``--use-miles-ext-kernels`` and hands the arguments of
:mod:`.arguments` to :func:`configure`. Importing this package does not import torch: the entry points load on first
access.
"""

from importlib import import_module

# public name -> the submodule defining it, imported on first access
_LAZY = {
    "SUPPORTED_CAPABILITIES": ".ops",
    "configure": ".arguments",
    "device_arch": ".ops",
    "log_probs": ".ops",
    "prebuild": ".ops",
}

__all__ = ["SUPPORTED_CAPABILITIES", "configure", "device_arch", "log_probs", "prebuild"]


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
