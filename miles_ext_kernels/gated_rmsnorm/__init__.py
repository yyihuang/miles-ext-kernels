"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

SiLU-gated RMSNorm training kernels for SM100a / SM103a / SM107a: ``RMSNorm(x) * weight * SiLU(z)`` over rows of
128, the output norm of miles' GDN linear-attention layers.

Two variants, both FlashInfer Cake packages vendored under ``_vendor/``: ``det`` sums the weight gradient in FP32 per
chunk and FP64 across chunks in a fixed order (bitwise reproducible); ``fast`` runs a one-kernel backward that adds
the chunks with FP64 atomics (``dx`` and ``dz`` are ``det``'s bits, the weight gradient within ``det``'s bound). miles
selects them with ``--linear-attention-output-norm-backend cake`` under ``--use-miles-ext-kernels`` and hands the
arguments of :mod:`.arguments` to :func:`configure`. Importing this package does not import torch: the entry points
load on first access.
"""

from importlib import import_module

# public name -> the submodule defining it, imported on first access
_LAZY = {
    "SUPPORTED_CAPABILITIES": ".ops",
    "check_supported": ".ops",
    "configure": ".arguments",
    "prebuild": ".ops",
    "rmsnorm_silu_gate": ".ops",
}

__all__ = ["SUPPORTED_CAPABILITIES", "check_supported", "configure", "prebuild", "rmsnorm_silu_gate"]


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
