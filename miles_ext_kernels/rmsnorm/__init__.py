"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Fused zero-centred RMSNorm training kernels for SM100a / SM103a / SM107a: ``x * rstd * (1 + w)``, the input
layernorm of miles' GDN linear-attention layers.

:class:`ZeroCenteredRMSNorm` holds the single ``weight`` parameter (w of the ``(1 + w)`` scale, as Transformer
Engine's ``zero_centered_gamma`` norm and HF's ``Qwen3NextRMSNorm`` hold it) and runs FlashInfer's Cake RMSNorm
packages, vendored under ``_vendor/``, on the FP32 effective weight ``1 + w``. Two variants: ``det`` reduces the
weight gradient in a fixed order (bitwise reproducible for a given token count and SM count); ``fast`` uses a
single-kernel backward with FP64 atomic weight-gradient sums in the token windows that ship it. miles selects them
with ``--linear-attention-input-norm cake`` under ``--use-miles-ext-kernels`` and hands the arguments of
:mod:`.arguments` to :func:`configure`. Importing this package does not import torch: the entry points load on first
access.
"""

from importlib import import_module

# public name -> the submodule defining it, imported on first access
_LAZY = {
    "SUPPORTED_CAPABILITIES": ".ops",
    "ZeroCenteredRMSNorm": ".ops",
    "check_supported": ".ops",
    "configure": ".arguments",
    "prebuild": ".ops",
    "supported_hidden_sizes": ".ops",
    "zero_centered_rmsnorm": ".ops",
}

__all__ = [
    "SUPPORTED_CAPABILITIES",
    "ZeroCenteredRMSNorm",
    "check_supported",
    "configure",
    "prebuild",
    "supported_hidden_sizes",
    "zero_centered_rmsnorm",
]


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
