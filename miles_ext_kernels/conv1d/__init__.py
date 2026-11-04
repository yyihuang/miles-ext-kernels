"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Packed causal depthwise conv1d + SiLU (kernel size 4) training kernels for Blackwell (SM100a / SM103a): the short
convolutions of miles' GDN and KDA linear-attention layers.

Two variants, both FlashInfer Cake packages vendored under ``_vendor/``: ``det`` reduces the weight gradient in a
fixed order (bitwise reproducible for a given token count, channel count and device); ``fast`` sums it in one
backward launch with unordered reductions (``y`` and ``dx`` are ``det``'s bits, the weight gradient stays within
``det``'s error budget). miles selects them with ``--linear-attention-conv-backend cake`` under
``--use-miles-ext-kernels`` and hands the arguments of :mod:`.arguments` to :func:`configure`. Importing this package
does not import torch: the entry points load on first access.
"""

from importlib import import_module

# public name -> the submodule defining it, imported on first access
_LAZY = {
    "SUPPORTED_CAPABILITIES": ".ops",
    "causal_conv1d": ".ops",
    "check_supported": ".ops",
    "configure": ".arguments",
    "prebuild": ".ops",
}

__all__ = ["SUPPORTED_CAPABILITIES", "causal_conv1d", "check_supported", "configure", "prebuild"]


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
