"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Cake ragged BF16 MoE grouped GEMM for Blackwell (SM100a / SM103a): the expert projections of a routed MoE layer,
``Y[offs[e-1]:offs[e]] = X[offs[e-1]:offs[e]] @ W[e].T``, with the input and weight gradients, as
``torch._grouped_mm`` computes them.

Two variants. ``det`` is bit-deterministic: every reduction runs in a fixed order, the forward, input and weight
gradients are bit-identical call to call. ``fast`` runs the same forward and input-gradient programs (the same bits)
and a weight gradient that, where it picks the 512-wide k tile, adds the split tail of its last wave with unordered
FP32 reductions. :func:`grouped_mm` runs ``det`` under ``torch.use_deterministic_algorithms(True)``.

miles selects it with ``--moe-grouped-gemm-backend cake`` under ``--train-backend torchtitan`` and
``--use-miles-ext-kernels``, the variant with ``--cake-moe-grouped-gemm-variant``, and hands the arguments of
:mod:`.arguments` to :func:`configure`. Importing this package does not import torch: the entry points load on first
access.
"""

from importlib import import_module

# public name -> the submodule defining it, imported on first access
_LAZY = {
    "SUPPORTED_CAPABILITIES": ".ops",
    "check_grouped_dims": ".ops",
    "configure": ".arguments",
    "grouped_mm": ".ops",
    "prebuild": ".ops",
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
