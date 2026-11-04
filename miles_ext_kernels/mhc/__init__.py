"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Cake four-stream mHC (manifold-constrained hyper-connection) training kernels for Blackwell (SM100a / SM103a).

One connection of a layer is ``U, handle = mhc_prepare(X, fn, scale, base)`` (the RMS-normalised mapping GEMM, the
pre / post / Sinkhorn coefficients and the stream collapse ``U``), the layer's branch on ``U``, then
``Y = mhc_finish(X, branch_output, handle)`` (the residual mixing). Both are autograd functions; the backward of the
pair returns one single-cast ``dX`` and FP32 ``dfn``, ``dscale``, ``dbase``. ``mhc_head(X)`` is the model head's
stream mean, bit for bit torch's BF16 ``X.mean(dim=1)`` and its gradient.

``--cake-mhc-variant`` selects the backward: ``det`` sums the parameter gradients in a fixed order (bitwise
reproducible for a token count on a device), ``fast`` adds the ``dfn`` partials with FP32 atomics (run-to-run
varying low-order bits of ``dfn``; ``dX`` and the forward are the ``det`` bits). The head has one variant.

miles selects these kernels with ``--mhc-backend cake`` under ``--use-miles-ext-kernels`` and hands the arguments of
:mod:`.arguments` to :func:`configure`. Importing this package does not import torch: the entry points load on first
access.
"""

from importlib import import_module

# public name -> the submodule defining it, imported on first access
_LAZY = {
    "HIDDEN_SIZES": ".arguments",
    "STREAMS": ".arguments",
    "SUPPORTED_CAPABILITIES": ".ops",
    "configure": ".arguments",
    "is_supported": ".ops",
    "mhc_finish": ".ops",
    "mhc_head": ".ops",
    "mhc_prepare": ".ops",
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
