"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Selected-token log-probabilities of an LM head on the vendored Cake chunked LM-head kernels (SM100a / SM103a).

``logp_t = z[t, y_t] / tau_t - logsumexp_v(z[t, v] / tau_t)`` with ``z = bf16(X @ W^T)``, computed chunk by chunk:
the forward keeps the FP32 log-sum-exp per row, the backward recomputes each chunk's logits and forms ``dX`` and
``dW`` through FP32 accumulators with one BF16 cast each. No vocabulary-sized buffer spans more than one chunk.

Two packages, routed here as FlashInfer's ``flashinfer.chunked_lm_head.chunked_lm_head_logprob`` routes them (that
facade is not vendored; its environment-variable knobs stay at the packages' defaults):

* ``det``: ``_vendor/cake_lm_head_det_loss`` (FlashInfer ``flashinfer/experimental/cake_lm_head_det_loss``). Fixed
  chunk order, no atomics: ``logp``, ``dX`` and ``dW`` are bit-reproducible run to run for a fixed chunk size.
* ``fast``: ``_vendor/cake/cake_lm_head_loss`` (FlashInfer ``flashinfer/experimental/cake/cake_lm_head_loss``). The
  same ``logp`` bits as ``det``; the dX GEMM of a K-sliced chunk adds its slices into the FP32 accumulator in an
  unordered reduction, so ``dX`` may differ run to run by FP32 rounding; ``dW`` keeps its order (``det``'s bits unless
  a valid row has a zero incoming gradient); a backward of more than one default chunk skips the GEMM work of the rows
  whose incoming gradient is zero (PPO-clipped or masked tokens). While ``torch.use_deterministic_algorithms(True)`` is
  in effect, a ``fast`` call runs ``det`` and says so once.

Each package's libraries are built from its ``build.json`` by :mod:`miles_ext_kernels.cake_native` on first use
(:func:`prebuild` builds them ahead, in parallel).
"""

from __future__ import annotations

import functools
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

import torch

from miles_ext_kernels import cake_native
from miles_ext_kernels._variant import VARIANTS, warn_det_fallback

__all__ = ["CHUNK_SIZE", "SUPPORTED_CAPABILITIES", "VARIANT", "device_arch", "log_probs", "prebuild"]

# compute capability -> the architecture of the vendored programs this package runs
SUPPORTED_CAPABILITIES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}
# The defaults of a call that names neither, read at call time: miles_ext_kernels.lm_head.configure sets them from
# --cake-lm-head-variant / --cake-lm-head-chunk-size.
VARIANT = "fast"
CHUNK_SIZE = 4096


class _Package(NamedTuple):
    cake_build: ModuleType  # the package's loader, building through cake_native
    cake_jit: ModuleType  # its registry of programs and stages
    entry: ModuleType  # cake_backend_opt_in: the log-probability entries, with and without num_valid_rows


@functools.cache
def _package(variant: str) -> _Package:
    """The vendored package of ``variant``, imported once with :mod:`cake_native` installed as its build toolchain."""
    if variant == "det":
        from ._vendor.cake_lm_head_det_loss import cake_build

        cake_native.install(cake_build)
        from ._vendor.cake_lm_head_det_loss import cake_backend_opt_in, cake_jit
    elif variant == "fast":
        from ._vendor.cake.cake_lm_head_loss import cake_build

        cake_native.install(cake_build)
        from ._vendor.cake.cake_lm_head_loss import cake_backend_opt_in, cake_jit
    else:
        raise ValueError(f"variant must be one of {VARIANTS}, not {variant!r}")
    return _Package(cake_build, cake_jit, cake_backend_opt_in)


def device_arch(device: torch.device | int | str | None = None) -> str:
    """The architecture tag of ``device`` (default: the current CUDA device); raises where the kernels do not run."""
    device = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
    if device.type != "cuda":
        raise RuntimeError(f"the Cake LM-head kernels run on a CUDA device, got {device}")
    major, minor = torch.cuda.get_device_capability(device)
    arch = SUPPORTED_CAPABILITIES.get((major, minor))
    if arch is None:
        raise RuntimeError(
            "the Cake LM-head kernels need compute capability 10.0 (B200 / GB200) or 10.3 (B300 / GB300); "
            f"{device} has {major}.{minor}"
        )
    return arch


def _runs(variant: str | None, *, warn: bool) -> str:
    """The variant a call of ``variant`` (``None``: :data:`VARIANT`) runs: ``fast`` runs ``det`` while
    ``torch.use_deterministic_algorithms(True)`` is in effect (said once with ``warn``)."""
    variant = VARIANT if variant is None else variant
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, not {variant!r}")
    if variant == "fast" and torch.are_deterministic_algorithms_enabled():
        if warn:
            warn_det_fallback("lm_head", "torch.use_deterministic_algorithms(True)")
        return "det"
    return variant


def log_probs(
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float | torch.Tensor | None = None,
    variant: str | None = None,
    chunk_size: int | None = None,
    num_valid_rows: int | None = None,
) -> torch.Tensor:
    """Differentiable FP32 ``logp [T]`` of ``labels`` under ``softmax(X @ W^T / temperature)``.

    Args:
        X: BF16 ``[T, H]`` hidden rows, row-major with a 16-byte row pitch (otherwise copied); ``H`` a multiple of 512.
        W: BF16 ``[V, H]`` output weight over the vocabulary the softmax runs over, contiguous; ``V`` a multiple of
            256. ``dW`` reaches it as BF16.
        labels: int64 ``[T]`` target ids; ``-100`` marks an ignored row (``logp`` 0, no gradient).
        temperature: ``None`` (or a number whose FP32 value is 1.0) for none; a positive finite number; or an FP32
            ``[T]`` tensor on ``X``'s device, positive and finite on every row (checked by a device-side assertion).
            Not differentiable.
        variant: ``"det"`` or ``"fast"`` (module docstring); ``None``: :data:`VARIANT`.
        chunk_size: token rows per chunk, at most 65535; ``None``: :data:`CHUNK_SIZE`. ``det``'s bits depend on it.
        num_valid_rows: the host count of ``labels >= 0`` when the caller knows it (every row valid: ``T``), which
            spares the device-to-host read of the count. It is checked on the device: a wrong count stops the stream
            with a device-side assertion. Bitwise the same outputs as without it.

    Raises ``RuntimeError`` off SM100a / SM103a and ``ValueError`` for shapes, dtypes or values the kernels do not
    serve. A call of ``T == 0`` rows returns an empty ``logp`` (zero ``dW``).
    """
    device_arch(X.device)
    variant = _runs(variant, warn=True)
    chunk_size = CHUNK_SIZE if chunk_size is None else chunk_size
    entry = _package(variant).entry
    if variant == "det":
        return entry.chunked_lm_head_det_logprob_temperature(
            X, W, labels, chunk_size=chunk_size, num_valid_rows=num_valid_rows, temperature=temperature
        )
    return entry.chunked_lm_head_logprob_temperature(
        X,
        W,
        labels,
        chunk_size=chunk_size,
        deterministic=False,
        num_valid_rows=num_valid_rows,
        temperature=temperature,
    )


def prebuild(
    device: torch.device | int | str | None = None,
    *,
    variant: str | None = None,
    temperature: bool = False,
    max_workers: int | None = None,
) -> list[Path]:
    """Build the libraries :func:`log_probs` of ``variant`` loads on ``device``'s architecture before the first call,
    ``max_workers`` compiles at a time; with ``temperature`` also the temperature forms. Returns the libraries it
    built (cached ones are skipped). Call it where the ``cake`` log-prob path is set up, so no rank compiles inside a
    training step.
    """
    arch = device_arch(device)
    package = _package(_runs(variant, warn=False))
    jit = package.cake_jit
    record = jit.select_module(arch)
    stages = jit.registered_stages_with_temperature(record) if temperature else jit.registered_stages(record)
    return cake_native.prebuild(package.cake_build, [jit.library_name(record, stage) for stage in stages], max_workers)
