"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Deterministic packed KDA chunked training on the vendored Cake package ``cake_kda_det_train`` (SM100a / SM103a /
SM107a).

:func:`chunk_kda`, :func:`chunk_kda_fwd` and :func:`chunk_kda_bwd` are FlashInfer's thin entry points
(``flashinfer/cake_kda_det_train.py``: ``chunk_kda_det``, ``chunk_kda_det_fwd``, ``chunk_kda_det_bwd``) without its
API logging: they default ``scale`` to ``128 ** -0.5`` and forward every keyword to the package's ``cake_backend``,
which checks the operands (strided views are refused, never copied, except the raw gate and the raw q / k under the
in-kernel L2 norm), plans the launches from the host scalars alone and runs them on the current stream.

Operands, one packed batch of ``S`` tokens in ``num_docs`` documents with ``H`` heads (1 to 64): ``q, k, v`` BF16
``[1, S, H, 128]`` (the batch axis is optional; outputs take the rank of the inputs), ``g`` the gate ``[1, S, H,
128]``, ``beta`` BF16 ``[1, S, H]`` after the sigmoid, ``cu_seqlens`` int32 ``[num_docs + 1]`` on the device. miles
calls the kernels in two modes at once:

* ``use_gate_in_kernel=True``: ``g`` is the raw BF16 forget gate, which the kernels activate as ``lower_bound *
  sigmoid(exp(A_log) * (g + dt_bias))`` with ``A_log`` FP32 ``[H]``, ``dt_bias`` FP32 ``[H * 128]`` and ``-5 <=
  lower_bound < 0``; the gradients reach ``g``, ``A_log`` and ``dt_bias``.
* ``use_qk_l2norm_in_kernel=True``: ``q`` / ``k`` are the raw rows, L2-normalised per token and head inside the
  kernels (eps ``1e-6``); they may be column slices of one packed projection (head stride 128, token stride a
  multiple of 8 elements, 16-byte aligned). ``dq`` / ``dk`` are the raw rows' gradients.

Every result is bit-identical across repeated calls on the same inputs, and the backward from the forward's saved
set, the checkpoint replay (``saved=None``) and the autograd node (under ``torch.utils.checkpoint`` too) give the
same gradient bits.

The kernels are compiled on first use from the package's ``build.json`` by :mod:`miles_ext_kernels.cake_native` and
cached under ``$TVM_FFI_CACHE_DIR``. :func:`prepare`, called when a layer is built, checks the head count and the
device and compiles the libraries that layer's calls launch, so no library compiles inside a training step.
"""

from __future__ import annotations

import functools
import logging

import torch

from .. import cake_native
from ._vendor.cake_kda_det_train import cake_backend, cake_build, cake_jit

__all__ = [
    "DEFAULT_SCALE",
    "MAX_HEADS",
    "chunk_kda",
    "chunk_kda_bwd",
    "chunk_kda_fwd",
    "device_arch",
    "prepare",
    "route_libraries",
]

cake_native.install(cake_build)

logger = logging.getLogger(__name__)

# The default scale of the kernels' contract: head_dim ** -0.5 with head dimension 128.
DEFAULT_SCALE = 128**-0.5
# The head counts the kernels are validated for (per call, so per tensor-parallel rank).
MAX_HEADS = 64
# The paths the autograd node launches: the forward that saves its set, and the backward from that set without and
# with the segmented route's saved segment states.
_AUTOGRAD_PATHS = ("fwd", "light", "light_states")
# The largest document count a selection rule of the registry names (the chunk-parallel layout table's 256).
_MAX_DOCS_THRESHOLD = 256


def _check_heads(num_heads: int) -> None:
    if not 1 <= num_heads <= MAX_HEADS:
        raise ValueError(
            f"the Cake KDA kernels serve 1 to {MAX_HEADS} heads per call (per tensor-parallel rank of a head-sharded "
            f"layer), got {num_heads}: shard the heads over more ranks with --tensor-model-parallel-size, or use "
            "--kda-backend fla"
        )


def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    scale: float | None = None,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> torch.Tensor:
    """Packed KDA chunked training as an autograd node: ``out`` BF16 like ``v``, with gradients to ``q, k, v, g,
    beta`` and, with ``use_gate_in_kernel=True``, to ``A_log`` / ``dt_bias``. Compatible with
    ``torch.utils.checkpoint``. See the module docstring for the operands."""
    _check_heads(int(v.shape[-2]))
    return cake_backend.chunk_kda_det(
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        scale=DEFAULT_SCALE if scale is None else scale,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


def chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    scale: float | None = None,
    save_for_backward: bool = True,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, object | None]:
    """The forward alone: ``(out, saved)``, ``saved`` the set :func:`chunk_kda_bwd` takes (``None`` without
    ``save_for_backward``)."""
    _check_heads(int(v.shape[-2]))
    return cake_backend.chunk_kda_det_fwd(
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        scale=DEFAULT_SCALE if scale is None else scale,
        save_for_backward=save_for_backward,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


def chunk_kda_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    dout: torch.Tensor,
    *,
    saved: object | None = None,
    scale: float | None = None,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, ...]:
    """The backward alone, from the forward's operands and modes: ``(dq, dk, dv, dbeta, dg)``, and ``(dq, dk, dv,
    dbeta, dg, dA_log, ddt_bias)`` with ``use_gate_in_kernel=True``. ``saved=None`` replays the forward (checkpoint
    replay); the gradients are the same bits either way."""
    _check_heads(int(v.shape[-2]))
    return cake_backend.chunk_kda_det_bwd(
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        dout,
        saved=saved,
        scale=DEFAULT_SCALE if scale is None else scale,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


def _device_index(device: torch.device | str | int | None) -> int:
    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, int):
        return device
    device = torch.device(device)
    return torch.cuda.current_device() if device.index is None else device.index


@functools.cache
def _min_cuda(arch: str) -> tuple[int, int]:
    """The newest ``min_cuda`` of the package's ``arch`` libraries: the toolkit that builds all of them."""
    versions = [
        tuple(int(part) for part in str(entry["min_cuda"]).split(".")[:2])
        for entry in cake_build.manifest()["libraries"]
        if entry["arch"] == arch
    ]
    return max(versions)


def device_arch(device: torch.device | str | int | None = None) -> str:
    """The package architecture of ``device`` (the current CUDA device by default), checked against the toolkit.

    ``RuntimeError`` for a device the package has no kernels for, or whose kernels the local nvcc cannot build
    (``sm_107a`` needs CUDA 13.4).
    """
    index = _device_index(device)
    capability = tuple(torch.cuda.get_device_capability(index))
    arch = cake_jit.arch_for_capability(capability)
    if arch is None:
        raise RuntimeError(
            f"the Cake KDA kernels need a GPU of compute capability 10.0, 10.3 or 10.7 (SM100a / SM103a / SM107a; "
            f"SM107a with a CUDA 13.4 or newer toolkit); device {index} has {capability[0]}.{capability[1]}"
        )
    need = _min_cuda(arch)
    cuda = cake_native.get_cuda_version()
    if (cuda.major, cuda.minor) < need:
        raise RuntimeError(
            f"the Cake KDA kernels of {arch} (device {index}, compute capability {capability[0]}.{capability[1]}) "
            f"build with CUDA {need[0]}.{need[1]} or newer; {cake_native.find_nvcc()} is CUDA {cuda.full}"
        )
    return arch


def _probe_doc_counts(num_heads: int, sm_count: int) -> list[int]:
    """Document counts whose plans together select every record a pack of any document count selects.

    The selection rules read the document count through the documents, the (document, head) chains, the forms
    that fit one wave of the device, the segmented route's segments and its carry. Above ``4 * sm_count / H``
    documents every document is one segment, and the registry's rules then pass one more threshold (8192
    segment-heads), so every count up to there is probed, then doubling counts. A record a later registry selects
    between two probes is not missed silently: it compiles on its first launch.
    """
    dense = max(_MAX_DOCS_THRESHOLD + 1, -(-4 * sm_count // num_heads) + 2)
    counts = list(range(1, dense + 1))
    while counts[-1] < cake_backend.MAX_DOCS:
        counts.append(min(2 * counts[-1], cake_backend.MAX_DOCS))
    return counts


@functools.cache
def _route_records(num_heads: int, arch: str, sm_count: int, gate: str, l2n: str) -> frozenset[str]:
    records: set[str] = set()
    for num_docs in _probe_doc_counts(num_heads, sm_count):
        try:
            plan = cake_backend.plan(num_docs, num_docs, num_heads, sm_count, arch, gate=gate, l2n=l2n)
        except ValueError:
            # so many documents overflow the kernels' int32 row offsets at this head count: no larger pack runs
            break
        for path in _AUTOGRAD_PATHS:
            records.update(name for name in plan.records.get(path, {}).values() if name is not None)
    return frozenset(records)


def route_libraries(
    num_heads: int, arch: str, sm_count: int, *, gate: str = "raw", l2n: str = "fused"
) -> tuple[str, ...]:
    """The ``build.json`` libraries the autograd node launches for ``num_heads`` heads on a device of ``arch`` with
    ``sm_count`` SMs, over every document count, in the gate mode ``gate`` (``"raw"``: ``use_gate_in_kernel=True``)
    and the q / k L2 norm mode ``l2n`` (``"fused"``: ``use_qk_l2norm_in_kernel=True``)."""
    records = _route_records(int(num_heads), arch, int(sm_count), gate, l2n)
    return tuple(sorted({cake_jit.library_name(name, arch) for name in records}))


@functools.cache
def _prepare(num_heads: int, arch: str, sm_count: int, max_workers: int | None) -> None:
    records = _route_records(num_heads, arch, sm_count, "raw", "fused")
    names = route_libraries(num_heads, arch, sm_count)
    cake_native.prebuild(cake_build, names, max_workers)
    for record in sorted(records):
        cake_jit.load(record, arch)
    logger.info("miles_ext_kernels.kda: %d %s libraries ready for %d heads", len(names), arch, num_heads)


def prepare(num_heads: int, device: torch.device | str | int | None = None, *, max_workers: int | None = None) -> None:
    """Check that a layer of ``num_heads`` heads per rank can run on ``device`` and get its kernels ready.

    ``ValueError`` outside 1 to :data:`MAX_HEADS` heads; ``RuntimeError`` for a device or toolkit the kernels do not
    serve (:func:`device_arch`). Then compiles, ``max_workers`` at a time, and loads every library the calls of
    miles (raw gate, in-kernel q / k L2 norm) launch for that head count on that device, over every document count
    (:func:`route_libraries`). Once per head count and device kind per process.
    """
    num_heads = int(num_heads)
    _check_heads(num_heads)
    index = _device_index(device)
    arch = device_arch(index)
    _prepare(num_heads, arch, cake_jit.sm_count(index), max_workers)
