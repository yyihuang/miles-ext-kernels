"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Arguments of the chunked LM-head log-probability kernels (``--log-probs-backend cake``).

Registered by ``miles_ext_kernels.add_arguments`` under ``--use-miles-ext-kernels``. :func:`validate_args` reads
miles' parsed flat namespace (the miles, Megatron and LoRA arguments next to these) and rejects at startup what the
kernels do not serve. miles hands the parsed values to :func:`configure` where it sets up the ``cake`` log-prob
path. Torch-free: :func:`configure` imports the kernels lazily.
"""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Iterator
from importlib import import_module

from miles_ext_kernels import _variant

BACKEND = "--log-probs-backend cake"
VARIANT_FLAG = "--cake-lm-head-variant"
CHUNK_SIZE_FLAG = "--cake-lm-head-chunk-size"
DEFAULT_CHUNK_SIZE = 4096
# the row kernels index a chunk's row by blockIdx.y
MAX_CHUNK_SIZE = 65535
# the losses whose scores are the log-probabilities of the target tokens alone
LOSS_TYPES = ("policy_loss", "sft_loss")
# the GEMM work items: V in 256-column logits tiles, H in 512-column dX / dW tiles
VOCAB_MULTIPLE = 256
HIDDEN_MULTIPLE = 512
# ``--target-modules`` names of the LoRA targets that adapt the output layer: the group and the HF / Megatron leaves
_UNEMBED_TARGETS = frozenset({"unembed", "lm_head", "head", "output_layer"})


def add_arguments(group) -> None:
    _variant.add_variant_argument(group, VARIANT_FLAG, BACKEND, fast_unordered=True)
    group.add_argument(
        CHUNK_SIZE_FLAG,
        type=int,
        default=None,
        help=(
            "Token rows per chunk of the Cake LM-head kernels: the largest vocabulary-sized buffer holds this many "
            f"rows (BF16 logits of [chunk, V]). Default: {DEFAULT_CHUNK_SIZE}. Only with {BACKEND}."
        ),
    )


def validate_args(args: Namespace) -> None:
    """Resolve the variant and the chunk size (written back to ``args``) and reject what the kernels do not serve.

    The checks read the miles / Megatron arguments where ``args`` has them and skip the ones it lacks.
    ``--log-probs-backend cake``'s miles-side rules (no true-on-policy mode, no sampling-support replay, a
    ``--vocab-size``) are miles' ``validate_log_probs_backend_args``.
    """
    selected = getattr(args, "log_probs_backend", None) == "cake"
    chunk_size = getattr(args, "cake_lm_head_chunk_size", None)
    if not selected and chunk_size is not None:
        raise ValueError(f"{CHUNK_SIZE_FLAG} requires {BACKEND}")
    # fast sums the K slices of dX in a run-dependent order
    _variant.resolve(args, VARIANT_FLAG, selected, fast_unordered=True, backend=BACKEND)
    if not selected:
        return
    if chunk_size is None:
        chunk_size = DEFAULT_CHUNK_SIZE
    elif isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or not 1 <= chunk_size <= MAX_CHUNK_SIZE:
        raise ValueError(f"{CHUNK_SIZE_FLAG} must be an integer in [1, {MAX_CHUNK_SIZE}], got {chunk_size!r}")
    args.cake_lm_head_chunk_size = chunk_size
    problem = next(_unsupported(args), None)
    if problem is not None:
        raise ValueError(f"{BACKEND} {problem}")


def _unsupported(args: Namespace) -> Iterator[str]:
    """What ``args`` sets that the kernels do not serve, one message each (skipping the attributes ``args`` lacks)."""
    train_backend = getattr(args, "train_backend", None)
    if train_backend is not None and train_backend != "megatron":
        yield f"needs --train-backend megatron (the output layer hook of Megatron's GPTModel), got {train_backend}"
    tensor_parallel = getattr(args, "tensor_model_parallel_size", None)
    if tensor_parallel is not None and tensor_parallel != 1:
        yield (
            "needs --tensor-model-parallel-size 1: the kernels take the whole output weight on one GPU, "
            f"got {tensor_parallel}"
        )
    loss_type = getattr(args, "loss_type", None)
    if loss_type is not None and loss_type not in LOSS_TYPES:
        yield f"supports --loss-type {' and '.join(LOSS_TYPES)}, got {loss_type}"
    entropy_coef = getattr(args, "entropy_coef", None)
    if entropy_coef is not None and entropy_coef != 0:
        yield f"computes no entropy: it needs --entropy-coef 0, got {entropy_coef}"
    for flag in ("observe_training_entropy", "use_rollout_entropy"):
        if getattr(args, flag, False):
            yield f"computes no entropy: drop --{flag.replace('_', '-')}"
    if hasattr(args, "vocab_size"):
        vocab_size = args.vocab_size
        if vocab_size is None:
            yield "needs --vocab-size (the HF config's vocab_size): the softmax runs over the true vocabulary"
        elif vocab_size % VOCAB_MULTIPLE:
            yield f"needs a --vocab-size that is a multiple of {VOCAB_MULTIPLE}, got {vocab_size}"
    hidden_size = getattr(args, "hidden_size", None)
    if hidden_size is not None and hidden_size % HIDDEN_MULTIPLE:
        yield f"needs a --hidden-size that is a multiple of {HIDDEN_MULTIPLE}, got {hidden_size}"
    if getattr(args, "bf16", None) is False:
        yield "needs --bf16: the kernels take BF16 hidden states and output weights"
    if getattr(args, "multi_lora_n_adapters", 0):
        yield "does not support multi-LoRA (--multi-lora-n-adapters)"
    if getattr(args, "lora_rank", 0) and _targets_unembed(getattr(args, "target_modules", None)):
        yield "does not support a LoRA adapter on the output layer (the unembed / lm_head --target-modules)"
    if getattr(args, "use_mup", False):
        yield "does not apply MuP's output logit scaling (--use-mup)"


def _targets_unembed(target_modules) -> bool:
    """Does ``--target-modules`` (a comma-separated string or a list) name the output layer?"""
    if not target_modules:
        return False
    names = target_modules.split(",") if isinstance(target_modules, str) else target_modules
    return any(str(name).strip().rpartition(".")[2] in _UNEMBED_TARGETS for name in names)


def configure(args) -> None:
    """Apply the parsed arguments to the kernels: idempotent, safe to call more than once.

    ``args`` holds the values :func:`validate_args` resolved; an unresolved variant runs ``det`` under deterministic
    mode (when ``args`` carries it) and ``fast`` otherwise.
    """
    variant = getattr(args, "cake_lm_head_variant", None)
    if variant is None:
        variant = "det" if _variant.deterministic_mode(args) else "fast"
    elif variant not in _variant.VARIANTS:
        raise ValueError(f"{VARIANT_FLAG} must be one of {_variant.VARIANTS}, not {variant!r}")
    chunk_size = getattr(args, "cake_lm_head_chunk_size", None)
    ops = import_module("miles_ext_kernels.lm_head.ops")
    ops.VARIANT = variant
    ops.CHUNK_SIZE = DEFAULT_CHUNK_SIZE if chunk_size is None else int(chunk_size)
