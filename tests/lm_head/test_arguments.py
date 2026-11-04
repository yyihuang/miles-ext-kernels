"""The LM-head kernel arguments: registration, variant and chunk-size resolution, the startup rejections and
``configure`` (torch-free)."""

import argparse
import sys
import types

import pytest

import miles_ext_kernels
from miles_ext_kernels.lm_head import arguments

# miles' flat namespace of a run the kernels serve (DeepSeek-V4-Flash-like geometry, TP 1)
SERVED = dict(
    log_probs_backend="cake",
    train_backend="megatron",
    tensor_model_parallel_size=1,
    loss_type="policy_loss",
    entropy_coef=0.0,
    observe_training_entropy=False,
    use_rollout_entropy=False,
    vocab_size=129280,
    hidden_size=4096,
    bf16=True,
    multi_lora_n_adapters=0,
    lora_rank=0,
    target_modules=None,
    deterministic_mode=False,
)


def _parse(argv: list[str], **attributes) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    arguments.add_arguments(parser.add_argument_group("miles-ext-kernels"))
    args = parser.parse_args(argv)
    for name, value in attributes.items():
        setattr(args, name, value)
    return args


def _served(argv: list[str] = (), **overrides) -> argparse.Namespace:
    return _parse(list(argv), **{**SERVED, **overrides})


def test_the_flags_are_registered_through_the_package_hook_and_the_defaults_validate():
    assert arguments in miles_ext_kernels._KERNEL_PACKAGES
    parser = argparse.ArgumentParser()
    miles_ext_kernels.add_arguments(parser)
    args = parser.parse_args([])

    assert args.cake_lm_head_variant is None and args.cake_lm_head_chunk_size is None
    miles_ext_kernels.validate_args(args)
    assert args.cake_lm_head_variant is None and args.cake_lm_head_chunk_size is None


def test_the_flags_parse_their_values():
    args = _parse(["--cake-lm-head-variant", "det", "--cake-lm-head-chunk-size", "2048"])

    assert args.cake_lm_head_variant == "det" and args.cake_lm_head_chunk_size == 2048
    with pytest.raises(SystemExit):
        _parse(["--cake-lm-head-variant", "fastest"])


@pytest.mark.parametrize("backend", [None, "torch", "fused"])
@pytest.mark.parametrize(
    ("argv", "flag"),
    [
        (["--cake-lm-head-variant", "det"], "--cake-lm-head-variant"),
        (["--cake-lm-head-variant", "fast"], "--cake-lm-head-variant"),
        (["--cake-lm-head-chunk-size", "4096"], "--cake-lm-head-chunk-size"),
    ],
)
def test_an_lm_head_argument_without_the_cake_backend_is_rejected(backend, argv, flag):
    args = _parse(argv)
    if backend is not None:
        args.log_probs_backend = backend

    with pytest.raises(ValueError, match=f"{flag} requires --log-probs-backend cake"):
        arguments.validate_args(args)


@pytest.mark.parametrize("backend", [None, "torch", "fused"])
def test_without_the_cake_backend_the_arguments_stay_unset(backend):
    # a run on another backend keeps every Megatron / miles setting the kernels would reject
    args = _served(log_probs_backend=backend, tensor_model_parallel_size=8, bf16=False, entropy_coef=0.01)

    arguments.validate_args(args)
    assert args.cake_lm_head_variant is None and args.cake_lm_head_chunk_size is None


@pytest.mark.parametrize(("deterministic", "variant"), [(False, "fast"), (True, "det")])
def test_unset_variant_resolves_to_fast_or_to_det_under_deterministic_mode(deterministic, variant):
    args = _served(deterministic_mode=deterministic)

    arguments.validate_args(args)
    assert args.cake_lm_head_variant == variant
    assert args.cake_lm_head_chunk_size == arguments.DEFAULT_CHUNK_SIZE


@pytest.mark.parametrize("deterministic", [False, True])
def test_an_explicit_det_variant_is_kept(deterministic):
    args = _served(["--cake-lm-head-variant", "det"], deterministic_mode=deterministic)

    arguments.validate_args(args)
    assert args.cake_lm_head_variant == "det"


def test_an_explicit_fast_variant_is_kept_outside_deterministic_mode_and_rejected_in_it():
    args = _served(["--cake-lm-head-variant", "fast"])
    arguments.validate_args(args)
    assert args.cake_lm_head_variant == "fast"

    # fast sums the dX K slices in a run-dependent order
    args = _served(["--cake-lm-head-variant", "fast"], deterministic_mode=True)
    with pytest.raises(ValueError, match="--deterministic-mode needs --cake-lm-head-variant det"):
        arguments.validate_args(args)


@pytest.mark.parametrize("chunk_size", [1, 1024, 4097, 65535])
def test_an_explicit_chunk_size_is_kept(chunk_size):
    args = _served(["--cake-lm-head-chunk-size", str(chunk_size)])

    arguments.validate_args(args)
    assert args.cake_lm_head_chunk_size == chunk_size


@pytest.mark.parametrize("chunk_size", [0, -4096, 65536])
def test_a_chunk_size_out_of_range_is_rejected(chunk_size):
    args = _served(["--cake-lm-head-chunk-size", str(chunk_size)])

    with pytest.raises(ValueError, match=r"--cake-lm-head-chunk-size must be an integer in \[1, 65535\]"):
        arguments.validate_args(args)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (dict(train_backend="fsdp"), "needs --train-backend megatron"),
        (dict(train_backend="torchtitan"), "needs --train-backend megatron"),
        (dict(tensor_model_parallel_size=2), "needs --tensor-model-parallel-size 1"),
        (dict(loss_type="custom_loss"), "supports --loss-type policy_loss and sft_loss, got custom_loss"),
        (dict(loss_type="score_centering"), "supports --loss-type policy_loss and sft_loss"),
        (dict(entropy_coef=0.01), "computes no entropy: it needs --entropy-coef 0"),
        (dict(observe_training_entropy=True), "computes no entropy: drop --observe-training-entropy"),
        (dict(use_rollout_entropy=True), "computes no entropy: drop --use-rollout-entropy"),
        (dict(vocab_size=None), "needs --vocab-size"),
        (dict(vocab_size=151936), "needs a --vocab-size that is a multiple of 256, got 151936"),
        (dict(hidden_size=2304), "needs a --hidden-size that is a multiple of 512, got 2304"),
        (dict(bf16=False), "needs --bf16"),
        (dict(multi_lora_n_adapters=4), "does not support multi-LoRA"),
        (dict(lora_rank=8, target_modules="attn,unembed"), "does not support a LoRA adapter on the output layer"),
        (dict(lora_rank=8, target_modules="q_proj, lm_head"), "does not support a LoRA adapter on the output layer"),
        (
            dict(lora_rank=8, target_modules=["language_model.lm_head"]),
            "does not support a LoRA adapter on the output layer",
        ),
        (dict(lora_rank=8, target_modules=["output_layer"]), "does not support a LoRA adapter on the output layer"),
        (dict(use_mup=True), "does not apply MuP's output logit scaling"),
    ],
)
def test_what_the_kernels_do_not_serve_is_rejected_at_startup(overrides, message):
    args = _served(**overrides)

    with pytest.raises(ValueError, match=f"^--log-probs-backend cake {message}"):
        arguments.validate_args(args)


@pytest.mark.parametrize(
    "overrides",
    [
        dict(loss_type="sft_loss"),
        dict(vocab_size=154880, hidden_size=6144),
        dict(vocab_size=248320, hidden_size=5120),
        dict(lora_rank=8, target_modules=None),
        dict(lora_rank=8, target_modules="attn,mlp"),
        dict(lora_rank=8, target_modules=["q_proj", "v_proj", "head_dim_proj"]),
        dict(lora_rank=0, target_modules="unembed"),
        dict(use_mup=False),
    ],
)
def test_what_the_kernels_serve_is_accepted(overrides):
    arguments.validate_args(_served(**overrides))


def test_the_rules_skip_the_settings_a_namespace_lacks():
    # e.g. a caller validating only the kernel arguments: nothing to reject but what it names
    args = _parse([], log_probs_backend="cake")

    arguments.validate_args(args)
    assert args.cake_lm_head_variant == "fast"
    assert args.cake_lm_head_chunk_size == arguments.DEFAULT_CHUNK_SIZE


@pytest.fixture
def ops(monkeypatch):
    """A stand-in for the torch-backed ops module: configure must reach it without importing torch here."""
    module = types.ModuleType("miles_ext_kernels.lm_head.ops")
    module.VARIANT = None
    module.CHUNK_SIZE = None
    monkeypatch.setitem(sys.modules, "miles_ext_kernels.lm_head.ops", module)
    return module


@pytest.mark.parametrize(
    ("argv", "deterministic", "variant", "chunk_size"),
    [
        ([], False, "fast", 4096),
        ([], True, "det", 4096),
        (["--cake-lm-head-variant", "det", "--cake-lm-head-chunk-size", "2048"], False, "det", 2048),
    ],
)
def test_configure_applies_the_resolved_arguments(ops, argv, deterministic, variant, chunk_size):
    args = _served(argv, deterministic_mode=deterministic)
    arguments.validate_args(args)
    # miles hands configure a namespace of just the kernel arguments' values
    values = argparse.Namespace(
        cake_lm_head_variant=args.cake_lm_head_variant, cake_lm_head_chunk_size=args.cake_lm_head_chunk_size
    )

    for _ in range(2):
        arguments.configure(values)
        assert (ops.VARIANT, ops.CHUNK_SIZE) == (variant, chunk_size)


@pytest.mark.parametrize(
    ("args", "variant"),
    [
        (argparse.Namespace(), "fast"),
        (argparse.Namespace(cake_lm_head_variant=None, cake_lm_head_chunk_size=None), "fast"),
        (argparse.Namespace(deterministic_mode=True), "det"),
    ],
)
def test_configure_resolves_unvalidated_arguments(ops, args, variant):
    arguments.configure(args)

    assert (ops.VARIANT, ops.CHUNK_SIZE) == (variant, arguments.DEFAULT_CHUNK_SIZE)


def test_configure_rejects_an_unknown_variant(ops):
    with pytest.raises(ValueError, match="--cake-lm-head-variant must be one of"):
        arguments.configure(argparse.Namespace(cake_lm_head_variant="fastest"))


def test_the_package_exports_configure_lazily():
    from miles_ext_kernels.lm_head import configure

    assert configure is arguments.configure
