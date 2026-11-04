"""The GDN kernel arguments, registered and validated through the package hooks miles calls (torch-free)."""

import argparse
import sys
import types

import pytest

import miles_ext_kernels
from miles_ext_kernels.gdn import arguments

REJECTION = "--cake-gdn-recompute-state requires --linear-attention-backend cake"
VARIANT_REJECTION = "--cake-gdn-variant requires --linear-attention-backend cake"


def _parse(argv: list[str], **attributes) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    miles_ext_kernels.add_arguments(parser)
    args = parser.parse_args(argv)
    for name, value in attributes.items():  # miles' own arguments of the flat namespace it validates
        setattr(args, name, value)
    return args


@pytest.fixture
def ops(monkeypatch):
    """A stand-in for the torch-backed ops module: configure must reach it without importing torch here."""
    module = types.ModuleType("miles_ext_kernels.gdn.ops")
    module.SAVE_STATE = None
    module.VARIANT = None
    monkeypatch.setitem(sys.modules, "miles_ext_kernels.gdn.ops", module)
    return module


def test_the_flags_are_registered_through_the_package_hook_and_the_defaults_validate():
    assert arguments in miles_ext_kernels._KERNEL_PACKAGES
    args = _parse([])

    assert args.cake_gdn_recompute_state is False
    assert args.cake_gdn_variant is None
    miles_ext_kernels.validate_args(args)
    assert args.cake_gdn_variant is None  # no cake backend: nothing to resolve


@pytest.mark.parametrize("backend", [None, "fla", "flashqla"])
def test_recompute_state_without_the_cake_backend_is_rejected(backend):
    args = _parse(["--cake-gdn-recompute-state"])
    if backend is not None:
        args.linear_attention_backend = backend

    with pytest.raises(ValueError, match=REJECTION):
        miles_ext_kernels.validate_args(args)


def test_recompute_state_with_the_cake_backend_is_accepted():
    args = _parse(["--cake-gdn-recompute-state"], linear_attention_backend="cake")

    miles_ext_kernels.validate_args(args)
    assert args.cake_gdn_recompute_state is True


@pytest.mark.parametrize("variant", ["det", "fast"])
@pytest.mark.parametrize("backend", [None, "fla", "flashqla"])
def test_a_variant_without_the_cake_backend_is_rejected(backend, variant):
    args = _parse(["--cake-gdn-variant", variant])
    if backend is not None:
        args.linear_attention_backend = backend

    with pytest.raises(ValueError, match=VARIANT_REJECTION):
        miles_ext_kernels.validate_args(args)


@pytest.mark.parametrize(("deterministic", "variant"), [(False, "fast"), (True, "det")])
def test_unset_resolves_to_fast_or_to_det_under_deterministic_mode(deterministic, variant):
    args = _parse([], linear_attention_backend="cake", deterministic_mode=deterministic)

    miles_ext_kernels.validate_args(args)
    assert args.cake_gdn_variant == variant


def test_deterministic_mode_of_the_typed_configuration_resolves_to_det():
    args = _parse([], linear_attention_backend="cake", backend=types.SimpleNamespace(deterministic_mode=True))

    arguments.validate_args(args)
    assert args.cake_gdn_variant == "det"


@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("variant", ["det", "fast"])
def test_an_explicit_variant_is_kept_and_fast_is_allowed_under_deterministic_mode(deterministic, variant):
    # every launch of the fast package is fixed-order: deterministic mode may run it
    args = _parse(["--cake-gdn-variant", variant], linear_attention_backend="cake", deterministic_mode=deterministic)

    miles_ext_kernels.validate_args(args)
    assert args.cake_gdn_variant == variant


def test_an_unknown_variant_is_rejected_by_the_parser():
    with pytest.raises(SystemExit):
        _parse(["--cake-gdn-variant", "fastest"])


@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("variant", ["det", "fast"])
def test_configure_sets_the_kernels_variant_and_save_state_switch(ops, recompute, variant):
    argv = ["--cake-gdn-variant", variant] + (["--cake-gdn-recompute-state"] if recompute else [])
    args = _parse(argv, linear_attention_backend="cake")
    miles_ext_kernels.validate_args(args)

    for _ in range(2):  # miles calls it once per layer
        arguments.configure(args)
        assert ops.SAVE_STATE is (not recompute)
        assert ops.VARIANT == variant


def test_configure_without_the_arguments_keeps_the_tape_and_runs_det(ops):
    # a namespace without the arguments (no --use-miles-ext-kernels registration, or never validated)
    arguments.configure(argparse.Namespace())
    assert ops.SAVE_STATE is True
    assert ops.VARIANT == "det"

    arguments.configure(argparse.Namespace(cake_gdn_variant=None))
    assert ops.VARIANT == "det"


def test_configure_rejects_an_unknown_variant(ops):
    with pytest.raises(ValueError, match="--cake-gdn-variant must be one of"):
        arguments.configure(argparse.Namespace(cake_gdn_variant="fastest"))


@pytest.mark.parametrize(("deterministic", "variant"), [(False, "fast"), (True, "det")])
def test_the_resolved_variant_reaches_configure_as_miles_hands_it_over(ops, deterministic, variant):
    # miles validates the flat namespace, then moves the registered values into miles_ext_kernels_args and builds
    # each cake GDN layer with configure(argparse.Namespace(**miles_ext_kernels_args))
    args = _parse([], linear_attention_backend="cake", deterministic_mode=deterministic)
    miles_ext_kernels.validate_args(args)
    registered = argparse.ArgumentParser(add_help=False)
    miles_ext_kernels.add_arguments(registered)
    ext_args = {action.dest: vars(args).pop(action.dest) for action in registered._actions}

    arguments.configure(argparse.Namespace(**ext_args))
    assert ops.VARIANT == variant
    assert ops.SAVE_STATE is True


def test_the_package_exports_configure_lazily():
    from miles_ext_kernels.gdn import configure

    assert configure is arguments.configure
