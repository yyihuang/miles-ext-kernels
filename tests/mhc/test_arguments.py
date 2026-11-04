"""The mHC kernel arguments, registered and validated through the package hooks miles calls (torch-free)."""

import argparse
import sys
import types

import pytest

import miles_ext_kernels
from miles_ext_kernels.mhc import arguments

FLAG = "--cake-mhc-variant"


def _parse(argv: list[str], **attributes) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    miles_ext_kernels.add_arguments(parser)
    args = parser.parse_args(argv)
    for name, value in attributes.items():
        setattr(args, name, value)
    return args


def test_the_flag_is_registered_through_the_package_hook_and_the_defaults_validate():
    assert arguments in miles_ext_kernels._KERNEL_PACKAGES
    args = _parse([])

    assert args.cake_mhc_variant is None
    miles_ext_kernels.validate_args(args)
    # without --mhc-backend cake the variant stays unset
    assert args.cake_mhc_variant is None


@pytest.mark.parametrize(("deterministic", "variant"), [(False, "fast"), (True, "det")])
def test_unset_resolves_to_fast_or_to_det_under_deterministic_mode(deterministic, variant):
    args = _parse([], mhc_backend="cake", deterministic_mode=deterministic)

    miles_ext_kernels.validate_args(args)
    assert args.cake_mhc_variant == variant


def test_deterministic_mode_of_the_typed_configuration_resolves_to_det():
    args = _parse([], mhc_backend="cake", backend=types.SimpleNamespace(deterministic_mode=True))

    miles_ext_kernels.validate_args(args)
    assert args.cake_mhc_variant == "det"


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_an_explicit_variant_is_kept_outside_deterministic_mode(variant):
    args = _parse([FLAG, variant], mhc_backend="cake", deterministic_mode=False)

    miles_ext_kernels.validate_args(args)
    assert args.cake_mhc_variant == variant


def test_explicit_fast_is_rejected_under_deterministic_mode_and_det_is_kept():
    # fast adds the dfn partials with FP32 atomics: deterministic mode cannot run it
    args = _parse([FLAG, "fast"], mhc_backend="cake", deterministic_mode=True)
    with pytest.raises(ValueError, match="--deterministic-mode needs --cake-mhc-variant det"):
        miles_ext_kernels.validate_args(args)

    args = _parse([FLAG, "det"], mhc_backend="cake", deterministic_mode=True)
    miles_ext_kernels.validate_args(args)
    assert args.cake_mhc_variant == "det"


@pytest.mark.parametrize("backend", [None, "megatron"])
@pytest.mark.parametrize("variant", ["det", "fast"])
def test_the_variant_without_the_cake_backend_is_rejected(backend, variant):
    args = _parse([FLAG, variant])
    if backend is not None:
        args.mhc_backend = backend

    with pytest.raises(ValueError, match="--cake-mhc-variant requires --mhc-backend cake"):
        miles_ext_kernels.validate_args(args)


@pytest.mark.parametrize("hidden", [None, 4096, 5120, 7168])
def test_the_kernels_hidden_sizes_validate(hidden):
    args = _parse([], mhc_backend="cake", hidden_size=hidden)

    miles_ext_kernels.validate_args(args)


def test_another_hidden_size_is_rejected_only_under_the_cake_backend():
    with pytest.raises(ValueError, match=r"--mhc-backend cake runs the hidden sizes \(4096, 5120, 7168\)"):
        miles_ext_kernels.validate_args(_parse([], mhc_backend="cake", hidden_size=2048))

    miles_ext_kernels.validate_args(_parse([], mhc_backend="megatron", hidden_size=2048))


def test_configure_sets_the_variant_of_the_kernels(monkeypatch):
    # a stand-in for the torch-backed ops module: configure must reach it without importing torch here
    ops = types.ModuleType("miles_ext_kernels.mhc.ops")
    ops.VARIANT = None
    monkeypatch.setitem(sys.modules, "miles_ext_kernels.mhc.ops", ops)

    for variant in ("det", "fast", "det"):  # miles calls it once per layer
        arguments.configure(_parse([FLAG, variant]))
        assert ops.VARIANT == variant

    # a namespace without the argument (no --use-miles-ext-kernels registration) runs fast
    arguments.configure(argparse.Namespace())
    assert ops.VARIANT == "fast"
    with pytest.raises(ValueError, match="--cake-mhc-variant must be one of"):
        arguments.configure(argparse.Namespace(cake_mhc_variant="fastest"))


def test_the_package_exports_its_names_lazily():
    import miles_ext_kernels.mhc as mhc

    assert mhc.configure is arguments.configure
    assert mhc.HIDDEN_SIZES == (4096, 5120, 7168)
    assert mhc.STREAMS == 4
    assert {"mhc_prepare", "mhc_finish", "mhc_head", "is_supported", "prebuild"} <= set(dir(mhc))
    with pytest.raises(AttributeError):
        mhc.no_such_name
