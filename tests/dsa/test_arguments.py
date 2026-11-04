"""The DSA kernel arguments, registered, resolved and validated through the package hooks miles calls (torch-free)."""

import argparse
import sys
import types

import pytest

import miles_ext_kernels
from miles_ext_kernels.dsa import arguments

GIB = 1 << 30


def _parse(argv: list[str], **miles_args) -> argparse.Namespace:
    """The DSA arguments parsed as miles parses them, plus the miles fields ``miles_args`` (backend, mode)."""
    parser = argparse.ArgumentParser()
    arguments.add_arguments(parser.add_argument_group("miles-ext-kernels"))
    args = parser.parse_args(argv)
    for name, value in miles_args.items():
        setattr(args, name, value)
    return args


def test_the_package_hook_registers_the_dsa_arguments():
    assert arguments in miles_ext_kernels._KERNEL_PACKAGES
    parser = argparse.ArgumentParser()
    miles_ext_kernels.add_arguments(parser)
    args = parser.parse_args([])

    assert (args.cake_dsa_variant, args.cake_dsa_bwd_workspace_bytes, args.cake_dsa_lazy_build) == (None, None, False)
    miles_ext_kernels.validate_args(args)


@pytest.mark.parametrize("backend", [None, "auto"])
def test_the_defaults_validate_without_the_cake_backend_and_leave_the_variant_unset(backend):
    args = _parse([]) if backend is None else _parse([], dsa_attention_backend=backend)

    arguments.validate_args(args)
    assert args.cake_dsa_variant is None


@pytest.mark.parametrize(
    ("deterministic", "expected"), [(False, "fast"), (True, "det")], ids=["default-fast", "deterministic-det"]
)
def test_an_unset_variant_resolves_by_deterministic_mode_and_is_written_back(deterministic, expected):
    args = _parse([], dsa_attention_backend="cake", deterministic_mode=deterministic)

    arguments.validate_args(args)
    assert args.cake_dsa_variant == expected


def test_the_typed_configuration_carries_deterministic_mode_on_the_backend():
    args = _parse([], dsa_attention_backend="cake", backend=types.SimpleNamespace(deterministic_mode=True))

    arguments.validate_args(args)
    assert args.cake_dsa_variant == "det"


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_an_explicit_variant_is_kept_outside_deterministic_mode(variant):
    args = _parse(["--cake-dsa-variant", variant], dsa_attention_backend="cake", deterministic_mode=False)

    arguments.validate_args(args)
    assert args.cake_dsa_variant == variant


def test_explicit_det_is_kept_under_deterministic_mode():
    args = _parse(["--cake-dsa-variant", "det"], dsa_attention_backend="cake", deterministic_mode=True)

    arguments.validate_args(args)
    assert args.cake_dsa_variant == "det"


def test_explicit_fast_is_rejected_under_deterministic_mode():
    # the fast backward sums dK/dV with FP32 atomics in a run-dependent order
    args = _parse(["--cake-dsa-variant", "fast"], dsa_attention_backend="cake", deterministic_mode=True)

    with pytest.raises(ValueError, match="--cake-dsa-variant fast is not bit-reproducible"):
        arguments.validate_args(args)


def test_an_unknown_variant_is_rejected_by_the_parser():
    with pytest.raises(SystemExit):
        _parse(["--cake-dsa-variant", "nd"])


@pytest.mark.parametrize(
    "argv",
    [
        ["--cake-dsa-variant", "det"],
        ["--cake-dsa-variant", "fast"],
        ["--cake-dsa-bwd-workspace-bytes", str(GIB)],
        ["--cake-dsa-lazy-build"],
    ],
    ids=["variant-det", "variant-fast", "workspace", "lazy-build"],
)
@pytest.mark.parametrize("backend", [None, "auto"])
def test_every_dsa_argument_requires_the_cake_backend(argv, backend):
    args = _parse(argv) if backend is None else _parse(argv, dsa_attention_backend=backend)

    with pytest.raises(ValueError, match=f"{argv[0]} requires --dsa-attention-backend cake"):
        arguments.validate_args(args)


@pytest.mark.parametrize("workspace", ["0", "-1"])
def test_a_non_positive_workspace_is_rejected(workspace):
    args = _parse(["--cake-dsa-bwd-workspace-bytes", workspace], dsa_attention_backend="cake")

    with pytest.raises(ValueError, match="must be positive"):
        arguments.validate_args(args)


@pytest.fixture
def stub_kernels(monkeypatch):
    """Stand-ins for the torch-backed modules ``configure`` sets: it must reach them without importing torch here."""
    modules = {}
    for name, attrs in (
        ("ops", {"VARIANT": None}),
        ("_kernels", {"LAZY_BUILD": None}),
        ("reduction", {"BWD_WORKSPACE_BYTES": None, "DEFAULT_WORKSPACE_BYTES": 2 * GIB}),
    ):
        module = types.ModuleType(f"miles_ext_kernels.dsa.{name}")
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, module.__name__, module)
        modules[name] = module
    return modules


def _settings(modules):
    return modules["ops"].VARIANT, modules["reduction"].BWD_WORKSPACE_BYTES, modules["_kernels"].LAZY_BUILD


def test_configure_applies_the_resolved_arguments_and_is_idempotent(stub_kernels):
    args = _parse(
        ["--cake-dsa-bwd-workspace-bytes", str(GIB), "--cake-dsa-lazy-build"],
        dsa_attention_backend="cake",
        deterministic_mode=False,
    )
    arguments.validate_args(args)

    for _ in range(2):  # miles calls it once per layer
        arguments.configure(args)
        assert _settings(stub_kernels) == ("fast", GIB, True)


def test_configure_takes_the_ext_values_miles_carries_as_a_dict(stub_kernels):
    # miles moves the values into TrainConfig.miles_ext_kernels_args and rebuilds a namespace from them
    carried = {"cake_dsa_variant": "det", "cake_dsa_bwd_workspace_bytes": None, "cake_dsa_lazy_build": False}

    arguments.configure(argparse.Namespace(**carried))
    assert _settings(stub_kernels) == ("det", 2 * GIB, False)


def test_configure_without_the_arguments_runs_det_with_the_defaults(stub_kernels):
    arguments.configure(argparse.Namespace())

    assert _settings(stub_kernels) == ("det", 2 * GIB, False)


def test_configure_rejects_an_unknown_variant(stub_kernels):
    with pytest.raises(ValueError, match="--cake-dsa-variant must be one of"):
        arguments.configure(argparse.Namespace(cake_dsa_variant="nd"))


def test_the_package_exports_configure_lazily():
    from miles_ext_kernels.dsa import configure

    assert configure is arguments.configure
