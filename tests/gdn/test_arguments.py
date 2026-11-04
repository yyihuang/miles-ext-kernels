"""The GDN kernel arguments, registered and validated through the package hooks miles calls (torch-free)."""

import argparse
import sys
import types

import pytest

import miles_ext_kernels
from miles_ext_kernels.gdn import arguments

REJECTION = "--cake-gdn-recompute-state requires --linear-attention-backend cake"


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    miles_ext_kernels.add_arguments(parser)
    return parser.parse_args(argv)


def test_the_flag_is_registered_through_the_package_hook_and_the_defaults_validate():
    assert arguments in miles_ext_kernels._KERNEL_PACKAGES
    args = _parse([])

    assert args.cake_gdn_recompute_state is False
    miles_ext_kernels.validate_args(args)


@pytest.mark.parametrize("backend", [None, "fla", "flashqla"])
def test_recompute_state_without_the_cake_backend_is_rejected(backend):
    args = _parse(["--cake-gdn-recompute-state"])
    if backend is not None:
        args.linear_attention_backend = backend

    with pytest.raises(ValueError, match=REJECTION):
        miles_ext_kernels.validate_args(args)


def test_recompute_state_with_the_cake_backend_is_accepted():
    args = _parse(["--cake-gdn-recompute-state"])
    args.linear_attention_backend = "cake"

    miles_ext_kernels.validate_args(args)
    assert args.cake_gdn_recompute_state is True


@pytest.mark.parametrize("recompute", [False, True])
def test_configure_sets_the_kernels_save_state_switch(monkeypatch, recompute):
    # a stand-in for the torch-backed ops module: configure must reach it without importing torch here
    ops = types.ModuleType("miles_ext_kernels.gdn.ops")
    ops.SAVE_STATE = None
    monkeypatch.setitem(sys.modules, "miles_ext_kernels.gdn.ops", ops)
    args = _parse(["--cake-gdn-recompute-state"] if recompute else [])

    for _ in range(2):  # miles calls it once per layer
        arguments.configure(args)
        assert ops.SAVE_STATE is (not recompute)

    # a namespace without the argument (no --use-miles-ext-kernels registration) keeps the tape
    arguments.configure(argparse.Namespace())
    assert ops.SAVE_STATE is True


def test_the_package_exports_configure_lazily():
    from miles_ext_kernels.gdn import configure

    assert configure is arguments.configure
