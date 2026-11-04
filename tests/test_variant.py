"""The shared ``--cake-<kernel>-variant`` resolution (torch-free)."""

import argparse
import logging
from types import SimpleNamespace

import pytest

from miles_ext_kernels import _variant

FLAG = "--cake-synth-variant"


def _parse(argv: list[str], **attributes) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    _variant.add_variant_argument(parser, FLAG, "--synth-backend cake")
    args = parser.parse_args(argv)
    for name, value in attributes.items():
        setattr(args, name, value)
    return args


def test_the_flag_takes_det_or_fast_and_defaults_to_unset():
    assert _parse([]).cake_synth_variant is None
    assert _parse([FLAG, "det"]).cake_synth_variant == "det"
    with pytest.raises(SystemExit):
        _parse([FLAG, "fastest"])


@pytest.mark.parametrize(
    ("args", "deterministic"),
    [
        (SimpleNamespace(), False),
        (SimpleNamespace(deterministic_mode=False), False),
        (SimpleNamespace(deterministic_mode=True), True),
        # miles' typed configuration
        (SimpleNamespace(backend=SimpleNamespace(deterministic_mode=True)), True),
        (SimpleNamespace(backend=SimpleNamespace()), False),
    ],
)
def test_deterministic_mode_is_read_from_the_namespace_miles_validates(args, deterministic):
    assert _variant.deterministic_mode(args) is deterministic


@pytest.mark.parametrize("fast_unordered", [False, True])
@pytest.mark.parametrize(("deterministic", "variant"), [(False, "fast"), (True, "det")])
def test_unset_resolves_to_fast_or_to_det_under_deterministic_mode(fast_unordered, deterministic, variant):
    args = _parse([], deterministic_mode=deterministic)

    assert _variant.resolve(args, FLAG, True, fast_unordered) == variant
    assert args.cake_synth_variant == variant


@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("variant", ["det", "fast"])
def test_an_explicit_variant_is_kept_when_deterministic_mode_allows_it(deterministic, variant):
    args = _parse([FLAG, variant], deterministic_mode=deterministic)

    # fast_unordered=False: a fast variant with a fixed reduction order is reproducible
    assert _variant.resolve(args, FLAG, True, False) == variant
    assert args.cake_synth_variant == variant


def test_an_unordered_fast_variant_is_rejected_under_deterministic_mode():
    args = _parse([FLAG, "fast"], deterministic_mode=True)

    with pytest.raises(ValueError, match="--deterministic-mode needs --cake-synth-variant det"):
        _variant.resolve(args, FLAG, True, True)
    assert _variant.resolve(_parse([FLAG, "fast"], deterministic_mode=False), FLAG, True, True) == "fast"


def test_a_variant_without_the_cake_backend_is_rejected_and_unset_stays_unset():
    with pytest.raises(ValueError, match="--cake-synth-variant requires --synth-backend cake"):
        _variant.resolve(_parse([FLAG, "det"]), FLAG, False, True, backend="--synth-backend cake")

    args = _parse([])
    assert _variant.resolve(args, FLAG, False, True) is None
    assert args.cake_synth_variant is None


def test_a_det_fallback_is_logged_once_per_family_and_reason(caplog, monkeypatch):
    monkeypatch.setattr(_variant, "_warned", set())

    with caplog.at_level(logging.WARNING, logger="miles_ext_kernels"):
        for _ in range(3):
            _variant.warn_det_fallback("synth", "context parallelism")
        _variant.warn_det_fallback("synth", "packed sequences")
        _variant.warn_det_fallback("other", "context parallelism")

    assert [record.getMessage() for record in caplog.records] == [
        "miles_ext_kernels.synth: the fast variant does not serve context parallelism; running det",
        "miles_ext_kernels.synth: the fast variant does not serve packed sequences; running det",
        "miles_ext_kernels.other: the fast variant does not serve context parallelism; running det",
    ]
