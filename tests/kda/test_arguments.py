"""The KDA package's argument hooks: no argument of its own, the settings ``--kda-backend cake`` rejects at startup
and a torch-free import (torch-free)."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pytest

from miles_ext_kernels.kda import arguments

REPO_ROOT = Path(__file__).resolve().parents[2]


def _miles_args(**values) -> argparse.Namespace:
    """The miles values validate_args reads, at their defaults unless given."""
    return argparse.Namespace(**{"kda_backend": "fla", "context_parallel_size": 1, "qkv_format": "thd", **values})


def test_the_package_registers_no_argument():
    parser = argparse.ArgumentParser()
    before = len(parser._actions)

    arguments.add_arguments(parser.add_argument_group("miles-ext-kernels"))

    assert len(parser._actions) == before
    arguments.validate_args(parser.parse_args([]))  # a namespace holding none of miles' values


@pytest.mark.parametrize(
    "values",
    [
        {},
        {"kda_backend": "cake"},
        {"kda_backend": "cake", "context_parallel_size": None},
        {"kda_backend": "fla", "context_parallel_size": 2},
        {"kda_backend": "fla", "qkv_format": "bshd"},
    ],
)
def test_supported_settings_validate(values):
    arguments.validate_args(_miles_args(**values))


@pytest.mark.parametrize("context_parallel_size", [2, 8])
def test_cake_rejects_context_parallelism(context_parallel_size):
    args = _miles_args(kda_backend="cake", context_parallel_size=context_parallel_size)

    with pytest.raises(
        ValueError, match=f"context parallelism yet \\(--context-parallel-size {context_parallel_size}"
    ):
        arguments.validate_args(args)


def test_cake_rejects_the_bshd_layout():
    with pytest.raises(ValueError, match="--kda-backend cake needs --qkv-format thd"):
        arguments.validate_args(_miles_args(kda_backend="cake", qkv_format="bshd"))


def test_configure_is_the_lazy_package_attribute_and_accepts_the_namespace():
    from miles_ext_kernels.kda import configure

    assert configure is arguments.configure
    configure(argparse.Namespace())


_NO_TORCH_ON_IMPORT = """
import argparse
import sys
from pathlib import Path

sys.modules["torch"] = None  # any import of torch or tvm_ffi raises ImportError
sys.modules["tvm_ffi"] = None

import miles_ext_kernels.kda
from miles_ext_kernels.kda import arguments, configure

assert Path(miles_ext_kernels.kda.__file__).resolve().parents[2] == Path(sys.argv[1]).resolve()
arguments.add_arguments(argparse.ArgumentParser())
arguments.validate_args(argparse.Namespace(kda_backend="cake", context_parallel_size=1, qkv_format="thd"))
configure(argparse.Namespace())
loaded = sorted(n for n in sys.modules if n.startswith("miles_ext_kernels.kda."))
assert loaded == ["miles_ext_kernels.kda.arguments"], loaded
"""


def test_importing_the_package_and_validating_does_not_import_torch_or_the_kernels():
    pythonpath = os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", _NO_TORCH_ON_IMPORT, str(REPO_ROOT)],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": pythonpath},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
