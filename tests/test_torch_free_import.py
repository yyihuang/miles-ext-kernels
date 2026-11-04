"""miles registers and validates the arguments of every kernel package in a process without torch or tvm_ffi."""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_WITHOUT_TORCH = """
import argparse
import importlib
import sys
from pathlib import Path

sys.modules["torch"] = None  # any import of torch or tvm_ffi raises ImportError
sys.modules["tvm_ffi"] = None

import miles_ext_kernels
from miles_ext_kernels import _variant, cake_native

assert Path(miles_ext_kernels.__file__).resolve().parents[1] == Path(sys.argv[1]).resolve(), miles_ext_kernels.__file__
for package in miles_ext_kernels._KERNEL_PACKAGES:
    importlib.import_module(package.__name__.rpartition(".")[0])

parser = argparse.ArgumentParser()
miles_ext_kernels.add_arguments(parser)
miles_ext_kernels.validate_args(parser.parse_args([]))
# miles hands validate_args its whole parsed namespace, deterministic mode included
args = parser.parse_args([])
args.deterministic_mode = True
miles_ext_kernels.validate_args(args)
print(len(miles_ext_kernels._KERNEL_PACKAGES))
"""


def test_the_packages_import_and_validate_without_torch_or_tvm_ffi():
    pythonpath = os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", _WITHOUT_TORCH, str(REPO_ROOT)],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": pythonpath},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert int(result.stdout) >= 1
