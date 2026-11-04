"""Training kernels for miles, enabled by miles' ``--use-miles-ext-kernels``.

miles owns backend selection (``--linear-attention-backend cake``, ...) and imports the kernel package at its
dispatch seam. Each kernel package here owns its own arguments and validates them.
"""

from argparse import ArgumentParser, Namespace

from miles_ext_kernels.conv1d import arguments as conv1d_arguments
from miles_ext_kernels.dense_gemm import arguments as dense_gemm_arguments
from miles_ext_kernels.dsa import arguments as dsa_arguments
from miles_ext_kernels.gated_rmsnorm import arguments as gated_rmsnorm_arguments
from miles_ext_kernels.gdn import arguments as gdn_arguments
from miles_ext_kernels.gqa import arguments as gqa_arguments
from miles_ext_kernels.kda import arguments as kda_arguments
from miles_ext_kernels.mhc import arguments as mhc_arguments
from miles_ext_kernels.rmsnorm import arguments as rmsnorm_arguments

# Each entry is a module exposing add_arguments(group) and validate_args(args); importing them stays torch-free.
_KERNEL_PACKAGES = (
    gdn_arguments,
    kda_arguments,
    dsa_arguments,
    mhc_arguments,
    gqa_arguments,
    conv1d_arguments,
    gated_rmsnorm_arguments,
    rmsnorm_arguments,
    dense_gemm_arguments,
)


def add_arguments(parser: ArgumentParser) -> None:
    group = parser.add_argument_group("miles-ext-kernels")
    for package in _KERNEL_PACKAGES:
        package.add_arguments(group)


def validate_args(args: Namespace) -> None:
    for package in _KERNEL_PACKAGES:
        package.validate_args(args)
