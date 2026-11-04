import argparse
from pathlib import Path

import miles_ext_kernels

PACKAGE_DIR = Path(miles_ext_kernels.__file__).resolve().parent


def test_every_kernel_package_registers_and_validates_its_arguments():
    parser = argparse.ArgumentParser()
    miles_ext_kernels.add_arguments(parser)

    miles_ext_kernels.validate_args(parser.parse_args([]))


def test_every_kernel_package_with_arguments_is_registered():
    packages = {path.parent.name for path in PACKAGE_DIR.glob("*/arguments.py")}
    registered = {package.__name__.split(".")[1] for package in miles_ext_kernels._KERNEL_PACKAGES}

    assert registered == packages
    assert len(miles_ext_kernels._KERNEL_PACKAGES) == len(registered)
