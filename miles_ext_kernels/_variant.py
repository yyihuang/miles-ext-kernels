"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

The ``--cake-<kernel>-variant {det,fast}`` choice of the kernel packages that ship two variants (torch-free).

``det`` is the bit-deterministic Cake kernel, ``fast`` its non-deterministic sibling. Unset, a package runs ``fast``,
or ``det`` when miles runs in deterministic mode. A package resolves its flag in ``validate_args`` with
:func:`resolve`, which writes the result back, so ``configure`` always sees ``det`` or ``fast``. A call that ``fast``
does not serve runs ``det`` and says so once with :func:`warn_det_fallback`.
"""

from __future__ import annotations

import logging
from argparse import Namespace

__all__ = ["VARIANTS", "add_variant_argument", "deterministic_mode", "resolve", "warn_det_fallback"]

VARIANTS = ("det", "fast")

logger = logging.getLogger("miles_ext_kernels")
_warned: set[tuple[str, str]] = set()


def add_variant_argument(group, flag: str, backend: str) -> None:
    """Register ``flag`` (``--cake-<kernel>-variant``) for the kernels selected by ``backend``."""
    group.add_argument(
        flag,
        choices=VARIANTS,
        default=None,
        help=(
            f"Kernel variant under {backend}: 'det' is bit-deterministic, 'fast' is the non-deterministic sibling. "
            "Default: 'fast', or 'det' under --deterministic-mode."
        ),
    )


def deterministic_mode(args) -> bool:
    """Does miles run in deterministic mode?

    miles hands ``validate_args`` its parsed flat namespace (``miles_validate_args``), where the Megatron flag
    ``--deterministic-mode`` and the FSDP backend's field of the same name are ``args.deterministic_mode``. The typed
    configuration built afterwards carries it as ``args.backend.deterministic_mode``. The torchtitan backend has
    neither, so it never runs in deterministic mode.
    """
    value = getattr(args, "deterministic_mode", None)
    if value is None:
        value = getattr(getattr(args, "backend", None), "deterministic_mode", False)
    return bool(value)


def resolve(
    args: Namespace, flag: str, backend_selected: bool, fast_unordered: bool, *, backend: str = "its cake backend"
) -> str | None:
    """Resolve the variant ``flag`` of ``args`` and write it back; ``None`` when the cake backend is not selected.

    ``fast_unordered``: ``fast`` reduces in a run-dependent order, so deterministic mode cannot run it. Raises
    ``ValueError`` for a variant given without its backend (``backend`` names it in the message) and for an explicit
    unordered ``fast`` under deterministic mode.
    """
    dest = flag.lstrip("-").replace("-", "_")
    variant = getattr(args, dest, None)
    if not backend_selected:
        if variant is not None:
            raise ValueError(f"{flag} requires {backend}")
        return None
    if variant is not None and variant not in VARIANTS:
        raise ValueError(f"{flag} must be one of {VARIANTS}, not {variant!r}")
    deterministic = deterministic_mode(args)
    if variant is None:
        variant = "det" if deterministic else "fast"
    elif variant == "fast" and deterministic and fast_unordered:
        raise ValueError(
            f"{flag} fast is not bit-reproducible (its reductions run in a run-dependent order): "
            f"--deterministic-mode needs {flag} det, or leave {flag} unset"
        )
    setattr(args, dest, variant)
    return variant


def warn_det_fallback(family: str, reason: str) -> None:
    """Log once per ``(family, reason)`` that a call ``fast`` does not serve runs ``det``."""
    key = (family, reason)
    if key in _warned:
        return
    _warned.add(key)
    logger.warning("miles_ext_kernels.%s: the fast variant does not serve %s; running det", family, reason)
