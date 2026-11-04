"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

The loader of the vendored Cake packages (``<kernel>/_vendor/``): plain nvcc and tvm-ffi, no FlashInfer.

Every vendored package keeps FlashInfer's byte-identical ``cake_build.py`` and its ``build.json`` manifest (schema
``cake-build/1``). Without ``aot/<name>.so``, ``cake_build.load_library(name)`` returns
``jit_spec(name).build_and_load()``, and ``jit_spec``, ``jit_specs`` and ``record_cuda_version`` look up the module
attribute ``_jit_toolchain`` at call time. :func:`install` rebinds it to this module's
``(gen_jit_spec, get_cuda_version)``, so the package compiles its manifest entries with nvcc and loads them with
tvm-ffi.

The command is the one of FlashInfer's vendored-build gate (``compile_command`` of
``scripts/check_cake_build_manifests.py``): nvcc, ``common.nvcc_flags``, the ``common.nvcc_flags_if`` items matching
the toolkit, the flags ``cake_build.jit_spec`` passes (``arch_flags[arch]``, the entry's ``nvcc_flags`` and its
matching ``nvcc_flags_if``), ``-I`` for ``common.include_dirs`` and for the tvm-ffi and dlpack headers, the sources,
``common.link_flags`` and ``-o``. ``cake_build.jit_spec`` leaves out the ``common`` flags because FlashInfer's JIT
adds its own defaults; this loader adds them.

A library is cached as ``${TVM_FFI_CACHE_DIR:-~/.cache/tvm-ffi}/miles_ext_kernels/<package>/<name>_<key16>.so``
(``<name>`` is ``cake_build``'s ``<entry>_<closure20>``), never in the package directory, which may be read-only. The
key hashes the flags, the full nvcc version, the tvm-ffi version and the machine, so builds of different toolkits or
CPU architectures in one shared checkout do not collide. All ranks may start a build at once: one compiles under a
file lock into a temporary file and renames it into place, the others wait and load it.

Importing this module does not import torch or tvm_ffi.
"""

from __future__ import annotations

import fcntl
import functools
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import uuid
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType
from typing import Any, NamedTuple

__all__ = ["CudaVersion", "JitSpec", "cache_dir", "find_nvcc", "get_cuda_version", "install", "prebuild"]

_CUDA_CONSTRAINT = re.compile(r"(==|>=)([0-9]+)\.([0-9]+)")
_NVCC_RELEASE = re.compile(r"release (\d+)\.(\d+)")
_NVCC_BUILD = re.compile(r"\bV(\d+\.\d+\.\d+)")
_STDERR_LINES = 40  # kept in the error of a failed compile


class CudaVersion(NamedTuple):
    """The toolkit of ``nvcc --version``: ``release X.Y`` and the full ``VX.Y.Z``."""

    major: int
    minor: int
    full: str

    @property
    def cudart(self) -> int:
        """``CUDART_VERSION`` of the toolkit: ``1000 * major + 10 * minor``."""
        return 1000 * self.major + 10 * self.minor


def find_nvcc() -> str:
    """``$CUDA_HOME/bin/nvcc``, else ``nvcc`` on ``PATH``."""
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        return str(Path(cuda_home) / "bin" / "nvcc")
    found = shutil.which("nvcc")
    if found is None:
        raise RuntimeError("no nvcc to build the Cake kernels: set CUDA_HOME or put nvcc on PATH")
    return found


@functools.lru_cache(maxsize=None)
def _nvcc_version(nvcc: str, stamp: tuple[int, int]) -> CudaVersion:
    # ``stamp`` (mtime, size) re-runs nvcc when the file at that path changes
    try:
        proc = subprocess.run([nvcc, "--version"], capture_output=True, text=True, check=False)
    except OSError as exc:
        raise RuntimeError(f"cannot run {nvcc} --version: {exc}") from exc
    release = _NVCC_RELEASE.search(proc.stdout)
    build = _NVCC_BUILD.search(proc.stdout)
    if proc.returncode or release is None or build is None:
        raise RuntimeError(
            f"{nvcc} --version (exit {proc.returncode}) reports no 'release X.Y, VX.Y.Z':\n{proc.stdout}{proc.stderr}"
        )
    return CudaVersion(int(release.group(1)), int(release.group(2)), build.group(1))


def _toolkit() -> tuple[str, CudaVersion]:
    nvcc = find_nvcc()
    try:
        stat = os.stat(nvcc)
    except OSError as exc:
        raise RuntimeError(f"cannot run {nvcc}: {exc}") from exc
    return nvcc, _nvcc_version(nvcc, (stat.st_mtime_ns, stat.st_size))


def get_cuda_version() -> CudaVersion:
    """The toolkit of :func:`find_nvcc` (``.major``, ``.minor`` as ``cake_build`` reads them, and ``.full``)."""
    return _toolkit()[1]


def _conditional_flags(items: Iterable[dict[str, Any]], cuda: tuple[int, int]) -> list[str]:
    """The flags of every ``nvcc_flags_if`` item whose ``==X.Y`` / ``>=X.Y`` constraint the toolkit meets."""
    flags: list[str] = []
    for item in items:
        match = _CUDA_CONSTRAINT.fullmatch(str(item["cuda"]))
        if match is None:
            raise ValueError(f"unsupported CUDA constraint {item['cuda']!r}; expected '==X.Y' or '>=X.Y'")
        operator, major, minor = match.groups()
        version = (int(major), int(minor))
        if (cuda == version) if operator == "==" else (cuda >= version):
            flags.extend(item["flags"])
    return flags


def cache_dir() -> Path:
    """``$TVM_FFI_CACHE_DIR`` (tvm-ffi's own cache variable), else ``~/.cache/tvm-ffi``, + ``miles_ext_kernels``."""
    root = os.environ.get("TVM_FFI_CACHE_DIR", str(Path("~/.cache/tvm-ffi").expanduser()))
    return Path(root).expanduser() / "miles_ext_kernels"


class JitSpec:
    """One manifest library of a vendored package: its nvcc build into the cache and its tvm-ffi load.

    The interface ``cake_build`` uses of FlashInfer's JIT spec: :meth:`build`, :meth:`get_library_path` and
    :meth:`build_and_load`.
    """

    def __init__(
        self,
        package: str,
        name: str,
        *,
        nvcc: str,
        cuda: CudaVersion,
        flags: list[str],
        include_dirs: list[str],
        sources: list[str],
        ldflags: list[str],
    ) -> None:
        import tvm_ffi.libinfo

        self.package = package
        self.name = name
        self.nvcc = nvcc
        self.cuda = cuda
        self.flags = flags
        self.include_dirs = [
            *include_dirs,
            tvm_ffi.libinfo.find_include_path(),
            tvm_ffi.libinfo.find_dlpack_include_path(),
        ]
        self.sources = sources
        self.ldflags = ldflags
        # Paths stay out of the key: the name carries the closure digest of the sources, and the tvm-ffi version
        # pins its headers, so checkouts in different directories share one entry.
        key = json.dumps(
            {
                "flags": [*flags, *ldflags],
                "nvcc": cuda.full,
                "tvm_ffi": tvm_ffi.__version__,
                "machine": platform.machine(),
            },
            sort_keys=True,
        )
        self.key = hashlib.sha256(key.encode()).hexdigest()
        self._path = cache_dir() / package / f"{name}_{self.key[:16]}.so"

    def command(self, output: Path) -> list[str]:
        """The nvcc command writing the library to ``output``."""
        return [
            self.nvcc,
            *self.flags,
            *(f"-I{directory}" for directory in self.include_dirs),
            *self.sources,
            *self.ldflags,
            "-o",
            str(output),
        ]

    def get_library_path(self) -> Path:
        """Where the library is cached (built or not)."""
        return self._path

    def build(self) -> Path:
        """Compile the library unless it is cached; return its path."""
        path = self._path
        if path.is_file():
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path.with_suffix(".lock"), "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)  # released when the file closes
            if path.is_file():  # another rank built it while this one waited
                return path
            # unique across hosts: a lock that is local to each node still lets two nodes build at once
            temporary = path.with_name(f"{path.stem}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp.so")
            command = self.command(temporary)
            try:
                try:
                    proc = subprocess.run(command, capture_output=True, text=True, check=False)
                except OSError as exc:
                    raise RuntimeError(f"cannot run {self.nvcc} to build {self.name}: {exc}") from exc
                if proc.returncode or not temporary.is_file():
                    stderr = "\n".join(proc.stderr.splitlines()[:_STDERR_LINES])
                    raise RuntimeError(
                        f"nvcc failed to build {self.name} (exit {proc.returncode}):\n{' '.join(command)}\n{stderr}"
                    )
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        return path

    def build_and_load(self) -> Any:
        """The loaded tvm-ffi module, checked to be compiled against the toolkit's CUDART."""
        path = self.build()
        import tvm_ffi

        module = tvm_ffi.load_module(str(path))
        built = int(module.build_cuda_version())
        if built != self.cuda.cudart:
            raise RuntimeError(
                f"{path} was compiled against CUDART {built}, but {self.nvcc} is CUDA {self.cuda.full} "
                f"(CUDART {self.cuda.cudart})"
            )
        return module


def install(cake_build_module: ModuleType) -> ModuleType:
    """Make the vendored ``cake_build`` module build its libraries through this loader; idempotent."""
    if not callable(getattr(cake_build_module, "_jit_toolchain", None)):
        raise TypeError(f"{cake_build_module.__name__} has no _jit_toolchain hook: not a cake-build/1 loader")
    package = Path(cake_build_module.__file__).parent.name

    def gen_jit_spec(
        name: str,
        sources: list[Path],
        *,
        extra_cuda_cflags: list[str],
        extra_ldflags: list[str],
        extra_include_paths: list[Path],
        use_fast_math: bool = False,
    ) -> JitSpec:
        if use_fast_math:
            raise ValueError(f"{name}: use_fast_math must come from the manifest's flags")
        nvcc, cuda = _toolkit()
        common = cake_build_module.manifest()["common"]
        return JitSpec(
            package,
            name,
            nvcc=nvcc,
            cuda=cuda,
            flags=[
                *common["nvcc_flags"],
                *_conditional_flags(common["nvcc_flags_if"], (cuda.major, cuda.minor)),
                *extra_cuda_cflags,
            ],
            include_dirs=[str(path) for path in extra_include_paths],
            sources=[str(path) for path in sources],
            ldflags=list(extra_ldflags),
        )

    def _jit_toolchain(name: str) -> tuple[Any, Any]:
        return gen_jit_spec, get_cuda_version

    cake_build_module._jit_toolchain = _jit_toolchain
    return cake_build_module


def prebuild(cake_build_module: ModuleType, names: Iterable[str], max_workers: int | None = None) -> list[Path]:
    """Compile the libraries ``names`` of a vendored package ahead of their first call, ``max_workers`` at a time.

    Installs this loader on the package. Libraries prebuilt under ``aot/`` are skipped; a library whose ``min_cuda``
    exceeds the toolkit raises, as loading it would. Returns the paths of the libraries in the cache.
    """
    install(cake_build_module)
    pending = [name for name in dict.fromkeys(names) if not cake_build_module.prebuilt_path(name).is_file()]
    if not pending:
        return []
    workers = max_workers or min(8, len(os.sched_getaffinity(0)))
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(pending)))) as pool:
        return list(pool.map(lambda name: cake_build_module.jit_spec(name).build(), pending))
