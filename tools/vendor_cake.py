#!/usr/bin/env python3
"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Vendor Cake packages of a FlashInfer commit into ``miles_ext_kernels/<family>/_vendor/``, byte-identical, and check
vendored trees.

Vendor (needs git and Python only)::

    python tools/vendor_cake.py --fi-checkout ../flashinfer --fi-commit <sha> \\
        --package gdn=flashinfer/experimental/cake/cake_gdn_train [--package FAMILY=FI_PATH ...]

Every file of each package is read from the FlashInfer commit (``git ls-tree`` / ``git cat-file``, not the working
tree), except ``aot/`` (prebuilt libraries) and ``__pycache__/``. A package keeps its position relative to
FlashInfer's: ``flashinfer/<name>`` and ``flashinfer/experimental/<name>`` go to ``_vendor/<name>/``,
``flashinfer/experimental/cake/<name>`` to ``_vendor/cake/<name>/`` (with an empty ``cake/__init__.py``), so the
sibling imports between packages keep working. Each named family's ``_vendor/`` is rewritten whole, with an empty
``_vendor/__init__.py`` and the provenance ``_vendor/VENDORED.json``::

    {"fi_repo": ..., "fi_commit": <40-hex>,
     "packages": {<name>: {"fi_path": ..., "cake_revision": <build.json generator.cake_revision>,
                           "files": {<package-relative path>: <git blob sha1>}}}}

then checked as ``--check`` does.

Check::

    python tools/vendor_cake.py --check [FAMILY ...]

:func:`check_family` verifies a family's vendored tree against its ``VENDORED.json``: the exact file set and every
blob, ``cake_build.py`` equal to FlashInfer's loader, no ``aot/``, ``build.json`` schema ``cake-build/1`` with its
recorded Cake revision, every source digest of the manifest, sm_100a and sm_103a libraries, and no ``flashinfer``
import or environment read in the vendored Python modules (allow-listed exceptions in :data:`ENV_READ_ALLOWED`).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = "miles_ext_kernels"
FI_REPO = "https://github.com/flashinfer-ai/flashinfer"
MANIFEST = "VENDORED.json"
# git blob of FlashInfer's cake_build.py, the loader every Cake package carries byte-identically
CAKE_BUILD_BLOB = "00756137eb159e30a77907eda84fd245123d87ea"
BUILD_SCHEMA = "cake-build/1"
REQUIRED_ARCHES = frozenset({"sm_100a", "sm_103a"})
# family -> package-relative modules that may read the environment (never imported by this repo)
ENV_READ_ALLOWED: dict[str, frozenset[str]] = {"conv1d": frozenset({"kcp.py"})}

_FAMILY = re.compile(r"[a-z][a-z0-9_]*")
_PARENTS = {"flashinfer": "", "flashinfer/experimental": "", "flashinfer/experimental/cake": "cake"}
_SKIPPED_DIRS = {"__pycache__"}


def git_blob_sha1(data: bytes) -> str:
    """The git object id of a blob holding ``data``."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def vendored_dir(fi_path: str) -> PurePosixPath:
    """``<name>`` or ``cake/<name>``: where the FlashInfer package ``fi_path`` goes under ``_vendor/``."""
    path = PurePosixPath(fi_path.strip("/"))
    parent = str(path.parent)
    if parent not in _PARENTS:
        raise ValueError(f"{fi_path}: expected a package directly under one of {sorted(_PARENTS)}")
    return PurePosixPath(_PARENTS[parent]) / path.name if _PARENTS[parent] else PurePosixPath(path.name)


# ---------------------------------------------------------------------------------------------------------------
# vendoring


def _git(checkout: Path, *args: str) -> bytes:
    proc = subprocess.run(["git", "-C", str(checkout), *args], capture_output=True, check=False)
    if proc.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed (exit {proc.returncode}): {proc.stderr.decode().strip()}")
    return proc.stdout


def _package_tree(checkout: Path, commit: str, fi_path: str) -> list[tuple[str, str, str]]:
    """``(mode, blob, package-relative path)`` of every vendored file of ``fi_path`` at ``commit``."""
    prefix = fi_path.strip("/") + "/"
    files = []
    for record in _git(checkout, "ls-tree", "-r", "-z", "--full-tree", commit, "--", prefix[:-1]).split(b"\0"):
        if not record:
            continue
        meta, path = record.decode().split("\t", 1)
        if not path.startswith(prefix):
            continue
        mode, kind, blob = meta.split()
        relative = path[len(prefix) :]
        parts = PurePosixPath(relative).parts
        if parts[0] == "aot" or _SKIPPED_DIRS.intersection(parts):
            continue
        if kind != "blob" or mode not in ("100644", "100755"):
            raise RuntimeError(f"{path}: cannot vendor a {kind} of mode {mode}")
        files.append((mode, blob, relative))
    if not files:
        raise RuntimeError(f"{fi_path} has no files at {commit}")
    return files


class _BlobReader:
    """``git cat-file --batch``: one process for every blob of a run."""

    def __init__(self, checkout: Path) -> None:
        self._proc = subprocess.Popen(
            ["git", "-C", str(checkout), "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )

    def read(self, blob: str) -> bytes:
        self._proc.stdin.write(blob.encode() + b"\n")
        self._proc.stdin.flush()
        header = self._proc.stdout.readline().split()
        if len(header) != 3 or header[0].decode() != blob or header[1] != b"blob":
            raise RuntimeError(f"git cat-file cannot read blob {blob}: {b' '.join(header).decode()}")
        data = self._proc.stdout.read(int(header[2]))
        self._proc.stdout.read(1)  # the newline after the content
        return data

    def close(self) -> None:
        self._proc.stdin.close()
        self._proc.wait()


def vendor(
    checkout: Path, commit: str, packages: dict[str, list[str]], *, repo_root: Path = REPO_ROOT, fi_repo: str = FI_REPO
) -> dict[str, dict[str, Any]]:
    """Vendor ``packages`` (family -> FlashInfer package paths) of ``commit``; returns family -> ``VENDORED.json``."""
    full_commit = _git(checkout, "rev-parse", "--verify", f"{commit}^{{commit}}").decode().strip()
    reader = _BlobReader(checkout)
    manifests = {}
    try:
        for family, fi_paths in packages.items():
            if not _FAMILY.fullmatch(family):
                raise ValueError(f"family {family!r} is not a package name")
            placements = [(fi_path, vendored_dir(fi_path)) for fi_path in fi_paths]
            names = [destination.name for _, destination in placements]
            if len(set(names)) != len(names):
                raise ValueError(f"{family}: two packages share a name in {names}")
            vendor_root = repo_root / PACKAGE_ROOT / family / "_vendor"
            if vendor_root.exists():
                shutil.rmtree(vendor_root)
            vendor_root.mkdir(parents=True)
            (vendor_root / "__init__.py").write_bytes(b"")
            entries = {}
            for fi_path, destination in placements:
                if destination.parent.name == "cake":
                    (vendor_root / "cake").mkdir(exist_ok=True)
                    (vendor_root / "cake" / "__init__.py").write_bytes(b"")
                files = {}
                build = None
                for mode, blob, relative in _package_tree(checkout, full_commit, fi_path):
                    data = reader.read(blob)
                    if git_blob_sha1(data) != blob:
                        raise RuntimeError(f"{fi_path}/{relative}: content does not hash to blob {blob}")
                    target = vendor_root / destination / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
                    if mode == "100755":
                        target.chmod(0o755)
                    files[relative] = blob
                    if relative == "build.json":
                        build = json.loads(data)
                if build is None:
                    raise RuntimeError(f"{fi_path} has no build.json: not a Cake package")
                entries[destination.name] = {
                    "fi_path": fi_path.strip("/"),
                    "cake_revision": build["generator"]["cake_revision"],
                    "files": dict(sorted(files.items())),
                }
            manifest = {"fi_repo": fi_repo, "fi_commit": full_commit, "packages": dict(sorted(entries.items()))}
            (vendor_root / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            manifests[family] = manifest
    finally:
        reader.close()
    return manifests


# ---------------------------------------------------------------------------------------------------------------
# checking


def vendored_families(repo_root: Path = REPO_ROOT) -> list[Path]:
    """Every family directory with a ``_vendor/`` tree."""
    return sorted(path.parent for path in (repo_root / PACKAGE_ROOT).glob("*/_vendor") if path.is_dir())


def _tree_files(root: Path) -> set[str]:
    files = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if _SKIPPED_DIRS.intersection(relative.parts) or path.is_dir():
            continue
        files.add(relative.as_posix())
    return files


def _python_findings(source: bytes, where: str, *, env_allowed: bool) -> list[str]:
    """``flashinfer`` imports and environment reads of one module."""
    try:
        tree = ast.parse(source, filename=where)
    except SyntaxError as exc:
        return [f"{where}: does not parse: {exc}"]
    findings = []
    for node in ast.walk(tree):
        modules = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules = [node.module]
            if node.module == "os" and not env_allowed:
                findings.extend(
                    f"{where}:{node.lineno}: reads the environment (from os import {alias.name})"
                    for alias in node.names
                    if alias.name in ("environ", "environb", "getenv", "getenvb")
                )
        findings.extend(
            f"{where}:{node.lineno}: imports {module}"
            for module in modules
            if module == "flashinfer" or module.startswith("flashinfer.")
        )
        if (
            not env_allowed
            and isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "os"
            and node.attr in ("environ", "environb", "getenv", "getenvb")
        ):
            findings.append(f"{where}:{node.lineno}: reads the environment (os.{node.attr})")
    return findings


def _check_build(package_dir: Path, recorded: dict[str, Any], where: str) -> list[str]:
    try:
        build = json.loads((package_dir / "build.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"{where}/build.json: {exc}"]
    if build.get("schema") != BUILD_SCHEMA:
        return [f"{where}/build.json: schema {build.get('schema')!r}, expected {BUILD_SCHEMA!r}"]
    findings = []
    revision = build.get("generator", {}).get("cake_revision")
    if revision != recorded.get("cake_revision"):
        findings.append(
            f"{where}: VENDORED.json cake_revision {recorded.get('cake_revision')!r} != build.json {revision!r}"
        )
    digests: dict[str, str | None] = {}
    arches = set()
    for entry in build.get("libraries", []):
        arches.add(entry["arch"])
        for source in entry["sources"]:
            if source not in digests:
                path = package_dir / source
                digests[source] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            if digests[source] is None:
                findings.append(f"{where}/{source}: missing, a source of {entry['name']}")
            elif digests[source] != entry["sha256"].get(source):
                findings.append(f"{where}/{source}: sha256 differs from build.json ({entry['name']})")
    if missing := sorted(REQUIRED_ARCHES - arches):
        findings.append(f"{where}/build.json: no libraries for {', '.join(missing)}")
    return findings


def check_family(family_dir: Path, *, cake_build_blob: str = CAKE_BUILD_BLOB) -> list[str]:
    """Findings of one family's ``_vendor/`` tree against its ``VENDORED.json``; empty when it is intact."""
    family = family_dir.name
    vendor_root = family_dir / "_vendor"
    try:
        manifest = json.loads((vendor_root / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"{family}: {MANIFEST}: {exc}"]
    findings = []
    if not isinstance(manifest.get("fi_repo"), str) or not manifest["fi_repo"]:
        findings.append(f"{family}: {MANIFEST} has no fi_repo")
    if not re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("fi_commit"))):
        findings.append(f"{family}: {MANIFEST} fi_commit {manifest.get('fi_commit')!r} is not a 40-hex commit")
    packages = manifest.get("packages")
    if not isinstance(packages, dict) or not packages:
        return findings + [f"{family}: {MANIFEST} lists no packages"]

    expected = {"__init__.py", MANIFEST}
    for name, recorded in sorted(packages.items()):
        try:
            destination = vendored_dir(recorded["fi_path"])
        except (KeyError, ValueError) as exc:
            findings.append(f"{family}/{name}: fi_path: {exc}")
            continue
        if destination.name != name:
            findings.append(f"{family}/{name}: fi_path {recorded['fi_path']} names package {destination.name}")
            continue
        if destination.parent.name == "cake":
            expected.add("cake/__init__.py")
        where = f"{family}/_vendor/{destination}"
        package_dir = vendor_root / destination
        files = recorded.get("files") or {}
        for relative, blob in sorted(files.items()):
            if PurePosixPath(relative).parts[0] == "aot":
                findings.append(f"{where}/{relative}: prebuilt aot/ content is not vendored")
            expected.add(f"{destination}/{relative}")
            path = package_dir / relative
            if path.is_file() and git_blob_sha1(path.read_bytes()) != blob:
                findings.append(f"{where}/{relative}: content differs from FlashInfer blob {blob}")
        if files.get("cake_build.py") != cake_build_blob:
            findings.append(
                f"{where}/cake_build.py: blob {files.get('cake_build.py')} is not the loader {cake_build_blob}"
            )
        if "build.json" not in files:
            findings.append(f"{where}: no build.json")
        elif (package_dir / "build.json").is_file():
            findings.extend(_check_build(package_dir, recorded, where))
        allowed = ENV_READ_ALLOWED.get(family, frozenset())
        for relative in sorted(files):
            path = package_dir / relative
            # cake_build.py is pinned by its blob; its lazy flashinfer.jit import is the hook cake_native replaces
            if relative.endswith(".py") and relative != "cake_build.py" and path.is_file():
                findings.extend(
                    _python_findings(path.read_bytes(), f"{where}/{relative}", env_allowed=relative in allowed)
                )

    actual = _tree_files(vendor_root)
    findings.extend(f"{family}/_vendor/{path}: missing" for path in sorted(expected - actual))
    findings.extend(f"{family}/_vendor/{path}: not listed in {MANIFEST}" for path in sorted(actual - expected))
    for marker in ("__init__.py", "cake/__init__.py"):
        path = vendor_root / marker
        if path.is_file() and path.stat().st_size:
            findings.append(f"{family}/_vendor/{marker}: must be empty")
    findings.extend(
        f"{family}/_vendor/{path.relative_to(vendor_root).as_posix()}: aot/ directory"
        for path in vendor_root.rglob("aot")
        if path.is_dir()
    )
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Vendor Cake packages of a FlashInfer commit, or check vendored trees."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="this repository's checkout")
    parser.add_argument("--check", nargs="*", metavar="FAMILY", help="check vendored families (default: all)")
    parser.add_argument("--fi-checkout", type=Path, help="a FlashInfer git checkout holding --fi-commit")
    parser.add_argument("--fi-commit", help="the FlashInfer commit to vendor from")
    parser.add_argument("--fi-repo", default=FI_REPO, help=f"recorded as fi_repo (default {FI_REPO})")
    parser.add_argument(
        "--package",
        action="append",
        default=[],
        metavar="FAMILY=FI_PATH",
        help="vendor the FlashInfer package FI_PATH (e.g. flashinfer/experimental/cake/cake_gdn_train) for FAMILY",
    )
    args = parser.parse_args(argv)

    if args.check is not None:
        if args.package:
            parser.error("--check does not vendor; drop --package")
        families = [args.repo_root / PACKAGE_ROOT / name for name in args.check] or vendored_families(args.repo_root)
    else:
        if not (args.fi_checkout and args.fi_commit and args.package):
            parser.error("vendoring needs --fi-checkout, --fi-commit and at least one --package")
        packages: dict[str, list[str]] = {}
        for item in args.package:
            family, sep, fi_path = item.partition("=")
            if not sep or not fi_path:
                parser.error(f"--package {item!r}: expected FAMILY=FI_PATH")
            packages.setdefault(family, []).append(fi_path)
        manifests = vendor(args.fi_checkout, args.fi_commit, packages, repo_root=args.repo_root, fi_repo=args.fi_repo)
        for family, manifest in manifests.items():
            for name, entry in manifest["packages"].items():
                print(
                    f"{family}: {name} <- {entry['fi_path']} ({len(entry['files'])} files) @ {manifest['fi_commit']}"
                )
        families = [args.repo_root / PACKAGE_ROOT / family for family in manifests]

    failed = False
    for family_dir in families:
        findings = check_family(family_dir)
        print(f"{family_dir.name}: {'ok' if not findings else f'{len(findings)} findings'}")
        for finding in findings:
            print(f"  {finding}")
        failed = failed or bool(findings)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
