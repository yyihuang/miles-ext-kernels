"""Vendored Cake packages match their FlashInfer provenance, and tools/vendor_cake.py writes and checks them (torch-free;
needs git).

Every ``miles_ext_kernels/<family>/_vendor/`` tree is checked against its ``VENDORED.json``. The tool itself runs here
on a synthetic FlashInfer repository built in a temporary directory.
"""

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path, PurePosixPath

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CAKE_BUILD = Path(__file__).resolve().parent / "_vendor" / "cake_build.py"

_spec = importlib.util.spec_from_file_location("vendor_cake", REPO_ROOT / "tools" / "vendor_cake.py")
vendor_cake = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vendor_cake)

FAST = "flashinfer/experimental/cake/cake_synth_fast"
DET = "flashinfer/cake_synth_det_backend"
GIT = ["git", "-c", "user.name=synthetic", "-c", "user.email=synthetic@example.com", "-c", "commit.gpgsign=false"]


# Every family with a committed _vendor/ tree. The commit that vendors a family's Cake packages adds it here, so a
# tree that was never committed fails in CI rather than leaving nothing to check.
VENDORED_FAMILIES: frozenset[str] = frozenset(
    {
        "conv1d",
        "gated_rmsnorm",
        "gdn",
        "kda",
    }
)


def test_the_vendored_families_are_the_expected_ones():
    assert {family.name for family in vendor_cake.vendored_families()} == VENDORED_FAMILIES


def test_every_vendored_family_matches_its_provenance():
    families = vendor_cake.vendored_families()
    if not families:
        pytest.skip("no family vendors Cake packages yet")
    # one entry per family that vendors packages; the tests below exercise the checker itself
    findings = {family.name: vendor_cake.check_family(family) for family in families}

    assert {family: problems for family, problems in findings.items() if problems} == {}


def _package(revision: str) -> dict[str, bytes]:
    kernel = b"// kernel\n"
    entries = [
        {
            "arch": arch,
            "closure_sha256": "0" * 64,
            "entry": "run",
            "min_cuda": "12.9",
            "name": f"prog_{arch}",
            "nvcc_flags": [],
            "nvcc_flags_if": [],
            "path": f"aot/prog_{arch}.so",
            "sha256": {"csrc/prog.cu": hashlib.sha256(kernel).hexdigest()},
            "sources": ["csrc/prog.cu"],
        }
        for arch in ("sm_100a", "sm_103a")
    ]
    build = {
        "schema": "cake-build/1",
        "generator": {"cake_revision": revision},
        "arch_flags": {},
        "common": {},
        "libraries": entries,
    }
    return {
        "__init__.py": b"from . import cake_build\n",
        "cake_build.py": CAKE_BUILD.read_bytes(),
        "build.json": json.dumps(build).encode(),
        "csrc/prog.cu": kernel,
        "csrc/cake_ffi_checks.h": b"// checks\n",
        ".gitignore": b"aot/\n",
        "aot/prog_sm_100a.so": b"a prebuilt library is never vendored",
    }


@pytest.fixture
def flashinfer(tmp_path) -> Path:
    """A git repository holding two Cake packages at FlashInfer's two positions."""
    root = tmp_path / "flashinfer"
    for fi_path, revision in ((FAST, "f" * 40), (DET, "d" * 40)):
        for relative, data in _package(revision).items():
            path = root / fi_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    (root / "flashinfer" / "experimental" / "cake" / "cake_synth_fast_other.py").write_bytes(b"# not vendored\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run([*GIT, "-C", str(root), "add", "-A", "-f", "."], check=True)
    subprocess.run([*GIT, "-C", str(root), "commit", "-q", "-m", "synthetic"], check=True)
    return root


@pytest.fixture
def vendored(flashinfer, tmp_path):
    """``synth`` vendored from the synthetic repository into a scratch checkout of this repository's layout."""
    repo = tmp_path / "ext"
    (repo / "miles_ext_kernels" / "synth").mkdir(parents=True)
    vendor_cake.vendor(flashinfer, "HEAD", {"synth": [FAST, DET]}, repo_root=repo)
    return repo / "miles_ext_kernels" / "synth"


def _head(flashinfer: Path, path: str = "") -> str:
    return subprocess.run(
        ["git", "-C", str(flashinfer), "rev-parse", f"HEAD{':' + path if path else ''}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _rewrite(family: Path, package: str, relative: str, data: bytes) -> None:
    """Change a vendored file and record its new blob, so only the rule under test sees the change."""
    (family / "_vendor" / package / relative).write_bytes(data)
    manifest_path = family / "_vendor" / "VENDORED.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["packages"][PurePosixPath(package).name]["files"][relative] = vendor_cake.git_blob_sha1(data)
    manifest_path.write_text(json.dumps(manifest))


def test_placement_mirrors_flashinfer():
    assert str(vendor_cake.vendored_dir(FAST)) == "cake/cake_synth_fast"
    assert str(vendor_cake.vendored_dir(DET)) == "cake_synth_det_backend"
    assert str(vendor_cake.vendored_dir("flashinfer/experimental/cake_kda_det_train/")) == "cake_kda_det_train"
    with pytest.raises(ValueError, match="expected a package directly under"):
        vendor_cake.vendored_dir("flashinfer/jit/cake_synth")


def test_vendoring_copies_each_package_byte_identically_with_its_provenance(flashinfer, vendored):
    root = vendored / "_vendor"
    manifest = json.loads((root / "VENDORED.json").read_text())

    assert manifest["fi_repo"] == "https://github.com/flashinfer-ai/flashinfer"
    assert manifest["fi_commit"] == _head(flashinfer)
    assert sorted(manifest["packages"]) == ["cake_synth_det_backend", "cake_synth_fast"]
    for fi_path, destination, revision in (
        (FAST, "cake/cake_synth_fast", "f" * 40),
        (DET, DET.split("/")[-1], "d" * 40),
    ):
        recorded = manifest["packages"][destination.split("/")[-1]]
        assert recorded["fi_path"] == fi_path
        assert recorded["cake_revision"] == revision
        assert sorted(recorded["files"]) == sorted(name for name in _package(revision) if not name.startswith("aot/"))
        for relative, blob in recorded["files"].items():
            assert blob == _head(flashinfer, f"{fi_path}/{relative}")
            assert (root / destination / relative).read_bytes() == (flashinfer / fi_path / relative).read_bytes()
    assert (root / "__init__.py").read_bytes() == b""
    assert (root / "cake" / "__init__.py").read_bytes() == b""
    assert not list(root.rglob("aot"))
    assert not (root / "cake" / "cake_synth_fast_other.py").exists()
    assert vendor_cake.check_family(vendored) == []


def test_vendoring_again_rewrites_the_family_tree(flashinfer, vendored):
    (vendored / "_vendor" / "stale.py").write_bytes(b"")
    vendor_cake.vendor(flashinfer, "HEAD", {"synth": [DET]}, repo_root=vendored.parents[1])

    assert not (vendored / "_vendor" / "stale.py").exists()
    assert not (vendored / "_vendor" / "cake").exists()
    assert vendor_cake.check_family(vendored) == []


@pytest.mark.parametrize(
    ("tamper", "finding"),
    [
        (lambda family: (family / "_vendor/cake/cake_synth_fast/csrc/prog.cu").write_bytes(b"// edited\n"), "differs"),
        (lambda family: (family / "_vendor/cake/cake_synth_fast/extra.py").write_bytes(b""), "not listed"),
        (lambda family: (family / "_vendor/cake_synth_det_backend/csrc/prog.cu").unlink(), "missing"),
        (lambda family: (family / "_vendor/cake/cake_synth_fast/aot").mkdir(), "aot/ directory"),
        (lambda family: (family / "_vendor/cake/__init__.py").write_bytes(b"x = 1\n"), "must be empty"),
        (lambda family: (family / "_vendor/VENDORED.json").unlink(), "VENDORED.json"),
    ],
)
def test_the_checker_reports_a_changed_tree(vendored, tamper, finding):
    tamper(vendored)

    findings = vendor_cake.check_family(vendored)
    assert any(finding in problem for problem in findings), findings


def test_the_checker_reports_a_source_that_differs_from_build_json(vendored):
    _rewrite(vendored, "cake/cake_synth_fast", "csrc/prog.cu", b"// edited\n")

    assert vendor_cake.check_family(vendored) == [
        f"synth/_vendor/cake/cake_synth_fast/csrc/prog.cu: sha256 differs from build.json (prog_{arch})"
        for arch in ("sm_100a", "sm_103a")
    ]


@pytest.mark.parametrize(
    ("edit", "finding"),
    [
        (lambda build: build.update(schema="cake-build/2"), "schema 'cake-build/2'"),
        (lambda build: build["libraries"].pop(), "no libraries for sm_103a"),
        (lambda build: build["generator"].update(cake_revision="e" * 40), "cake_revision"),
    ],
)
def test_the_checker_reports_a_manifest_it_cannot_serve(vendored, edit, finding):
    path = vendored / "_vendor" / "cake_synth_det_backend" / "build.json"
    build = json.loads(path.read_text())
    edit(build)
    _rewrite(vendored, "cake_synth_det_backend", "build.json", json.dumps(build).encode())

    findings = vendor_cake.check_family(vendored)
    assert len(findings) == 1 and finding in findings[0], findings


def test_the_checker_pins_the_flashinfer_loader(vendored):
    findings = vendor_cake.check_family(vendored, cake_build_blob="0" * 40)

    assert len(findings) == 2 and all("is not the loader" in problem for problem in findings), findings


@pytest.mark.parametrize(
    ("source", "env_allowed", "finding"),
    [
        ("import flashinfer.jit\n", False, "imports flashinfer.jit"),
        ("from flashinfer.api_logging import flashinfer_api\n", False, "imports flashinfer.api_logging"),
        ("import os\nflag = os.environ.get('CAKE_KNOB')\n", False, "reads the environment (os.environ)"),
        ("import os\nflag = os.getenv('CAKE_KNOB')\n", False, "reads the environment (os.getenv)"),
        ("from os import environ\n", False, "reads the environment (from os import environ)"),
        ("import os\nflag = os.environ.get('CAKE_KNOB')\n", True, None),
        ("from .. import cake_rmsnorm_det_train_backend\nimport torch\nimport tvm_ffi\n", False, None),
    ],
)
def test_the_checker_finds_flashinfer_imports_and_environment_reads(source, env_allowed, finding):
    findings = vendor_cake._python_findings(source.encode(), "module.py", env_allowed=env_allowed)

    assert findings == ([] if finding is None else [f"module.py:{len(source.splitlines())}: {finding}"])


def test_the_command_line_vendors_and_checks(flashinfer, tmp_path, capsys):
    repo = tmp_path / "cli"
    argv = ["--repo-root", str(repo), "--fi-checkout", str(flashinfer), "--fi-commit", _head(flashinfer)]

    assert vendor_cake.main([*argv, "--package", f"synth={FAST}", "--package", f"synth={DET}"]) == 0
    assert vendor_cake.main(["--repo-root", str(repo), "--check"]) == 0
    (repo / "miles_ext_kernels/synth/_vendor/cake_synth_det_backend/csrc/prog.cu").write_bytes(b"// edited\n")
    assert vendor_cake.main(["--repo-root", str(repo), "--check", "synth"]) == 1
    assert "synth: ok" in capsys.readouterr().out


def test_the_command_line_says_when_there_is_nothing_to_check(tmp_path, capsys):
    assert vendor_cake.main(["--repo-root", str(tmp_path), "--check"]) == 0
    assert capsys.readouterr().out == "nothing to check: no miles_ext_kernels/*/_vendor/ tree\n"
