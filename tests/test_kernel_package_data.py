from fnmatch import fnmatchcase
from importlib.resources import files
from pathlib import Path
import re

import pytest


@pytest.mark.parametrize("backend, name", [
    ("metal", "tq_decode.h"),
    ("metal", "tq_matmul.h"),
    ("cuda", "tq_ops.cu"),
])
def test_kernel_source_resources_are_installed(backend, name):
    resource = files("tyloinfer").joinpath(f"kernels/{backend}/{name}")
    assert resource.is_file()
    assert resource.read_text(encoding="utf-8").strip()


def test_cuda_tq_local_include_dependencies_are_installed():
    root = files("tyloinfer").joinpath("kernels/cuda")
    pending, seen = ["tq_ops.cu"], set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        resource = root.joinpath(name)
        assert resource.is_file(), name
        pending.extend(re.findall(r'^\s*#\s*include\s*"([^"]+)"',
            resource.read_text(encoding="utf-8"), re.MULTILINE))
    assert "tq_decode.cuh" in seen
    assert "tq_prefill_gemm.cuh" in seen


def test_wheel_declares_metal_kernel_resources():
    tomllib = pytest.importorskip("tomllib")
    project = Path(__file__).resolve().parents[1] / "pyproject.toml"
    packages = tomllib.loads(project.read_text(encoding="utf-8"))["tool"]["setuptools"]["package-data"]
    patterns = packages.get("tyloinfer.kernels.metal", [])
    for name in ("tq_decode.h", "tq_matmul.h"):
        assert any(fnmatchcase(name, pattern) for pattern in patterns), name
