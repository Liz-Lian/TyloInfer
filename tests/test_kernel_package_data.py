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


def test_cuda_jit_sources_and_shared_header_includes_resolve():
    from tyloinfer.kernels.cuda import _ext

    repository = Path(__file__).resolve().parents[1]
    assert Path(_ext._KERNEL_DIR) == repository / "csrc/backends/cuda/kernels"
    roots = [
        Path(_ext._REPOSITORY_ROOT),
        Path(_ext._KERNEL_DIR),
        Path(_ext._CUDA_RUNTIME_INCLUDE),
        Path(_ext._CUDA_NATIVE_INCLUDE),
        Path(_ext._CUDA_NATIVE_TENSOR_INCLUDE),
        Path(_ext._CORE_INCLUDE),
        Path(_ext._GGML_INCLUDE),
        Path(_ext._GGML_SOURCE_INCLUDE),
        Path(_ext._GGML_CUDA_INCLUDE),
    ]
    pending, seen = list(_ext._SOURCES), set()
    while pending:
        path = Path(pending.pop())
        if path in seen:
            continue
        seen.add(path)
        assert path.is_file(), path
        for target in re.findall(r'^\s*#\s*include\s*"([^"]+)"',
            path.read_text(encoding="utf-8"), re.MULTILINE):
            resolved = next((base / target for base in [path.parent, *roots]
                if (base / target).is_file()), None)
            assert resolved is not None, f"{path}: {target}"
            pending.append(resolved)
    assert any(path.name == "mfq_cuda.cpp" for path in seen)


def test_cuda_package_keeps_only_tq_kernel_resources():
    root = files("tyloinfer").joinpath("kernels/cuda")
    sources = [path.name for path in root.iterdir()
        if path.suffix in {".cu", ".cuh", ".h", ".cpp"}]
    assert sources
    assert all(name == "mfq_cuda.cpp" or name.startswith("tq_") for name in sources), sources


def test_wheel_declares_metal_kernel_resources():
    tomllib = pytest.importorskip("tomllib")
    project = Path(__file__).resolve().parents[1] / "pyproject.toml"
    packages = tomllib.loads(project.read_text(encoding="utf-8"))["tool"]["setuptools"]["package-data"]
    patterns = packages.get("tyloinfer.kernels.metal", [])
    for name in ("tq_decode.h", "tq_matmul.h"):
        assert any(fnmatchcase(name, pattern) for pattern in patterns), name
