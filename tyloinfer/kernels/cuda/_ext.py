"""Loader for the unified mfq_cuda extension.

All CUDA ops (norm/acc/rope/attention/gated_delta_net/kv_cache/embedding/sampling/nint_matmul) are compiled into a
single extension, matching llama.cpp's single libggml-cuda (avoids one ~45s compile per op).
Cached under ``torch_extensions`` after the first build. Thin per-op wrappers pull their
function from here.

Requires nvcc + (on Windows) MSVC cl on PATH (source vcvars64 or use a Developer Prompt).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

_DIR = os.path.dirname(__file__)
_REPOSITORY_ROOT = str(Path(_DIR).resolve().parents[2])
# The shared CUDA kernels have one source tree, consumed by the native CMake
# targets and by this extension. TQ is built separately by tq.py and its
# sources are also collected into wheels from that tree.
_CUDA_BACKEND = Path(_REPOSITORY_ROOT) / "csrc" / "backends" / "cuda"
_KERNEL_DIR = str(Path(_REPOSITORY_ROOT) / "csrc" / "kernels" / "cuda")
_CUDA_RUNTIME_INCLUDE = str(_CUDA_BACKEND / "include")
_CUDA_NATIVE_INCLUDE = str(Path(_REPOSITORY_ROOT) / "csrc" / "native" / "cuda")
_CUDA_NATIVE_TENSOR_INCLUDE = os.path.join(_CUDA_NATIVE_INCLUDE, "tensor")
_CSRC_ROOT = str(Path(_REPOSITORY_ROOT) / "csrc")
_GGML_ROOT = Path(_REPOSITORY_ROOT) / "csrc" / "components" / "ggml"
_GGML_INCLUDE = str(_GGML_ROOT / "include")
_GGML_SOURCE_INCLUDE = str(_GGML_ROOT / "src")
_GGML_CUDA_INCLUDE = str(_GGML_ROOT / "src" / "ggml-cuda")
# This is the closure of the functions bound by mfq_cuda.cpp, plus providers
# called from those units.  It is deliberately not the native runtime's full
# TYLOINFER_CUDA_KERNEL_SOURCES list: attention_mma.cu requires ggml CUDA objects,
# and gdn_decode_window.cu is a native-runtime-only decode path.  In contrast,
# nvq_matmul.cu calls the provider in nvq_routed_reuse.cu, so both belong here.
_SOURCES = [
    os.path.join(_KERNEL_DIR, name)
    for name in (
        "norm.cu",
        "acc.cu",
        "rope.cu",
        "attention.cu",
        "gated_delta_net.cu",
        "kv_cache.cu",
        "activation.cu",
        "embedding.cu",
        "selected_attention.cu",
        "qwen4_exp.cu",
        "glm5_next.cu",
        "sampling.cu",
        "ssm_conv.cu",
        "moe.cu",
        "mx_matmul.cu",
        "fp8_sq.cu",
        "mxfp4_sq.cu",
        "nint_matmul.cu",
        "nint_short_decode.cu",
        "nint_q8_decode.cu",
        "nvq_matmul.cu",
        "nvq_routed_reuse.cu",
        "nepq.cu",
        "nepq_residual.cu",
    )
] + [os.path.join(_DIR, "mfq_cuda.cpp"), os.path.join(_CUDA_NATIVE_INCLUDE, "env.cpp")]
_module = None


def _ensure_msvc() -> None:
    if os.name != "nt" or shutil.which("cl"):
        return
    roots = tuple(
        Path(value) / "Microsoft Visual Studio"
        for name in ("ProgramFiles", "ProgramFiles(x86)")
        if (value := os.environ.get(name))
    )
    candidates = [
        path
        for root in roots
        if root.exists()
        for path in root.glob("*/*/VC/Auxiliary/Build/vcvars64.bat")
    ]
    if not candidates:
        raise RuntimeError("MSVC vcvars64.bat was not found")
    vcvars = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    command = f'cmd.exe /d /c "call ""{vcvars}"" >nul && set"'
    output = subprocess.check_output(
        ["powershell.exe", "-NoProfile", "-Command", command],
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    for line in output.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            os.environ[key] = value
    if not shutil.which("cl"):
        raise RuntimeError(f"vcvars64 did not expose cl.exe: {vcvars}")


def ext():
    global _module
    if _module is None:
        import torch

        if torch.version.hip:
            # AMD ROCm host: build the hipify-translated kernels instead.
            from tyloinfer.kernels.hip._ext import ext as _hip_ext

            _module = _hip_ext()
            return _module
        _ensure_msvc()
        import torch  # noqa: F401  ensure torch imported (cpp_extension depends on it)
        from torch.utils.cpp_extension import load

        _module = load(
            name="mfq_cuda",
            sources=_SOURCES,
            extra_include_paths=[
                _REPOSITORY_ROOT,
                str(_CUDA_BACKEND),
                _KERNEL_DIR,
                _CUDA_RUNTIME_INCLUDE,
                _CUDA_NATIVE_INCLUDE,
                _CUDA_NATIVE_TENSOR_INCLUDE,
                _CSRC_ROOT,
                _GGML_INCLUDE,
                _GGML_SOURCE_INCLUDE,
                _GGML_CUDA_INCLUDE,
            ],
            # CUDA 13 CCCL headers reject MSVC's traditional preprocessor.
            extra_cflags=(
                ["/std:c++20", "/Zc:preprocessor", "/DTYLOINFER_ENABLE_MFQ_COMPAT=1"]
                if os.name == "nt"
                else ["-std=c++20", "-DTYLOINFER_ENABLE_MFQ_COMPAT=1"]
            ),
            extra_cuda_cflags=[
                *(["-Xcompiler=/Zc:preprocessor"] if os.name == "nt" else []),
                # The Torch bindings export legacy MFQ ops as well as shared ops.
                "-DTYLOINFER_ENABLE_MFQ_COMPAT=1",
                "-O3",
                "--use_fast_math",
                "--extended-lambda",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-std=c++20",
            ],
            extra_ldflags=["cublas.lib"] if os.name == "nt" else ["-lcublas"],
            verbose=False,
        )
    return _module
