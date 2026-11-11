# Self-contained releases

MFQ desktop releases support macOS and Windows. Build scripts live in
`packaging/` and write artifacts to `packaging/dist/`.

The maintained entry points are `build_release_mac.sh` and
`build_release_windows.ps1`. Supporting launcher and validation code is embedded
in these scripts; the PyInstaller hook is generated in the build directory. The
Metal compiler tool belongs to `cpp_runtime/backends/metal/tools/`.

The packages include the server, Python environment, native runtime, and
required runtime libraries. `mfq serve` or TyloInfer manages the native
sidecars.

## macOS

The macOS release packages the `mfq` service CLI and TyloInfer into an
Apple Silicon DMG. It includes:

- the unified `mfq` CLI with service and model management;
- the private native `mfq-decode-metal` sidecar;
- `libmlx`, `libjaccl`, `mlx.metallib`, and the AVFoundation video bridge;
- TyloInfer and its production Web UI, without PyTorch.

Build from a clean checkout on Apple Silicon:

```shell
cd packaging
./build_release_mac.sh
```

The script builds from the canonical `cpp_runtime` source tree, validates the
native runtime, CLI subcommands, Rust desktop app, Web UI, architectures, and
code signatures, then copies the DMG to `packaging/dist/`.

Set `TYLOINFER_RELEASE_NATIVE_BUILD_DIR` to select a native build cache (relative
paths use the repository root), and `CARGO_TARGET_DIR` to select a Cargo cache
(relative paths use the Studio directory). A stable release environment can be
reused with `TYLOINFER_RELEASE_VENV` and `TYLOINFER_RELEASE_REUSE_VENV=1`.
Build path mappings are applied before Python extensions are compiled.

Only one macOS release can run in a checkout at a time. If a process is forcibly
terminated, first verify that no release is running, then remove the empty
`packaging/build/.mac-release.lock` directory before retrying. Separate checkouts
have independent locks.

## Windows

The Windows release packages `mfq serve`, TyloInfer, and the native CUDA
`mfq-runtime` sidecar into a single x64 NSIS installer. Service and model management are included. The package
excludes PyTorch and the training, calibration, quantization, and MiniCPM-o
dependency groups. The CUDA, cuBLAS, MSVC, and optional OpenMP runtime DLLs are
bundled with the sidecar, so end users do not need Python, the CUDA Toolkit, or a
native build environment.

Build from PowerShell 7 or newer on 64-bit Windows:

```powershell
cd packaging
.\build_release_windows.ps1
```

The build host needs Visual Studio Build Tools with the C++ x64 workload, a CUDA
Toolkit compatible with the selected compiler, `uv`, CMake, Ninja, Rust,
Node.js/npm, and NSIS. Tauri can download its pinned NSIS tools when they are not
already available. The script does not pin a Visual Studio release year.

Use an already configured x64 Developer PowerShell, or select a Visual Studio
installation with `-VisualStudioPath` / `TYLOINFER_RELEASE_VS_PATH`. Without an
explicit or active environment, the script discovers an installed C++ toolchain.
An incompatible compiler/Toolkit combination fails configuration with guidance
to select a compatible environment; the script does not bypass compiler checks.

The selected CUDA SDK follows `CUDACXX`, `nvcc` on `PATH`, `CUDA_HOME`,
`CUDA_PATH`, then installed Toolkit versions. `CUDACXX` names the Toolkit's
`bin/nvcc.exe`; compiler options belong in `CUDAFLAGS`. The native build and
bundled runtime DLLs use that same SDK. Invalid higher-priority settings are
reported rather than silently replaced.

Existing tools on `PATH`, `VSLANG`, and npm configuration (including `.npmrc` and
`npm_config_cache`) are respected. Configure a writable npm cache through npm's
normal configuration if the selected cache is unavailable. The script restores
its process environment on success or failure. CMake and Ninja manage compiler
dependency discovery; the release script does not rewrite their generated
compiler metadata. If localized compiler output is decoded incorrectly, check
the CMake/Ninja versions and terminal encoding in the build environment.

The default release CUDA architectures are `80-real;86-real;89`. Override release
inputs when needed:

```powershell
.\build_release_windows.ps1 -CudaArchitectures 89 -Jobs 2
```

`-NativeBuildDirectory` / `TYLOINFER_RELEASE_NATIVE_BUILD_DIR` selects the native
cache relative to the repository root. `CARGO_TARGET_DIR` selects the Cargo cache
relative to the Studio directory. Normal builds reuse these caches while using
fresh Python and release staging directories.
When changing the compiler or CUDA Toolkit, select an unused native cache (or
use `-Clean`) instead of reusing a CMake cache configured for another toolchain.

`-Clean` requests fresh native and Cargo caches without deleting previous
builds. When cache paths are explicitly selected, they must be empty for a clean
build. `-PortableOnly` reuses existing complete build inputs and accepts
`-ReleaseVenv` to select the release Python environment containing PyInstaller;
it cannot be combined with `-Clean`. Only one release or portable packaging run
can use a checkout at a time.

Concurrent builds in separate checkouts must also use separate native caches,
Cargo caches, and output directories.

The installer supports 64-bit Windows with a compatible NVIDIA GPU and driver.
It bundles CUDA runtime DLLs from the build machine, so users do not need the
CUDA Toolkit.

## Runtime behavior

TyloInfer starts `mfq serve` locally without a model. The catalog and native
file picker can load, unload, or replace `.mfq` models. Studio records external
paths without copying files; selecting one shard registers the full sibling
shard family.
