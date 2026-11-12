"""Assemble the Python distribution and build its wheel."""

import argparse
import re
import runpy
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def cuda_files(repository: Path) -> tuple[Path, ...]:
    """Validate and return the quoted-include closure of the CUDA JIT manifest."""
    repository = repository.resolve()
    package = repository / 'tyloinfer/kernels/cuda'
    manifest = runpy.run_path(str(package / '_sources.py'))
    roots = manifest['include_roots'](repository)
    pending = list(manifest['sources'](repository, package))
    visited: set[Path] = set()
    kernel_root = repository / 'csrc/engine/backends/cuda/kernels'
    native_root = repository / 'csrc/engine/backends/cuda/native'
    encoding_root = repository / 'csrc/engine/weights/encoding'
    ggml_root = repository / 'csrc/components/ggml'
    include = re.compile(r'^\s*#\s*include\s*"([^"\n]+)"', re.MULTILINE)
    while pending:
        source = pending.pop().resolve()
        if source in visited:
            continue
        if not source.is_file():
            raise RuntimeError(f'CUDA JIT source is missing: {source}')
        local = source.is_relative_to(package)
        allowed = (
            local
            or (source.is_relative_to(kernel_root) and source.suffix in {'.h', '.cuh', '.inc'})
            or source in {kernel_root / name for name in manifest['KERNELS']}
            or source in {native_root / 'env.h', native_root / 'env.cpp'}
            or (source.is_relative_to(encoding_root) and source.suffix == '.h')
            or (source.is_relative_to(ggml_root) and source.suffix in {'.h', '.cuh', '.inc'})
        )
        if not allowed:
            raise RuntimeError(f'CUDA JIT depends on non-portable source: {source}')
        visited.add(source)
        for name in include.findall(source.read_text(encoding='utf-8')):
            # These mutually exclusive vendor headers are outside the CUDA JIT.
            if source == ggml_root / 'src/ggml-cuda/common.cuh' and name in {'vendors/hip.h', 'vendors/musa.h'}:
                continue
            candidates = (source.parent / name, *(root / name for root in roots))
            target = next((path.resolve() for path in candidates if path.is_file()), None)
            if target is None:
                raise RuntimeError(f'CUDA JIT include {name!r} from {source} cannot be resolved')
            pending.append(target)
    return tuple(sorted(visited))


def collect_cuda_sources(repository: Path, project: Path, files: tuple[Path, ...]) -> None:
    """Stage validated shared sources; package-local adapters are copied separately."""
    repository = repository.resolve()
    package = repository / 'tyloinfer/kernels/cuda'
    for source in files:
        if source.is_relative_to(package):
            continue
        destination = project / 'tyloinfer/kernels/cuda/_src' / source.relative_to(repository)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def build_wheel(output_dir: Path) -> Path:
    """Build a wheel after staging the shared CUDA source closure."""
    repository = Path(__file__).resolve().parents[2]
    files = cuda_files(repository)
    with TemporaryDirectory(prefix="tyloinfer-wheel-") as temporary:
        project = Path(temporary) / "project"
        project.mkdir()
        for name in ("pyproject.toml", "README.md", "LICENSE", "NOTICE"):
            shutil.copy2(repository / name, project / name)
        shutil.copytree(repository / "LICENSES", project / "LICENSES")
        shutil.copytree(
            repository / "tyloinfer", project / "tyloinfer",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )

        kernels = repository / "csrc/engine/backends/cuda/kernels"
        resources = project / "tyloinfer/kernels/cuda"
        for source in (kernels / "tq_ops.cu", kernels / "tq_ops.h",
                       *sorted(kernels.glob("tq_*.cuh"))):
            shutil.copy2(source, resources / source.name)

        collect_cuda_sources(repository, project, files)

        artifacts = Path(temporary) / "dist"
        subprocess.run(
            ["uv", "build", "--wheel", str(project), "--out-dir", str(artifacts)],
            check=True, stdout=sys.stderr,
        )
        wheel, = artifacts.glob("*.whl")
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / wheel.name
        shutil.copy2(wheel, destination)
        return destination


def main() -> None:
    """Build the distribution in the requested output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "dist",
    )
    args = parser.parse_args()
    print(build_wheel(args.output_dir))


if __name__ == "__main__":
    main()
