"""Assemble the Python distribution and build its wheel."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory


def build_wheel(output_dir: Path) -> Path:
    repository = Path(__file__).resolve().parents[2]
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

        kernels = repository / "csrc/kernels/cuda"
        resources = project / "tyloinfer/kernels/cuda"
        for source in (kernels / "tq_ops.cu", kernels / "tq_ops.h",
                       *sorted(kernels.glob("tq_*.cuh"))):
            shutil.copy2(source, resources / source.name)

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "dist",
    )
    args = parser.parse_args()
    print(build_wheel(args.output_dir))


if __name__ == "__main__":
    main()
