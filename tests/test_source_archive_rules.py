from pathlib import Path
import shutil
import subprocess

import pytest


def test_model_weights_are_ignored_without_excluding_architecture_sources(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git is required to check source archive rules")
    root = Path(__file__).resolve().parents[1]
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text((root / ".gitignore").read_text())
    for path, ignored in (
        ("models/local/checkpoint.tq", True),
        ("build/checkpoint.tq", True),
        ("cpp_runtime/models/model.cpp", False),
        ("cpp_runtime/backends/metal/models/deepseek_v41/model.cpp", False),
        ("cpp_runtime/backends/cuda/models/qwen35/model.cpp", False),
        ("cpp_runtime/backends/hip/models/qwen35/model.cpp", False),
    ):
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", path],
            cwd=tmp_path, capture_output=True, text=True,
        )
        assert result.returncode in (0, 1), result.stderr
        assert (result.returncode == 0) is ignored, path
