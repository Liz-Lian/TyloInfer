from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
import zipfile

import pytest

spec = importlib.util.spec_from_file_location(
    "windows_portable", Path(__file__).parents[1] / "packaging/build_portable_windows.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def project(tmp_path):
    root = tmp_path / "source tree"
    tauri = root / "TyloInferStudio/src-tauri"
    tauri.mkdir(parents=True)
    (tauri / "icons").mkdir()
    images = []
    entries = []
    offset = 6 + 16 * 3
    for size in (16, 32, 48):
        mask = b"\0" * (((size + 31) // 32) * 4 * size)
        pixels = b"\x20\x40\x60\xff" * size * size
        image = struct.pack("<IiiHHIIiiII", 40, size, size * 2, 1, 32, 0, len(pixels) + len(mask), 0, 0, 0, 0) + pixels + mask
        entries.append(struct.pack("<BBBBHHII", size, size, 0, 0, 1, 32, len(image), offset))
        images.append(image)
        offset += len(image)
    (tauri / "icons/icon.ico").write_bytes(struct.pack("<HHH", 0, 1, 3) + b"".join(entries + images))
    (root / "packaging").mkdir()
    shutil.copyfile(Path(__file__).parents[1] / "packaging/windows_standalone.nsi", root / "packaging/windows_standalone.nsi")
    (tauri / "tauri.conf.json").write_text(json.dumps({"version": "0.4.0"}))
    (tauri / "tauri.release-windows.conf.json").write_text(json.dumps({"bundle": {
        "resources": {
            "../dist": "web-ui", "../../packaging/Resources/tyloinfer-server": "tyloinfer-server",
            "../../packaging/Resources/licenses": "licenses",
        },
    }}))
    for path in (
        "build/tyloinfer.exe", "TyloInferStudio/dist/index.html",
        "packaging/Resources/tyloinfer-server/tyloinfer-server.exe",
        "packaging/Resources/tyloinfer-server/cudart64_12.dll",
        "packaging/Resources/tyloinfer-server/libcurl.dll", "packaging/Resources/licenses/NOTICE",
        "packaging/sidecars/old-sidecar.exe", "packaging/Resources/tyloinfer-cli/python.dll",
        "build/private-server.log", "packaging/.venv/private.txt",
    ):
        file = root / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(file.name.encode())
    return root


def test_zip_contains_the_installer_resources_without_build_files(tmp_path):
    root = project(tmp_path)
    archive = module.build_archive(root, root / "build/tyloinfer.exe", tmp_path / "output")
    with zipfile.ZipFile(archive) as package:
        assert package.testzip() is None
        assert set(package.namelist()) == {
            "TyloInfer/tyloinfer.exe", "TyloInfer/tyloinfer-server/tyloinfer-server.exe",
            "TyloInfer/tyloinfer-server/cudart64_12.dll", "TyloInfer/tyloinfer-server/libcurl.dll",
            "TyloInfer/web-ui/index.html", "TyloInfer/licenses/NOTICE",
        }
        assert package.read("TyloInfer/tyloinfer-server/tyloinfer-server.exe") == b"tyloinfer-server.exe"
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert archive.with_suffix(".zip.sha256").read_text() == f"{digest}  {archive.name}\n"


def test_incomplete_payload_does_not_replace_a_previous_archive(tmp_path):
    root = project(tmp_path)
    (root / "TyloInferStudio/dist/index.html").unlink()
    output = tmp_path / "output"
    output.mkdir()
    archive = output / "TyloInfer_0.4.0_x64-portable.zip"
    archive.write_bytes(b"previous package")
    with pytest.raises(FileNotFoundError, match="web-ui/index.html"):
        module.build_archive(root, root / "build/tyloinfer.exe", output)
    assert archive.read_bytes() == b"previous package"
    assert not archive.with_suffix(".zip.partial").exists()


@pytest.mark.parametrize("destination", ["../outside", "/outside", "C:/outside", "TYLOINFER.EXE"])
def test_unsafe_or_conflicting_resource_destination_is_rejected(tmp_path, destination):
    root = project(tmp_path)
    config = root / "TyloInferStudio/src-tauri/tauri.release-windows.conf.json"
    value = json.loads(config.read_text())
    value["bundle"]["resources"] = {"../../build/tyloinfer.exe": destination}
    config.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        module.payload(root, root / "build/tyloinfer.exe")


def test_missing_native_server_preserves_the_previous_release(tmp_path):
    root = project(tmp_path)
    (root / "packaging/Resources/tyloinfer-server/tyloinfer-server.exe").unlink()
    output = tmp_path / "output"
    output.mkdir()
    archive = output / "TyloInfer_0.4.0_x64-portable.zip"
    archive.write_bytes(b"previous package")
    with pytest.raises(FileNotFoundError, match="tyloinfer-server/tyloinfer-server.exe"):
        module.build_archive(root, root / "build/tyloinfer.exe", output)
    assert archive.read_bytes() == b"previous package"


@pytest.mark.parametrize("legacy", ["python312.dll", "old.py", "_internal/data.bin"])
def test_python_runtime_cannot_be_added_to_native_resources(tmp_path, legacy):
    root = project(tmp_path)
    file = root / "packaging/Resources/tyloinfer-server" / legacy
    file.parent.mkdir(parents=True, exist_ok=True)
    file.touch()
    with pytest.raises(ValueError, match="Forbidden legacy product runtime"):
        module.payload(root, root / "build/tyloinfer.exe")


def test_legacy_external_bin_is_rejected(tmp_path):
    root = project(tmp_path)
    config = root / "TyloInferStudio/src-tauri/tauri.release-windows.conf.json"
    value = json.loads(config.read_text())
    value["bundle"]["externalBin"] = ["old-cli"]
    config.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="externalBin"):
        module.payload(root, root / "build/tyloinfer.exe")


def test_resource_source_outside_the_project_is_rejected(tmp_path):
    root = project(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("private")
    config = root / "TyloInferStudio/src-tauri/tauri.release-windows.conf.json"
    value = json.loads(config.read_text())
    value["bundle"]["resources"][str(outside)] = "outside.txt"
    config.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Resource source escapes project"):
        module.payload(root, root / "build/tyloinfer.exe")


def test_compiler_failure_preserves_previous_outputs(tmp_path, monkeypatch):
    root = project(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    previous = [output / "TyloInfer_0.4.0_x64.exe", output / "TyloInfer_0.4.0_x64-portable.zip"]
    for file in previous:
        file.write_bytes(b"previous release")

    def fail_compile(*args, **kwargs):
        (output / "TyloInfer_0.4.0_x64.exe.partial").write_bytes(b"incomplete")
        raise subprocess.CalledProcessError(1, "makensis")

    monkeypatch.setattr(module.subprocess, "run", fail_compile)
    with pytest.raises(subprocess.CalledProcessError):
        module.build_executable(root, root / "build/tyloinfer.exe", output, Path("makensis"))
    assert all(file.read_bytes() == b"previous release" for file in previous)
    assert not list(output.glob("*.partial"))


@pytest.mark.skipif(sys.platform != "win32" or not os.environ.get("TYLOINFER_TEST_NSIS"), reason="Windows NSIS execution")
@pytest.mark.parametrize("cache_option", [False, True])
def test_single_executable_extracts_reuses_and_repairs_its_cache(tmp_path, cache_option):
    root = project(tmp_path)
    executable = module.build_executable(root, root / "build/tyloinfer.exe", tmp_path / "output", Path(os.environ["TYLOINFER_TEST_NSIS"]))
    for artifact in (executable, executable.parent / "TyloInfer_0.4.0_x64-portable.zip"):
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        assert artifact.with_suffix(artifact.suffix + ".sha256").read_text() == f"{digest}  {artifact.name}\n"
    with zipfile.ZipFile(executable.parent / "TyloInfer_0.4.0_x64-portable.zip") as package:
        assert package.namelist() == ["TyloInfer.exe"]
        assert package.read("TyloInfer.exe") == executable.read_bytes()
    cache = tmp_path / "runtime cache"
    environment_cache = tmp_path / "unused environment cache" if cache_option else cache
    environment = dict(os.environ, TYLOINFER_STUDIO_RUNTIME_CACHE=str(environment_cache))
    arguments = [str(executable), "/EXTRACT_ONLY"]
    if cache_option:
        arguments = subprocess.list2cmdline(arguments) + f' /CACHE_DIR="{cache}"'
    subprocess.run(arguments, check=True, timeout=30, env=environment)
    if cache_option:
        assert not environment_cache.exists()
    directory, = cache.glob("0.4.0-*")
    native = directory / "tyloinfer-server/tyloinfer-server.exe"
    assert native.read_bytes() == b"tyloinfer-server.exe"
    assert (directory / "tyloinfer-server/cudart64_12.dll").is_file()
    sentinel = directory / "cache-hit-check"
    sentinel.touch()
    timestamp = native.stat().st_mtime_ns
    subprocess.run(arguments, check=True, timeout=30, env=environment)
    assert sentinel.exists() and native.stat().st_mtime_ns == timestamp
    (directory / "tyloinfer-server/libcurl.dll").unlink()
    subprocess.run(arguments, check=True, timeout=30, env=environment)
    assert (directory / "tyloinfer-server/libcurl.dll").is_file()
    assert not sentinel.exists()
    preserved, = cache.glob("*/previous/cache-hit-check")
    assert preserved.is_file()
    children = [subprocess.Popen(arguments, env=environment) for _ in range(3)]
    assert [child.wait(timeout=30) for child in children] == [0, 0, 0]


@pytest.mark.skipif(sys.platform != "win32" or not os.environ.get("TYLOINFER_TEST_STANDALONE_APP"), reason="Windows launcher fixture")
def test_single_executable_launches_the_app_and_waits_for_an_update_parent(tmp_path):
    root = project(tmp_path)
    shutil.copyfile(os.environ["TYLOINFER_TEST_STANDALONE_APP"], root / "build/tyloinfer.exe")
    executable = module.build_executable(root, root / "build/tyloinfer.exe", tmp_path / "output", Path(os.environ["TYLOINFER_TEST_NSIS"]))
    cache, log = tmp_path / "runtime cache", tmp_path / "launched.txt"
    environment = dict(os.environ, TYLOINFER_STUDIO_RUNTIME_CACHE=str(cache), TYLOINFER_TEST_LAUNCH_LOG=str(log))
    parent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1)"])
    try:
        launcher = subprocess.Popen([str(executable), f"/WAITPID={parent.pid}"], env=environment)
        time.sleep(.1)
        assert not log.exists()
        parent.wait(timeout=10)
        assert launcher.wait(timeout=15) == 0
        deadline = time.monotonic() + 10
        while not log.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        original, actual, directory = log.read_text(encoding="utf-16le").splitlines()
        assert Path(original) == executable
        assert Path(actual).parent.parent == cache
        assert Path(directory) == Path(actual).parent
    finally:
        if parent.poll() is None:
            parent.terminate()
            parent.wait(timeout=5)
