from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
import types
import zipfile

import pytest

BUILD_SCRIPT = Path(__file__).parents[1] / 'packaging/build_release_windows.ps1'
POWERSHELL = shutil.which('pwsh')
if not POWERSHELL:
    pytest.skip('PowerShell 7 is required for the Windows packaging implementation', allow_module_level=True)


def powershell(command, environment=None):
    """Call the real Windows packaging functions without starting a release build."""
    env = dict(os.environ, TYLOINFER_TEST_BUILD_SCRIPT=str(BUILD_SCRIPT))
    env.update(environment or {})
    return subprocess.run(
        [POWERSHELL, '-NoProfile', '-NonInteractive', '-Command',
         '. $env:TYLOINFER_TEST_BUILD_SCRIPT; ' + command],
        env=env, capture_output=True, text=True, encoding='utf-8', timeout=45,
    )


source = powershell('[Console]::Out.Write((Get-PortableBuilderSource))')
assert source.returncode == 0, source.stdout + source.stderr
module = types.ModuleType('windows_portable')
exec(compile(source.stdout, str(BUILD_SCRIPT), 'exec'), module.__dict__)


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
    (tauri / "tauri.conf.json").write_text(json.dumps({"version": "0.4.0"}))
    (tauri / "tauri.release-windows.conf.json").write_text(json.dumps({"bundle": {
        "resources": {
            "../dist": "web-ui", "../../packaging/Resources/tyloinfer-cli": "tyloinfer-cli",
            "../../packaging/windows-runtime": "", "../../packaging/Resources/licenses": "licenses",
        },
        "externalBin": ["../../packaging/sidecars/mfq-runtime", "../../packaging/sidecars/mfq-diagnostics"],
    }}))
    for path in (
        "build/tyloinfer.exe", "TyloInferStudio/dist/index.html",
        "packaging/Resources/tyloinfer-cli/tyloinfer-cli.exe", "packaging/Resources/tyloinfer-cli/_internal/python.dll",
        "packaging/windows-runtime/cudart64_12.dll", "packaging/Resources/licenses/NOTICE",
        "packaging/sidecars/mfq-runtime-x86_64-pc-windows-msvc.exe",
        "packaging/sidecars/mfq-diagnostics-x86_64-pc-windows-msvc.exe",
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
            "TyloInfer/tyloinfer.exe", "TyloInfer/mfq-runtime.exe", "TyloInfer/mfq-diagnostics.exe",
            "TyloInfer/tyloinfer-cli/tyloinfer-cli.exe", "TyloInfer/tyloinfer-cli/_internal/python.dll",
            "TyloInfer/cudart64_12.dll", "TyloInfer/web-ui/index.html", "TyloInfer/licenses/NOTICE",
        }
        assert package.read("TyloInfer/mfq-runtime.exe") == b"mfq-runtime-x86_64-pc-windows-msvc.exe"
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert archive.with_suffix(".zip.sha256").read_text() == f"{digest}  {archive.name}\n"


def test_portable_hashes_and_payload_ids_work_without_python_311_hash_api(tmp_path):
    """Build fixture archives with the current interpreter or an explicit Python 3.10."""
    root = project(tmp_path)
    source_file = tmp_path / 'embedded-packer.py'
    source_file.write_text(source.stdout, encoding='utf-8')
    python = os.environ.get('TYLOINFER_TEST_PYTHON310', sys.executable)
    verification = r'''
import hashlib
import pathlib
import sys
import types
import zipfile

source, root, output = map(pathlib.Path, sys.argv[1:])
packer = types.ModuleType('windows_portable')
exec(compile(source.read_text(encoding='utf-8'), str(source), 'exec'), packer.__dict__)
if hasattr(hashlib, 'file_digest'):
    del hashlib.file_digest
executable = root / 'build/tyloinfer.exe'
archive = packer.build_archive(root, executable, output / 'resources')
assert archive.with_suffix('.zip.sha256').read_text().split()[0] == hashlib.sha256(archive.read_bytes()).hexdigest()
large_file = output / 'multiple-chunks.bin'
large_file.write_bytes(b'chunked-content' * 160000)
assert packer.file_sha256(large_file) == hashlib.sha256(large_file.read_bytes()).digest()
digest = hashlib.sha256()
for name, path in sorted(packer.payload(root, executable).items()):
    digest.update(name.encode() + b'\0')
    digest.update(hashlib.sha256(path.read_bytes()).digest())
expected_payload_id = '0.4.0-' + digest.hexdigest()[:16]
def compile_fixture(arguments, check):
    script = pathlib.Path(arguments[-1]).read_text(encoding='utf-8')
    assert expected_payload_id in script
    target = next(line for line in script.splitlines() if line.startswith('OutFile '))
    pathlib.Path(target.removeprefix('OutFile "').removesuffix('"')).write_bytes(b'fixture launcher')
packer.subprocess.run = compile_fixture
standalone = packer.build_executable(root, executable, output / 'standalone', pathlib.Path('fixture-makensis'))
assert standalone.with_suffix('.exe.sha256').read_text().split()[0] == hashlib.sha256(standalone.read_bytes()).hexdigest()
archive = standalone.parent / 'TyloInfer_0.4.0_x64-portable.zip'
assert archive.with_suffix('.zip.sha256').read_text().split()[0] == hashlib.sha256(archive.read_bytes()).hexdigest()
with zipfile.ZipFile(archive) as package:
    assert package.namelist() == ['TyloInfer.exe']
    assert package.read('TyloInfer.exe') == standalone.read_bytes()
print('PASS portable archive checksums and content-derived payload ID on Python', sys.version.split()[0])
'''
    result = subprocess.run(
        [python, '-c', verification, str(source_file), str(root), str(tmp_path / 'output')],
        capture_output=True, text=True, encoding='utf-8', timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PASS portable archive checksums and content-derived payload ID' in result.stdout


def test_powershell_entry_builds_the_validated_portable_archive(tmp_path):
    root = project(tmp_path)
    output = tmp_path / 'output with spaces'
    result = powershell(
        'Invoke-PortablePackage -Python $env:TEST_PYTHON -ProjectDirectory $env:TEST_PROJECT '
        '-Executable $env:TEST_EXECUTABLE -OutputDirectory $env:TEST_OUTPUT',
        {'TEST_PYTHON': sys.executable, 'TEST_PROJECT': str(root),
         'TEST_EXECUTABLE': str(root / 'build/tyloinfer.exe'), 'TEST_OUTPUT': str(output)},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    archive, = output.glob('*.zip')
    with zipfile.ZipFile(archive) as package:
        assert package.testzip() is None
        assert 'TyloInfer/mfq-runtime.exe' in package.namelist()
    assert archive.with_suffix('.zip.sha256').is_file()


def test_powershell_entry_preserves_previous_output_for_invalid_payload(tmp_path):
    root = project(tmp_path)
    (root / 'TyloInferStudio/dist/index.html').unlink()
    output = tmp_path / 'output'
    output.mkdir()
    previous = output / 'TyloInfer_0.4.0_x64-portable.zip'
    previous.write_bytes(b'previous complete archive')
    result = powershell(
        'Invoke-PortablePackage -Python $env:TEST_PYTHON -ProjectDirectory $env:TEST_PROJECT '
        '-Executable $env:TEST_EXECUTABLE -OutputDirectory $env:TEST_OUTPUT',
        {'TEST_PYTHON': sys.executable, 'TEST_PROJECT': str(root),
         'TEST_EXECUTABLE': str(root / 'build/tyloinfer.exe'), 'TEST_OUTPUT': str(output)},
    )
    assert result.returncode != 0
    assert previous.read_bytes() == b'previous complete archive'


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


def test_launcher_zip_failure_preserves_previous_executable_and_archive(tmp_path, monkeypatch):
    root = project(tmp_path)
    output = tmp_path / 'output'
    output.mkdir()
    executable = output / 'TyloInfer_0.4.0_x64.exe'
    archive = output / 'TyloInfer_0.4.0_x64-portable.zip'
    executable.write_bytes(b'previous executable')
    archive.write_bytes(b'previous archive')

    def compile_fixture(arguments, check):
        script = Path(arguments[-1]).read_text()
        target = next(line for line in script.splitlines() if line.startswith('OutFile '))
        Path(target.removeprefix('OutFile "').removesuffix('"')).write_bytes(b'new executable')

    class FailingArchive:
        def __init__(self, *args, **kwargs):
            raise OSError('simulated archive failure')

    monkeypatch.setattr(module.subprocess, 'run', compile_fixture)
    monkeypatch.setattr(module.zipfile, 'ZipFile', FailingArchive)
    with pytest.raises(OSError, match='simulated archive failure'):
        module.build_executable(root, root / 'build/tyloinfer.exe', output, Path('fixture-makensis'))
    assert executable.read_bytes() == b'previous executable'
    assert archive.read_bytes() == b'previous archive'
    assert not list(output.glob('*.partial-*'))


@pytest.mark.skipif(sys.platform != "win32" or not os.environ.get("TYLOINFER_TEST_NSIS"), reason="Windows NSIS execution")
@pytest.mark.parametrize('cache_directory', [r'C:\cache with spaces', 'C:\\缓存 with spaces'])
def test_cache_option_reader_stops_at_the_argument_terminator(tmp_path, cache_directory):
    """Read a final argument whose terminator borders an inaccessible memory page."""
    argument = '/CACHE_DIR=' + cache_directory
    start = module.NSIS_TEMPLATE.index('Function ReadCacheDirectoryOption\n')
    end = module.NSIS_TEMPLATE.index('FunctionEnd', start) + len('FunctionEnd')
    reader = module.NSIS_TEMPLATE[start:end]
    allocate = "System::Call 'shell32::CommandLineToArgvW(p r0, *i .r1) p .r2'"
    release = "System::Call 'kernel32::LocalFree(p r2)'"
    assert reader.count(allocate) == reader.count(release) == 1
    reader = reader.replace(allocate, 'Call BuildGuardedArgumentVector').replace(
        release, "System::Call 'kernel32::VirtualFree(p r2, p 0, i 0x8000)'",
    )
    offset = 4096 - len((argument + '\0').encode('utf-16le'))
    executable = tmp_path / 'guarded-argument.exe'
    script = tmp_path / 'guarded-argument.nsi'
    script.write_text(f'''Unicode true
RequestExecutionLevel user
SilentInstall silent
OutFile "{module.nsis_string(str(executable))}"
Var CacheRoot
Function .onInit
  System::Call 'kernel32::SetErrorMode(i 0x8003)'
FunctionEnd
Function BuildGuardedArgumentVector
  System::Call 'kernel32::VirtualAlloc(p 0, p 8192, i 0x3000, i 0x04) p .r2'
  StrCmp $2 0 fixture_failed
  IntOp $4 $2 + 4096
  System::Call 'kernel32::VirtualProtect(p r4, p 4096, i 0x01, *i .r9) i .r8'
  StrCmp $8 0 fixture_failed
  IntOp $5 $2 + {offset}
  System::Call 'kernel32::lstrcpyW(p r5, w "{module.nsis_string(argument)}")'
  System::Call '*$2(p 0, p r5)'
  StrCpy $1 2
  Return
fixture_failed:
  SetErrorLevel 99
  Quit
FunctionEnd
{reader}
Section
  Call ReadCacheDirectoryOption
  StrCmp $CacheRoot "{module.nsis_string(cache_directory)}" 0 failed
  SetErrorLevel 0
  Quit
failed:
  SetErrorLevel 1
SectionEnd
''', encoding='utf-8')
    subprocess.run(
        [os.environ['TYLOINFER_TEST_NSIS'], '-INPUTCHARSET', 'UTF8', '-V2', str(script)],
        check=True, capture_output=True, timeout=30,
    )
    result = subprocess.run([str(executable), '/S'], timeout=30)
    assert result.returncode == 0, f'Argument reader exited with 0x{result.returncode & 0xffffffff:08x}'


@pytest.mark.skipif(sys.platform != "win32" or not os.environ.get("TYLOINFER_TEST_NSIS"), reason="Windows NSIS execution")
@pytest.mark.parametrize('release_lock', [True, False])
def test_publication_retries_a_temporary_lock_and_preserves_a_permanent_failure(tmp_path, monkeypatch, release_lock):
    """Hold a real extracted file open until a failed publication attempt is observed."""
    import ctypes
    from ctypes import wintypes

    gate = r'''
  FileOpen $0 "$CacheRoot\test-publish-ready" w
  FileClose $0
test_publication_wait:
  IfFileExists "$CacheRoot\test-publish-continue" test_publication_continue
  Sleep 10
  Goto test_publication_wait
test_publication_continue:
'''
    failure_signal = r'''
  FileOpen $1 "$CacheRoot\test-publish-failed" w
  FileClose $1
'''
    template = module.NSIS_TEMPLATE
    assert template.count('publish:\n') == template.count('  IntOp $0 $0 - 1\n') == 1
    template = template.replace('publish:\n', 'publish:\n' + gate).replace(
        '  IntOp $0 $0 - 1\n', failure_signal + '  IntOp $0 $0 - 1\n',
    )
    monkeypatch.setattr(module, 'NSIS_TEMPLATE', template)
    root = project(tmp_path)
    executable = module.build_executable(
        root, root / 'build/tyloinfer.exe', tmp_path / 'output', Path(os.environ['TYLOINFER_TEST_NSIS']),
    )
    cache = tmp_path / 'runtime cache'
    environment = dict(os.environ, TYLOINFER_STUDIO_RUNTIME_CACHE=str(cache))
    launcher = subprocess.Popen([str(executable), '/S', '/EXTRACT_ONLY'], env=environment)
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.CreateFileW.argtypes = (
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    )
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = None

    def wait_for(path):
        """Wait for a fixture handshake while rejecting an early launcher exit."""
        deadline = time.monotonic() + 15
        while not path.exists():
            assert launcher.poll() is None, 'Launcher exited before the publication handshake'
            assert time.monotonic() < deadline, f'Launcher did not reach {path.name}'
            time.sleep(.01)

    try:
        wait_for(cache / 'test-publish-ready')
        stage, = cache.glob('0.4.0-*.partial-*')
        handle = kernel32.CreateFileW(str(stage / 'mfq-runtime.exe'), 0x80000000, 3, None, 3, 0x80, None)
        assert handle != wintypes.HANDLE(-1).value, ctypes.WinError(ctypes.get_last_error())
        (cache / 'test-publish-continue').touch()
        wait_for(cache / 'test-publish-failed')
        if release_lock:
            assert kernel32.CloseHandle(handle)
            handle = None
        assert launcher.wait(timeout=10) == (0 if release_lock else 1)
        destination = stage.with_name(stage.name.split('.partial-', 1)[0])
        if release_lock:
            assert (destination / '.ready').is_file()
            assert not stage.exists()
        else:
            assert (stage / '.ready').is_file()
            assert not destination.exists()
    finally:
        if handle not in (None, wintypes.HANDLE(-1).value):
            kernel32.CloseHandle(handle)
        if launcher.poll() is None:
            launcher.terminate()
            launcher.wait(timeout=5)


@pytest.mark.skipif(sys.platform != "win32" or not os.environ.get("TYLOINFER_TEST_NSIS"), reason="Windows NSIS execution")
def test_single_executable_extracts_reuses_and_repairs_its_cache(tmp_path):
    root = project(tmp_path)
    executable = module.build_executable(root, root / "build/tyloinfer.exe", tmp_path / "output", Path(os.environ["TYLOINFER_TEST_NSIS"]))
    with zipfile.ZipFile(executable.parent / "TyloInfer_0.4.0_x64-portable.zip") as package:
        assert package.namelist() == ["TyloInfer.exe"]
        assert package.read("TyloInfer.exe") == executable.read_bytes()
    cache = tmp_path / "runtime cache"
    environment = dict(os.environ, TYLOINFER_STUDIO_RUNTIME_CACHE=str(cache))
    arguments = [str(executable), '/S', "/EXTRACT_ONLY"]
    subprocess.run(arguments, check=True, timeout=30, env=environment)
    directory, = cache.glob("0.4.0-*")
    native = directory / "mfq-runtime.exe"
    assert native.read_bytes() == b"mfq-runtime-x86_64-pc-windows-msvc.exe"
    assert (directory / "tyloinfer-cli/_internal/python.dll").is_file()
    sentinel = directory / "cache-hit-check"
    sentinel.touch()
    timestamp = native.stat().st_mtime_ns
    subprocess.run(arguments, check=True, timeout=30, env=environment)
    assert sentinel.exists() and native.stat().st_mtime_ns == timestamp
    (directory / "tyloinfer-cli/tyloinfer-cli.exe").unlink()
    subprocess.run(arguments, check=True, timeout=30, env=environment)
    assert (directory / "tyloinfer-cli/tyloinfer-cli.exe").is_file()
    assert not sentinel.exists()
    backups = list(cache.glob(directory.name + '.previous-*'))
    assert len(backups) == 1
    assert (backups[0] / 'cache-hit-check').exists()
    (directory / 'cudart64_12.dll').unlink()
    subprocess.run(arguments, check=True, timeout=30, env=environment)
    assert (directory / 'cudart64_12.dll').is_file()
    children = [subprocess.Popen(arguments, env=environment) for _ in range(3)]
    assert [child.wait(timeout=30) for child in children] == [0, 0, 0]
    override = tmp_path / 'explicit cache with spaces'
    subprocess.run(arguments + [f'/CACHE_DIR={override}'], check=True, timeout=30, env=environment)
    alternate, = override.glob('0.4.0-*')
    assert (alternate / 'mfq-runtime.exe').is_file()


@pytest.mark.skipif(sys.platform != "win32" or not os.environ.get("TYLOINFER_TEST_STANDALONE_APP"), reason="Windows launcher fixture")
def test_single_executable_launches_the_app_and_waits_for_an_update_parent(tmp_path):
    root = project(tmp_path)
    shutil.copyfile(os.environ["TYLOINFER_TEST_STANDALONE_APP"], root / "build/tyloinfer.exe")
    executable = module.build_executable(root, root / "build/tyloinfer.exe", tmp_path / "output", Path(os.environ["TYLOINFER_TEST_NSIS"]))
    cache, log = tmp_path / "runtime cache", tmp_path / "launched.txt"
    environment = dict(os.environ, TYLOINFER_STUDIO_RUNTIME_CACHE=str(cache), TYLOINFER_TEST_LAUNCH_LOG=str(log))
    parent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1)"])
    try:
        launcher = subprocess.Popen([str(executable), '/S', f"/WAITPID={parent.pid}"], env=environment)
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
        if 'launcher' in locals() and launcher.poll() is None:
            launcher.terminate()
            launcher.wait(timeout=5)
        if parent.poll() is None:
            parent.terminate()
            parent.wait(timeout=5)
