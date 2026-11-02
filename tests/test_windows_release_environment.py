"""Release preparation must preserve development environments and previous resources."""

from pathlib import Path
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import urllib.request

import pytest

pytestmark = pytest.mark.skipif(sys.platform != 'win32', reason='Windows release preparation')
BUILD_SCRIPT = Path(__file__).resolve().parents[1] / 'packaging/build_release_windows.ps1'


def run_helpers(tmp_path, body, *arguments, timeout=30):
    """Load build helpers without executing the release entry point."""
    script = tmp_path / 'check.ps1'
    script.write_text(r'''
param([string]$BuildScript, [string]$FixtureRoot, [string]$Python)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$syntaxErrors = $null
$syntax = [System.Management.Automation.Language.Parser]::ParseFile($BuildScript, [ref]$null, [ref]$syntaxErrors)
if ($syntaxErrors) { throw ($syntaxErrors | Out-String) }
foreach ($definition in $syntax.FindAll({param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst]}, $false)) {
    Invoke-Expression $definition.Extent.Text
}
''' + body, encoding='utf-8')
    result = subprocess.run([shutil.which('pwsh') or 'powershell.exe', '-NoProfile', '-NonInteractive',
                             '-File', str(script), str(BUILD_SCRIPT), str(tmp_path), *arguments],
                            capture_output=True, text=True, timeout=timeout)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_build_python_is_reused_without_changing_the_environment(tmp_path):
    output = run_helpers(tmp_path, r'''
$context = @{ Python=$null }
$before = $env:VIRTUAL_ENV
Initialize-BuildPython $context $Python
if ($context.Python -ne $Python) { throw 'Wrong build interpreter' }
if ($env:VIRTUAL_ENV -cne $before) { throw 'Development environment changed' }
Write-Output 'PASS build-only Python'
''', sys.executable)
    assert 'PASS build-only Python' in output


def test_resource_publication_preserves_old_resources_and_isolates_build_caches(tmp_path):
    output = run_helpers(tmp_path, r'''
$context = New-BuildContext -Output (Join-Path $FixtureRoot 'output') -ScriptDirectory $FixtureRoot
$other = New-BuildContext -Output $context.OutputDirectory -ScriptDirectory $FixtureRoot
if ($context.NativeBuildDirectory -eq $other.NativeBuildDirectory) { throw 'Build caches collide' }
if ($context.TemporaryDirectory -eq $other.TemporaryDirectory) { throw 'Temporary directories collide' }
Initialize-ReleaseDirectories $context
if ($env:TEMP -ne $context.TemporaryDirectory -or $env:TMP -ne $context.TemporaryDirectory) { throw 'Temporary tools escaped the build directory' }
if (-not (Test-Path -LiteralPath $context.TemporaryDirectory -PathType Container)) { throw 'Missing temporary directory' }
New-Item -ItemType Directory -Path $context.ServerDirectory | Out-Null
[IO.File]::WriteAllText((Join-Path $context.ServerDirectory 'previous.exe'), 'previous')
[IO.File]::WriteAllText((Join-Path $context.ServerStageDirectory 'tyloinfer-server.exe'), 'native')
Publish-NativeServer $context
if ([IO.File]::ReadAllText((Join-Path $context.ServerDirectory 'tyloinfer-server.exe')) -cne 'native') { throw 'Missing new server' }
$previous = @(Get-ChildItem -LiteralPath $context.ResourceDirectory -Directory -Filter 'tyloinfer-server.previous-*')
if ($previous.Count -ne 1) { throw 'Previous resources were not preserved' }
if ([IO.File]::ReadAllText((Join-Path $previous[0].FullName 'previous.exe')) -cne 'previous') { throw 'Previous resources changed' }
$context.ServerDirectory = Join-Path $FixtureRoot 'outside'
try { Publish-NativeServer $context; throw 'Unsafe path accepted' }
catch { if ($_.Exception.Message -notmatch 'outside the release staging paths') { throw } }
Write-Output 'PASS preserved resources'
''')
    assert 'PASS preserved resources' in output


def test_clean_rejects_without_deleting_existing_files(tmp_path):
    sentinel = tmp_path / 'keep.txt'
    sentinel.write_text('keep')
    result = subprocess.run([shutil.which('pwsh') or 'powershell.exe', '-NoProfile', '-NonInteractive',
                             '-File', str(BUILD_SCRIPT), '-Clean', '-OutputDirectory', str(tmp_path)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert '-Clean no longer deletes release trees' in result.stdout + result.stderr
    assert sentinel.read_text() == 'keep'


def test_explicit_npm_cache_is_preserved(tmp_path):
    output = run_helpers(tmp_path, r'''
$context = New-BuildContext -Output (Join-Path $FixtureRoot 'output') -ScriptDirectory $FixtureRoot
$env:npm_config_cache = Join-Path $FixtureRoot 'existing cache'
Initialize-ReleaseDirectories $context
if ($env:npm_config_cache -ne (Join-Path $FixtureRoot 'existing cache')) { throw 'Explicit npm cache changed' }
Write-Output 'PASS explicit npm cache'
''')
    assert 'PASS explicit npm cache' in output


@pytest.mark.skipif(not os.environ.get('TYLOINFER_TEST_NATIVE_SERVER'), reason='Real native service opt-in')
def test_real_native_service_dependency_closure_starts_with_system_only_path(tmp_path):
    run_helpers(tmp_path, r'''
Import-VisualStudioEnvironment
$stage = Join-Path $FixtureRoot 'native-runtime'
New-Item -ItemType Directory -Path $stage | Out-Null
Copy-Item -LiteralPath $env:TYLOINFER_TEST_NATIVE_SERVER -Destination (Join-Path $stage 'tyloinfer-server.exe')
Copy-VisualCppRuntimeDlls $stage
$collector = Join-Path (Split-Path -Parent $BuildScript) 'collect_windows_runtime.cmake'
$cmake = Require-Command 'cmake'
$dumpbin = Require-Command 'dumpbin'
& $cmake "-DSERVER=$stage/tyloinfer-server.exe" "-DDESTINATION=$stage" "-DDUMPBIN=$dumpbin" '-P' $collector
if ($LASTEXITCODE -eq 0) { throw 'Missing dependencies were accepted' }
Invoke-Checked $cmake @(
    "-DSERVER=$stage/tyloinfer-server.exe", "-DDESTINATION=$stage", "-DDUMPBIN=$dumpbin",
    "-DSEARCH_DIRECTORIES=$env:TYLOINFER_TEST_NATIVE_DLL_DIRS", '-P', $collector
)
''', timeout=120)
    stage = tmp_path / 'native-runtime'
    assert not list(stage.glob('python*'))
    environment = dict(os.environ, PATH=f'{stage};{os.environ["SystemRoot"]}/System32;{os.environ["SystemRoot"]}',
                       TYLOINFER_SERVER_API_KEY='packaging-test-root-key')
    process = subprocess.Popen([str(stage / 'tyloinfer-server.exe'), '--host', '127.0.0.1', '--port', '0',
                                '--data-dir', str(tmp_path / '数据目录'), '--exit-on-stdin-close'],
                               env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, encoding='utf-8')
    lines = queue.Queue()
    threading.Thread(target=lambda: lines.put(process.stdout.readline()), daemon=True).start()
    try:
        listening = json.loads(lines.get(timeout=15))
        assert listening['event'] == 'listening'
        base = f'http://127.0.0.1:{listening["port"]}'
        for endpoint in ('/health', '/health/identity', '/api/v1/models'):
            request = urllib.request.Request(base + endpoint, headers={'Authorization': 'Bearer packaging-test-root-key'})
            with urllib.request.urlopen(request, timeout=5) as response:
                assert response.status == 200
                json.load(response)
        process.stdin.close()
        assert process.wait(timeout=15) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        process.stdout.close()
        process.stderr.close()
