"""Keep a running development service safe during release preparation."""

from pathlib import Path
import os
import shutil
import subprocess
import sys

import pytest


BUILD_SCRIPT = Path(__file__).resolve().parents[1] / 'packaging/build_release_windows.ps1'
POWERSHELL = shutil.which('pwsh')


def invoke_functions(tmp_path, source):
    """Run the actual importable release functions in an isolated PowerShell process."""
    if not POWERSHELL:
        pytest.skip('PowerShell 7 is required')
    script = tmp_path / 'verify.ps1'
    script.write_text("$ErrorActionPreference = 'Stop'\n. $env:TEST_BUILD_SCRIPT\n" + source,
                      encoding='utf-8')
    result = subprocess.run(
        [POWERSHELL, '-NoProfile', '-NonInteractive', '-File', str(script)],
        env=dict(os.environ, TEST_BUILD_SCRIPT=str(BUILD_SCRIPT), TEST_ROOT=str(tmp_path)),
        capture_output=True, text=True, encoding='utf-8', timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows executable sharing rules')
def test_release_environment_rejects_locked_python_before_cleanup(tmp_path):
    script = tmp_path / 'check.ps1'
    script.write_text(r'''
param([string]$BuildScript, [string]$Venv)
$ErrorActionPreference = 'Stop'
$syntax = [System.Management.Automation.Language.Parser]::ParseFile($BuildScript, [ref]$null, [ref]$null)
foreach ($definition in $syntax.FindAll({param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst]}, $false)) {
    if ($definition.Name -in @('Fail', 'Assert-ReleaseEnvironmentAvailable')) {
        Invoke-Expression $definition.Extent.Text
    }
}
$context = @{VenvDirectory=$Venv}
Assert-ReleaseEnvironmentAvailable $context
New-Item -ItemType Directory -Force -Path (Join-Path $Venv 'Scripts') | Out-Null
$python = Join-Path $Venv 'Scripts/python.exe'
$sentinel = Join-Path $Venv 'keep.txt'
[IO.File]::WriteAllText($python, 'fixture')
[IO.File]::WriteAllText($sentinel, 'keep')
$lock = [IO.File]::Open($python, [IO.FileMode]::Open, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
$rejected = $false
try {
    try { Assert-ReleaseEnvironmentAvailable $context }
    catch {
        if ($_.Exception.Message -notmatch 'choose an unused -ReleaseVenv') { throw }
        $rejected = $true
    }
} finally { $lock.Dispose() }
if (-not $rejected) { throw 'Locked interpreter was accepted' }
if ([IO.File]::ReadAllText($sentinel) -cne 'keep') { throw 'Environment was modified' }
Assert-ReleaseEnvironmentAvailable $context
Write-Output 'PASS release environment guard'
''', encoding='utf-8')
    build_script = Path(__file__).resolve().parents[1] / 'packaging/build_release_windows.ps1'
    if not POWERSHELL:
        pytest.skip('PowerShell 7 is required')
    result = subprocess.run([POWERSHELL, '-NoProfile', '-NonInteractive', '-File',
                             str(script), str(build_script), str(tmp_path / 'release env')],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PASS release environment guard' in result.stdout


def test_each_build_has_fresh_work_directories(tmp_path):
    """Separate release staging while retaining the native incremental build cache."""
    invoke_functions(tmp_path, r'''
$env:TYLOINFER_RELEASE_NATIVE_BUILD_DIR = $null
$packaging = Join-Path $env:TEST_ROOT 'packaging'
$output = Join-Path $env:TEST_ROOT 'dist'
$first = New-BuildContext '' $output $packaging
$second = New-BuildContext '' $output $packaging
if ($first.BuildDirectory -eq $second.BuildDirectory) { throw 'Reused build directory' }
if ($first.VenvDirectory -eq $second.VenvDirectory) { throw 'Reused default Python environment' }
if (-not $first.PyInstallerBuildDirectory.StartsWith($first.BuildDirectory)) { throw 'Shared PyInstaller stage' }
$nativeCache = Join-Path $packaging 'build/runtime'
if ($first.NativeBuildDirectory -ne $nativeCache -or $second.NativeBuildDirectory -ne $nativeCache) {
    throw 'Native incremental cache changed between release builds'
}
''')


def test_native_build_cache_override_uses_repository_root_and_honors_precedence(tmp_path):
    """Reuse an explicitly selected native cache without modifying its existing files."""
    invoke_functions(tmp_path, r'''
$packaging = Join-Path $env:TEST_ROOT 'packaging'
$output = Join-Path $env:TEST_ROOT 'dist'
$relative = 'native caches/cuda-runtime'
$expected = [IO.Path]::GetFullPath((Join-Path $env:TEST_ROOT $relative))
New-Item -ItemType Directory -Force -Path $expected | Out-Null
$sentinel = Join-Path $expected 'CMakeCache.txt'
[IO.File]::WriteAllText($sentinel, 'existing native cache')
$env:TYLOINFER_RELEASE_NATIVE_BUILD_DIR = $relative
$environment = New-BuildContext '' $output $packaging
if ($environment.NativeBuildDirectory -ne $expected) { throw 'Native cache environment override was not resolved from the repository root' }
$explicit = Join-Path $env:TEST_ROOT 'another native cache'
$argument = New-BuildContext '' $output $packaging -NativeBuildDirectory $explicit
if ($argument.NativeBuildDirectory -ne $explicit) { throw 'Explicit native cache did not override the environment' }
$relativeArgument = New-BuildContext '' $output $packaging -NativeBuildDirectory $relative
if ($relativeArgument.NativeBuildDirectory -ne $expected) { throw 'Relative native cache argument was not resolved from the repository root' }
if ([IO.File]::ReadAllText($sentinel) -cne 'existing native cache') { throw 'Native cache contents were changed' }
''')


def test_msvc_prefix_repair_preserves_good_prefixes_and_original_files(tmp_path):
    """Repair only reversible legacy decoding and keep the compiler and Ninja prefixes aligned."""
    output = invoke_functions(tmp_path, r'''
if ([Text.Encoding]::GetEncoding(0).CodePage -eq 65001) { Write-Output 'SKIP UTF8'; return }
$context = @{ NativeBuildDirectory = (Join-Path $env:TEST_ROOT 'native'); BuildId = 'test' }
$files = Join-Path $context.NativeBuildDirectory 'CMakeFiles'
$version = Join-Path $files '3.31.6'
$goodVersion = Join-Path $files '4.0.0'
New-Item -ItemType Directory -Force -Path $version, $goodVersion | Out-Null
$correct = 'é: à:  '
$bad = [Text.Encoding]::GetEncoding(0).GetString([Text.Encoding]::UTF8.GetBytes($correct))
$rules = Join-Path $files 'rules.ninja'
$c = Join-Path $version 'CMakeCCompiler.cmake'
$cxx = Join-Path $version 'CMakeCXXCompiler.cmake'
$good = Join-Path $goodVersion 'CMakeCXXCompiler.cmake'
$ascii = Join-Path $goodVersion 'CMakeCCompiler.cmake'
$originals = @{
    $rules = "msvc_deps_prefix = $bad`nother_rule = unchanged`n"
    $c = "set(CMAKE_C_CL_SHOWINCLUDES_PREFIX `"$bad`")`n"
    $cxx = "set(CMAKE_CXX_CL_SHOWINCLUDES_PREFIX `"$bad`")`n"
}
foreach ($file in $originals.Keys) { [IO.File]::WriteAllText($file, $originals[$file]) }
[IO.File]::WriteAllText($good, "set(CMAKE_CXX_CL_SHOWINCLUDES_PREFIX `"$correct`")`n")
[IO.File]::WriteAllText($ascii, 'set(CMAKE_C_CL_SHOWINCLUDES_PREFIX "Note: including file:  ")')
Repair-MsvcDependencyPrefixEncoding $context
foreach ($file in $originals.Keys) {
    if (-not [IO.File]::ReadAllText($file).Contains($correct)) { throw 'Prefix was not repaired' }
    if ([IO.File]::ReadAllText("$file.before-prefix-fix-test") -cne $originals[$file]) {
        throw 'Original generated file was not preserved'
    }
}
if ([IO.File]::ReadAllText($rules) -cne "msvc_deps_prefix = $correct`nother_rule = unchanged`n") {
    throw 'Unrelated Ninja rules changed'
}
Repair-MsvcDependencyPrefixEncoding $context
if (Test-Path -LiteralPath "$good.before-prefix-fix-test") { throw 'Correct Unicode prefix changed' }
if (Test-Path -LiteralPath "$ascii.before-prefix-fix-test") { throw 'ASCII prefix changed' }
''')
    if 'SKIP UTF8' in output:
        pytest.skip('The system ANSI encoding is already UTF-8')


def test_msvc_prefix_repair_rejects_an_already_built_bad_cache_before_writing(tmp_path):
    """Do not accept stale dependency records or invalidate CUDA outputs automatically."""
    output = invoke_functions(tmp_path, r'''
if ([Text.Encoding]::GetEncoding(0).CodePage -eq 65001) { Write-Output 'SKIP UTF8'; return }
$context = @{ NativeBuildDirectory = (Join-Path $env:TEST_ROOT 'native'); BuildId = 'test' }
$files = Join-Path $context.NativeBuildDirectory 'CMakeFiles'
New-Item -ItemType Directory -Force -Path $files | Out-Null
$bad = [Text.Encoding]::GetEncoding(0).GetString([Text.Encoding]::UTF8.GetBytes('é: à:  '))
$rules = Join-Path $files 'rules.ninja'
$original = "msvc_deps_prefix = $bad`n"
[IO.File]::WriteAllText($rules, $original)
$deps = Join-Path $context.NativeBuildDirectory '.ninja_deps'
[IO.File]::WriteAllText($deps, 'previous dependency records')
$rejected = $false
try { Repair-MsvcDependencyPrefixEncoding $context }
catch {
    if ($_.Exception.Message -notlike '*unused -NativeBuildDirectory*') { throw }
    $rejected = $true
}
if (-not $rejected) { throw 'Unsafe existing cache was accepted' }
if ([IO.File]::ReadAllText($rules) -cne $original) { throw 'Existing rules changed' }
if ([IO.File]::ReadAllText($deps) -cne 'previous dependency records') { throw 'Existing dependencies changed' }
if (Test-Path -LiteralPath "$rules.before-prefix-fix-test") { throw 'Rejection happened after writing' }
''')
    if 'SKIP UTF8' in output:
        pytest.skip('The system ANSI encoding is already UTF-8')


def test_resource_preparation_retains_previous_files_and_rejects_outside_paths(tmp_path):
    invoke_functions(tmp_path, r'''
$packaging = Join-Path $env:TEST_ROOT 'packaging'
$context = New-BuildContext '' (Join-Path $env:TEST_ROOT 'dist') $packaging
New-Item -ItemType Directory -Force -Path $context.ResourceDirectory | Out-Null
$sentinel = Join-Path $context.ResourceDirectory 'keep.txt'
[IO.File]::WriteAllText($sentinel, 'previous resources')
Preserve-ReleaseDirectory $context.ResourceDirectory $context
$backup = "$($context.ResourceDirectory).previous-$($context.BuildId)"
if ([IO.File]::ReadAllText((Join-Path $backup 'keep.txt')) -ne 'previous resources') { throw 'Lost old resources' }
$outside = Join-Path $env:TEST_ROOT 'outside'
New-Item -ItemType Directory -Force -Path $outside | Out-Null
$rejected = $false
try { Preserve-ReleaseDirectory $outside $context }
catch { $rejected = $true }
if (-not $rejected -or -not (Test-Path -LiteralPath $outside)) { throw 'Outside path was moved' }
''')


def test_nonempty_stage_is_preserved(tmp_path):
    invoke_functions(tmp_path, r'''
$stage = Join-Path $env:TEST_ROOT 'existing stage'
New-Item -ItemType Directory -Force -Path $stage | Out-Null
$sentinel = Join-Path $stage 'keep.txt'
[IO.File]::WriteAllText($sentinel, 'preserve')
$rejected = $false
try { Initialize-EmptyDirectory $stage }
catch { $rejected = $true }
if (-not $rejected -or [IO.File]::ReadAllText($sentinel) -ne 'preserve') { throw 'Existing stage was changed' }
''')


def test_frontend_cache_defaults_to_build_directory_and_preserves_override(tmp_path):
    """Use a writable release cache while honoring an explicit npm cache directory."""
    invoke_functions(tmp_path, r'''
$context = New-BuildContext '' (Join-Path $env:TEST_ROOT 'dist') (Join-Path $env:TEST_ROOT 'packaging')
$env:npm_config_cache = $null
Initialize-FrontendCache $context
$expected = Join-Path $context.BuildDirectory 'npm-cache'
if ($env:npm_config_cache -ne $expected -or -not (Test-Path -LiteralPath $expected)) {
    throw 'Default npm cache is not inside the build directory'
}
$explicit = Join-Path $env:TEST_ROOT 'explicit cache'
$env:npm_config_cache = $explicit
Initialize-FrontendCache $context
if ($env:npm_config_cache -ne $explicit -or -not (Test-Path -LiteralPath $explicit)) {
    throw 'Explicit npm cache was replaced'
}
''')


def test_clean_is_rejected_before_modifying_release_resources(tmp_path):
    if sys.platform != 'win32' or not POWERSHELL:
        pytest.skip('Windows PowerShell execution is required')
    output = tmp_path / 'previous output'
    output.mkdir()
    sentinel = output / 'previous.exe'
    sentinel.write_bytes(b'previous complete release')
    result = subprocess.run(
        [POWERSHELL, '-NoProfile', '-NonInteractive', '-File', str(BUILD_SCRIPT),
         '-Clean', '-OutputDirectory', str(output)],
        capture_output=True, text=True, encoding='utf-8', timeout=30,
    )
    assert result.returncode != 0
    assert '-Clean is not supported' in result.stderr
    assert sentinel.read_bytes() == b'previous complete release'


def test_portable_preflight_rejects_python_without_pyinstaller_before_touching_output(tmp_path):
    if sys.platform != 'win32' or not POWERSHELL:
        pytest.skip('Windows PowerShell execution is required')
    environment = tmp_path / 'stdlib-only python'
    subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(environment)],
                   check=True, capture_output=True, timeout=30)
    output = tmp_path / 'previous output'
    output.mkdir()
    previous = output / 'previous-portable.zip'
    previous.write_bytes(b'previous complete archive')
    script = tmp_path / 'preflight.ps1'
    script.write_text(r'''
$ErrorActionPreference = 'Stop'
. $env:TEST_BUILD_SCRIPT
$context = New-BuildContext $env:TEST_VENV $env:TEST_OUTPUT
$context.Python = Join-Path $context.VenvDirectory 'Scripts/python.exe'
Build-PortableArchive $context
''', encoding='utf-8')
    result = subprocess.run(
        [POWERSHELL, '-NoProfile', '-NonInteractive', '-File', str(script)],
        env=dict(os.environ, TEST_BUILD_SCRIPT=str(BUILD_SCRIPT),
                 TEST_VENV=str(environment), TEST_OUTPUT=str(output)),
        capture_output=True, text=True, encoding='utf-8', timeout=30,
    )
    assert result.returncode != 0
    assert 'requires PyInstaller' in result.stderr
    assert '-ReleaseVenv' in result.stderr
    assert previous.read_bytes() == b'previous complete archive'
    assert list(output.iterdir()) == [previous]


def test_visual_studio_import_selects_supported_2022_toolchain(tmp_path):
    if sys.platform != 'win32' or not POWERSHELL:
        pytest.skip('Windows PowerShell execution is required')
    installer = Path(os.environ.get('ProgramFiles(x86)', 'C:/Program Files (x86)'))
    vswhere = installer / 'Microsoft Visual Studio/Installer/vswhere.exe'
    if not vswhere.is_file():
        pytest.skip('Visual Studio Installer is not installed')
    installed = subprocess.run(
        [str(vswhere), '-latest', '-products', '*', '-version', '[17.0,18.0)',
         '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
         '-property', 'installationPath'],
        capture_output=True, text=True, encoding='utf-8', timeout=15,
    )
    if installed.returncode != 0 or not installed.stdout.strip():
        pytest.skip('Visual Studio 2022 C++ tools are not installed')
    invoke_functions(tmp_path, r'''
Import-VisualStudioEnvironment
if ($env:VisualStudioVersion -notlike '17.*') { throw 'Selected an unsupported Visual Studio version' }
if ($env:VSLANG -ne '1033') { throw 'Compiler diagnostics are not deterministic across languages' }
$compiler = Get-Command cl.exe -ErrorAction Stop
if (-not $compiler.Source.StartsWith($env:VSINSTALLDIR, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Compiler does not belong to the selected Visual Studio instance'
}
$bundledNinja = Join-Path $env:VSINSTALLDIR 'Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe'
if (Test-Path -LiteralPath $bundledNinja) {
    $ninja = Get-Command ninja.exe -ErrorAction Stop
    if ($ninja.Source -ne $bundledNinja) { throw 'The bundled Ninja directory was not added to PATH' }
}
''')
