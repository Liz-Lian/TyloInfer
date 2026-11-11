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


def test_clean_uses_fresh_caches_and_preserves_explicit_locations(tmp_path):
    """Keep normal incremental caches while requiring fresh explicit clean-build locations."""
    invoke_functions(tmp_path, r'''
$env:CARGO_TARGET_DIR = $null
$env:TYLOINFER_RELEASE_NATIVE_BUILD_DIR = $null
$packaging = Join-Path $env:TEST_ROOT 'packaging'
$output = Join-Path $env:TEST_ROOT 'dist'
$normal = New-BuildContext '' $output $packaging
$fresh = New-BuildContext '' $output $packaging -Clean
if (-not $fresh.NativeBuildDirectory.StartsWith($fresh.BuildDirectory)) { throw 'Clean reused native cache' }
if (-not $fresh.CargoTargetDirectory.StartsWith($fresh.BuildDirectory)) { throw 'Clean reused Cargo cache' }
if ($fresh.NativeBuildDirectory -eq $normal.NativeBuildDirectory) { throw 'Clean reused normal native cache' }
$explicit = Join-Path $env:TEST_ROOT 'explicit-native'
$env:CARGO_TARGET_DIR = Join-Path $env:TEST_ROOT 'explicit-cargo'
$chosen = New-BuildContext '' $output $packaging -NativeBuildDirectory $explicit -Clean
if ($chosen.NativeBuildDirectory -ne $explicit -or $chosen.CargoTargetDirectory -ne $env:CARGO_TARGET_DIR) { throw 'Changed explicit cache locations' }
New-Item -ItemType Directory -Force -Path $explicit | Out-Null
$sentinel = Join-Path $explicit 'keep.obj'
[IO.File]::WriteAllText($sentinel, 'previous object')
$rejected = $false
try { New-BuildContext '' $output $packaging -NativeBuildDirectory $explicit -Clean | Out-Null }
catch { if ($_.Exception.Message -notlike '*empty explicitly selected*') { throw }; $rejected = $true }
if (-not $rejected -or [IO.File]::ReadAllText($sentinel) -cne 'previous object') { throw 'Destroyed or accepted nonempty cache' }
''')


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows developer environment')
def test_active_developer_environment_preserves_tools_language_and_path(tmp_path):
    """Use the caller's valid environment instead of selecting a different installed VS."""
    invoke_functions(tmp_path, r'''
$tools = Join-Path $env:TEST_ROOT 'caller-tools'
New-Item -ItemType Directory -Force -Path $tools | Out-Null
[IO.File]::WriteAllText((Join-Path $tools 'cl.cmd'), '@exit /b 0')
[IO.File]::WriteAllText((Join-Path $tools 'ninja.cmd'), '@exit /b 0')
$env:PATH = "$tools;$env:PATH"
$env:INCLUDE = 'caller includes'
$env:LIB = 'caller libraries'
$env:VSCMD_ARG_TGT_ARCH = 'x64'
$env:VSLANG = 'caller-language'
$env:TYLOINFER_RELEASE_VS_PATH = $null
$before = $env:PATH
$ninja = (Get-Command ninja).Source
Import-VisualStudioEnvironment
if ($env:PATH -cne $before -or $env:VSLANG -cne 'caller-language') { throw 'Changed caller environment' }
if ((Get-Command ninja).Source -ne $ninja) { throw 'Changed selected Ninja' }
$env:VSCMD_ARG_TGT_ARCH = 'x86'
$rejected = $false
try { Import-VisualStudioEnvironment }
catch { if ($_.Exception.Message -notlike '*x64*') { throw }; $rejected = $true }
if (-not $rejected) { throw 'Accepted an incompatible target environment' }
''')


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows developer environment')
def test_explicit_visual_studio_preserves_caller_ninja_and_language(tmp_path):
    """Initialize the selected installation without imposing a VS year or unrelated tools."""
    invoke_functions(tmp_path, r'''
$install = Join-Path $env:TEST_ROOT 'selected VS'
$tools = Join-Path $install 'compiler'
$caller = Join-Path $env:TEST_ROOT 'caller'
$batch = Join-Path $install 'Common7/Tools/VsDevCmd.bat'
New-Item -ItemType Directory -Force -Path $tools, $caller, (Split-Path $batch) | Out-Null
[IO.File]::WriteAllText((Join-Path $tools 'cl.cmd'), '@exit /b 0')
[IO.File]::WriteAllText((Join-Path $caller 'ninja.cmd'), '@exit /b 0')
[IO.File]::WriteAllText($batch, "@echo off`nset `"PATH=$tools;%PATH%`"`nset `"INCLUDE=chosen`"`nset `"LIB=chosen`"`nset `"VSCMD_ARG_TGT_ARCH=x64`"`nset `"VSLANG=changed-by-batch`"`n")
$env:PATH = "$caller;$env:PATH"
$env:VSLANG = 'caller-language'
$ninja = (Get-Command ninja).Source
Import-VisualStudioEnvironment -InstallationPath $install
if ((Get-Command cl).Source -ne (Join-Path $tools 'cl.cmd')) { throw 'Ignored selected installation' }
if ((Get-Command ninja).Source -ne $ninja -or $env:VSLANG -cne 'caller-language') { throw 'Overrode unrelated caller settings' }
''')


@pytest.mark.parametrize('variable,value', [('Jobs', '-1'), ('TYLOINFER_RELEASE_JOBS', '0'), ('TYLOINFER_RELEASE_JOBS', 'invalid')])
def test_invalid_parallelism_is_rejected(tmp_path, variable, value):
    """Fail invalid concurrency settings before starting any compilation."""
    invoke_functions(tmp_path, r'''
$env:TYLOINFER_RELEASE_JOBS = 'ENV_VALUE'
$rejected = $false
try { Get-ReleaseJobs REQUESTED | Out-Null }
catch { if ($_.Exception.Message -notlike '*positive*') { throw }; $rejected = $true }
if (-not $rejected) { throw 'Accepted invalid parallelism' }
'''.replace('ENV_VALUE', value if variable != 'Jobs' else '').replace('REQUESTED', value if variable == 'Jobs' else '0'))


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows script entry')
def test_script_entry_restores_all_environment_on_failure(tmp_path):
    """Restore existing and newly created process variables when an invoked build fails."""
    invoke_functions(tmp_path, r'''
$env:TYLOINFER_RELEASE_JOBS = $null
$before = [Environment]::GetEnvironmentVariables('Process')
function Get-CimInstance {
    $env:PATH = 'changed'
    $env:VSLANG = 'changed'
    $env:TEMP = 'changed'
    $env:CFLAGS = 'changed'
    $env:VSCMD_ARG_TGT_ARCH = 'changed'
    $env:TYLOINFER_TEST_NEW_ENVIRONMENT = 'new'
    throw 'controlled initialization failure'
}
$rejected = $false
try { & $env:TEST_BUILD_SCRIPT }
catch { if ($_.Exception.Message -notlike '*controlled initialization failure*') { throw }; $rejected = $true }
if (-not $rejected) { throw 'Expected controlled failure' }
$after = [Environment]::GetEnvironmentVariables('Process')
if ($before.Count -ne $after.Count) { throw 'Environment variable count changed' }
foreach ($key in $before.Keys) { if ($before[$key] -cne $after[$key]) { throw "Leaked environment variable: $key" } }
''')


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows file sharing')
def test_checkout_lock_rejects_a_second_process_and_releases_for_the_owner(tmp_path):
    """Serialize one checkout while allowing independent checkouts and later invocations."""
    if not POWERSHELL:
        pytest.skip('PowerShell 7 is required')
    environment = dict(os.environ, TEST_BUILD_SCRIPT=str(BUILD_SCRIPT), TEST_ROOT=str(tmp_path))
    owner = subprocess.Popen(
        [POWERSHELL, '-NoProfile', '-NonInteractive', '-Command',
         '. $env:TEST_BUILD_SCRIPT; $lock = Get-CheckoutBuildLock $env:TEST_ROOT; '
         'Write-Output READY; [Console]::ReadLine() | Out-Null; $lock.Dispose()'],
        env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        assert owner.stdout.readline().strip() == 'READY'
        invoke_functions(tmp_path, r'''
$rejected = $false
try { $unexpected = Get-CheckoutBuildLock $env:TEST_ROOT; $unexpected.Dispose() }
catch { if ($_.Exception.Message -notlike '*checkout release lock*') { throw }; $rejected = $true }
if (-not $rejected) { throw 'Concurrent checkout build was accepted' }
$other = Get-CheckoutBuildLock (Join-Path $env:TEST_ROOT 'other checkout')
$other.Dispose()
''')
        owner.communicate('\n', timeout=10)
        assert owner.returncode == 0
        invoke_functions(tmp_path, '$lock = Get-CheckoutBuildLock $env:TEST_ROOT; $lock.Dispose()')
    finally:
        if owner.poll() is None:
            owner.terminate()
            owner.wait(timeout=5)


def test_cuda_selection_honors_explicit_compilers_and_numeric_versions(tmp_path):
    """Use one selected SDK and reject invalid overrides instead of silently switching."""
    invoke_functions(tmp_path, r'''
$first = Join-Path $env:TEST_ROOT 'first SDK'
$second = Join-Path $env:TEST_ROOT 'second SDK'
foreach ($root in @($first, $second)) {
    New-Item -ItemType Directory -Force -Path (Join-Path $root 'bin') | Out-Null
    [IO.File]::WriteAllText((Join-Path $root 'bin/nvcc.exe'), 'fixture')
}
$script:pathCompiler = Join-Path $second 'bin/nvcc.exe'
function Get-Command { param($Name, $ErrorAction); if ($Name -eq 'nvcc' -and $script:pathCompiler) { [pscustomobject]@{ Source = $script:pathCompiler } } }
$env:CUDACXX = Join-Path $first 'bin/nvcc.exe'
$env:CUDA_HOME = $second
$env:CUDA_PATH = $second
if ((Find-CudaRoot) -ne $first) { throw 'CUDACXX was ignored' }
$env:CUDACXX = Join-Path $first 'missing.exe'
$rejected = $false
try { Find-CudaRoot | Out-Null }
catch { if ($_.Exception.Message -notlike '*CUDACXX*') { throw }; $rejected = $true }
if (-not $rejected) { throw 'Invalid CUDACXX fell back silently' }
$env:CUDACXX = $null
$env:CUDA_HOME = $first
if ((Find-CudaRoot) -ne $second) { throw 'PATH compiler was ignored' }
$script:pathCompiler = $null
if ((Find-CudaRoot) -ne $first) { throw 'CUDA_HOME was ignored' }
$env:CUDA_HOME = Join-Path $env:TEST_ROOT 'missing SDK'
$rejected = $false
try { Find-CudaRoot | Out-Null }
catch { if ($_.Exception.Message -notlike '*CUDA_HOME*') { throw }; $rejected = $true }
if (-not $rejected) { throw 'Invalid CUDA_HOME fell back silently' }
$env:CUDA_HOME = $null
if ((Find-CudaRoot) -ne $second) { throw 'CUDA_PATH was ignored' }
$env:CUDA_PATH = $null
$env:ProgramFiles = Join-Path $env:TEST_ROOT 'installed'
foreach ($version in @('v9.0', 'v12.0')) {
    $bin = Join-Path $env:ProgramFiles "NVIDIA GPU Computing Toolkit/CUDA/$version/bin"
    New-Item -ItemType Directory -Force -Path $bin | Out-Null
    [IO.File]::WriteAllText((Join-Path $bin 'nvcc.exe'), 'fixture')
}
if ((Find-CudaRoot) -notlike '*v12.0') { throw 'Automatic discovery sorted versions lexically' }
''')


@pytest.mark.skipif(sys.platform != 'win32' or not shutil.which('npm.cmd'), reason='Windows npm configuration')
@pytest.mark.parametrize('explicit_cache', [False, True])
def test_build_preparation_preserves_npmrc_and_explicit_cache(tmp_path, explicit_cache):
    """Leave cache precedence to npm, including project configuration and environment overrides."""
    invoke_functions(tmp_path, r'''
$script:npm = (Get-Command npm.cmd).Source
$context = New-BuildContext '' (Join-Path $env:TEST_ROOT 'output') (Join-Path $env:TEST_ROOT 'packaging')
New-Item -ItemType Directory -Force -Path $context.StudioDirectory | Out-Null
$configuredCache = Join-Path $env:TEST_ROOT 'project cache'
[IO.File]::WriteAllText((Join-Path $context.StudioDirectory '.npmrc'), "cache=$($configuredCache.Replace('\', '/'))`n")
[IO.File]::WriteAllText((Join-Path $context.StudioDirectory 'package.json'), '{"name":"cache-fixture","version":"1.0.0"}')
$env:NPM_CONFIG_USERCONFIG = Join-Path $env:TEST_ROOT 'empty-user-config'
$env:NPM_CONFIG_GLOBALCONFIG = Join-Path $env:TEST_ROOT 'empty-global-config'
$env:npm_config_cache = EXPLICIT
$expected = if ($env:npm_config_cache) { $env:npm_config_cache } else { $configuredCache }
function Import-VisualStudioEnvironment { param($InstallationPath) }
function Find-CudaRoot { $env:TEST_ROOT }
function Initialize-ReleasePathMapping { param($Context) }
function Require-Command { param($Name); if ($Name -eq 'npm.cmd') { return $script:npm }; return 'fixture-tool' }
Initialize-BuildEnvironment $context
if ($env:npm_config_cache -cne EXPLICIT) { throw 'Build preparation overrode npm configuration' }
Push-Location $context.StudioDirectory
try { $actual = & $script:npm config get cache; if ($LASTEXITCODE -ne 0) { throw 'npm config failed' } }
finally { Pop-Location }
if ([IO.Path]::GetFullPath($actual.Trim()) -ne [IO.Path]::GetFullPath($expected)) { throw 'npmrc cache precedence changed' }
'''.replace('EXPLICIT', "(Join-Path $env:TEST_ROOT 'explicit cache')" if explicit_cache else '$null'))


def test_native_build_delegates_dependency_metadata_to_cmake(tmp_path):
    """Build configured outputs without probing compilers or modifying generated Ninja metadata."""
    invoke_functions(tmp_path, r'''
$context = New-BuildContext '' (Join-Path $env:TEST_ROOT 'out') (Join-Path $env:TEST_ROOT 'packaging')
$context.Tools.CMake = 'fixture-cmake'
$context.CudaRoot = Join-Path $env:TEST_ROOT 'selected CUDA'
New-Item -ItemType Directory -Force -Path $context.NativeBuildDirectory, $context.SidecarDirectory | Out-Null
$script:commands = @()
$script:rules = Join-Path $context.NativeBuildDirectory 'CMakeFiles/rules.ninja'
function Get-ReleaseNativeFlags { param($Context); '/DPORTABLE' }
function Invoke-Checked {
    param($FilePath, $Arguments)
    if ($FilePath -ne 'fixture-cmake') { throw 'Invoked an unexpected tool' }
    $script:commands += ,$Arguments
    if ($Arguments[0] -eq '-S') {
        New-Item -ItemType Directory -Force -Path (Split-Path $script:rules) | Out-Null
        [IO.File]::WriteAllText($script:rules, "msvc_deps_prefix = compiler-owned: `n  deps = msvc`n")
        $script:originalHash = (Get-FileHash -LiteralPath $script:rules).Hash
    } elseif ($Arguments[0] -eq '--build') {
        foreach ($name in @('mfq-runtime.exe', 'mfq-diagnostics.exe')) {
            [IO.File]::WriteAllText((Join-Path $context.NativeBuildDirectory $name), 'built output')
        }
    } else { throw 'Unexpected CMake invocation' }
}
Build-NativeSidecar $context '80' 1
if ($script:commands.Count -ne 2) { throw 'Unexpected toolchain self-test' }
if ((Get-FileHash -LiteralPath $script:rules).Hash -cne $script:originalHash) { throw 'Rewrote CMake metadata' }
if (-not (Test-Path -LiteralPath $context.NativeArtifact)) { throw 'Runtime output was not staged' }
$expected = "-DCMAKE_CUDA_COMPILER=$([IO.Path]::Combine($context.CudaRoot, 'bin', 'nvcc.exe'))"
if ($script:commands[0] -notcontains $expected) { throw 'Configured a different CUDA Toolkit' }
function Invoke-Checked { param($FilePath, $Arguments); throw 'unsupported compiler fixture' }
$rejected = $false
try { Build-NativeSidecar $context '80' 1 }
catch {
    if ($_.Exception.Message -notlike '*VisualStudioPath*unsupported compiler fixture*') { throw }
    $rejected = $true
}
if (-not $rejected) { throw 'Native configuration error was swallowed' }
''')


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows path aliases')
def test_external_native_cache_is_included_in_privacy_path_mappings(tmp_path):
    """Remap and scan external cache paths using the same prefix, including Unicode and spaces."""
    invoke_functions(tmp_path, r'''
$external = Join-Path $env:TEST_ROOT '外部 native cache'
New-Item -ItemType Directory -Force -Path $external | Out-Null
$packaging = Join-Path $env:TEST_ROOT 'checkout/packaging'
$context = New-BuildContext '' (Join-Path $env:TEST_ROOT 'out') $packaging -NativeBuildDirectory $external
$mappings = @(Get-ReleasePathMappings $context)
if (-not @($mappings | Where-Object { $_.Source -eq $external -and $_.Destination -eq '/mfq-build/native' }).Count) {
    throw 'External native build paths were omitted from the shared mappings'
}
$flags = @(Get-ReleaseNativeFlags $context) -join ' '
if (-not $flags.Contains("/pathmap:$external=/mfq-build/native")) { throw 'External native path was not remapped' }
''')
