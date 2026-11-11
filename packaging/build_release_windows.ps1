# Build the Windows x64 TyloInfer installer, single-file app, and CUDA sidecar.
#
# The release Python environment is isolated from development environments. It
# contains only the base dependencies needed by `mfq serve`; training,
# Calibration, quantization, MiniCPM-o, and PyTorch are not packaged.
# The script imports the Visual Studio environment and adds all required tools to
# its own PATH, so it can be run from ordinary PowerShell.

#requires -Version 7
[CmdletBinding()]
param(
    [string] $PythonVersion = $env:TYLOINFER_RELEASE_PYTHON,
    [string] $ReleaseVenv = $env:TYLOINFER_RELEASE_VENV,
    [string] $OutputDirectory = $env:TYLOINFER_RELEASE_OUTPUT_DIR,
    [string] $CudaArchitectures = $env:TYLOINFER_RELEASE_CUDA_ARCHITECTURES,
    [int] $Jobs = 0,
    [switch] $PortableOnly,
    [switch] $Clean,
    [string] $NativeBuildDirectory = $env:TYLOINFER_RELEASE_NATIVE_BUILD_DIR
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Fail {
    param([Parameter(Mandatory)][string] $Message)

    throw "error: $Message"
}

function Resolve-BuildPath {
    param(
        [Parameter(Mandatory)][string] $Path,
        [Parameter(Mandatory)][string] $BaseDirectory
    )

    if ([System.IO.Path]::IsPathFullyQualified($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $BaseDirectory $Path))
}

function Add-ToPath {
    param([Parameter(Mandatory)][string] $Directory)

    if (-not (Test-Path -LiteralPath $Directory -PathType Container)) {
        return
    }
    if (($env:PATH -split [System.IO.Path]::PathSeparator) -notcontains $Directory) {
        $env:PATH = "$Directory$([System.IO.Path]::PathSeparator)$env:PATH"
    }
}

function Require-Command {
    param([Parameter(Mandatory)][string] $Name)

    $command = Get-Command $Name -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $command) {
        Fail "$Name is required; install it and make it available to this release script"
    }
    return $command.Source
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory)][string] $FilePath,
        [string[]] $Arguments = @()
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        Fail "command failed ($LASTEXITCODE): $FilePath $($Arguments -join ' ')"
    }
}

function Import-VisualStudioEnvironment {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path -LiteralPath $vswhere -PathType Leaf)) {
        Fail "Visual Studio Build Tools with the C++ workload are required"
    }

    $installDirectory = & $vswhere -latest -products * -version '[17.0,18.0)' `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath | Select-Object -First 1
    if ([string]::IsNullOrWhiteSpace($installDirectory)) {
        $installDirectory = & $vswhere -all -products * -version '[17.0,18.0)' -property installationPath |
            Where-Object {
                Test-Path -LiteralPath (Join-Path $_ "Common7\Tools\VsDevCmd.bat") `
                    -PathType Leaf
            } |
            Select-Object -First 1
    }
    if ([string]::IsNullOrWhiteSpace($installDirectory)) {
        Fail "Visual Studio 2022 C++ x64 tools are required for the supported CUDA toolchain"
    }

    $vsDevCmd = Join-Path $installDirectory.Trim() "Common7\Tools\VsDevCmd.bat"
    if (-not (Test-Path -LiteralPath $vsDevCmd -PathType Leaf)) {
        Fail "VsDevCmd.bat is missing from $installDirectory"
    }

    $env:VSLANG = '1033'
    $environmentLines = & cmd.exe /d /s /c "call `"$vsDevCmd`" -no_logo -arch=x64 -host_arch=x64 >nul && set"
    if ($LASTEXITCODE -ne 0) {
        Fail "failed to initialize the Visual Studio x64 environment"
    }
    foreach ($line in $environmentLines) {
        $separator = $line.IndexOf('=')
        if ($separator -le 0) {
            continue
        }
        Set-Item -Path "Env:$($line.Substring(0, $separator))" -Value $line.Substring($separator + 1)
    }
    Add-ToPath (Join-Path $installDirectory.Trim() 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja')
}

function Find-CudaRoot {
    if (-not [string]::IsNullOrWhiteSpace($env:CUDA_PATH)) {
        if (Test-Path -LiteralPath (Join-Path $env:CUDA_PATH "bin\nvcc.exe") -PathType Leaf) {
            return $env:CUDA_PATH
        }
    }

    $cudaParent = Join-Path ${env:ProgramFiles} "NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path -LiteralPath $cudaParent -PathType Container) {
        $cudaDirectory = Get-ChildItem -LiteralPath $cudaParent -Directory |
            Sort-Object -Property Name -Descending |
            Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName "bin\nvcc.exe") -PathType Leaf } |
            Select-Object -First 1
        if ($null -ne $cudaDirectory) {
            return $cudaDirectory.FullName
        }
    }
    Fail "CUDA Toolkit with nvcc is required"
}

function Find-CudaRuntimeDirectory {
    param([Parameter(Mandatory)][string] $CudaRoot)

    foreach ($candidate in @(
            (Join-Path $CudaRoot "bin\x64"),
            (Join-Path $CudaRoot "bin"))) {
        if (Test-Path -LiteralPath $candidate -PathType Container) {
            $runtime = Get-ChildItem -LiteralPath $candidate -File `
                -Filter "cudart64_*.dll" -ErrorAction SilentlyContinue |
                Select-Object -First 1
            if ($null -ne $runtime) {
                return $candidate
            }
        }
    }
    Fail "the CUDA Toolkit runtime DLL directory is missing from $CudaRoot"
}

function Copy-RuntimeDlls {
    param(
        [Parameter(Mandatory)][string] $SourceDirectory,
        [Parameter(Mandatory)][string] $DestinationDirectory,
        [string[]] $Patterns = @("*.dll")
    )

    if (-not (Test-Path -LiteralPath $SourceDirectory -PathType Container)) {
        return
    }
    foreach ($pattern in $Patterns) {
        Get-ChildItem -LiteralPath $SourceDirectory -File -Filter $pattern -ErrorAction SilentlyContinue |
            ForEach-Object {
                Copy-Item -LiteralPath $_.FullName -Destination $DestinationDirectory -Force
            }
    }
}

function Copy-VisualCppRuntimeDlls {
    param([Parameter(Mandatory)][string] $DestinationDirectory)

    if ([string]::IsNullOrWhiteSpace($env:VCToolsRedistDir)) {
        Fail "the Visual Studio environment did not provide VCToolsRedistDir"
    }
    $x64Directory = Join-Path $env:VCToolsRedistDir "x64"
    if (-not (Test-Path -LiteralPath $x64Directory -PathType Container)) {
        Fail "the Visual C++ x64 redistributable directory is missing from $env:VCToolsRedistDir"
    }

    $crtDirectory = Get-ChildItem -LiteralPath $x64Directory -Directory |
        Where-Object { $_.Name -match '^Microsoft\.VC\d+\.CRT$' } |
        Select-Object -First 1
    if ($null -eq $crtDirectory) {
        Fail "the Visual C++ x64 CRT redistributable is missing from $x64Directory"
    }
    Copy-RuntimeDlls $crtDirectory.FullName $DestinationDirectory

    # CMake may enable OpenMP for mfq-runtime when the VS workload provides it.
    # Staging the runtime when present is harmless if OpenMP was not selected.
    Get-ChildItem -LiteralPath $x64Directory -Directory |
        Where-Object { $_.Name -match '^Microsoft\.VC\d+\.OpenMP$' } |
        ForEach-Object {
            Copy-RuntimeDlls $_.FullName $DestinationDirectory
        }
}

function Initialize-EmptyDirectory {
    <# .SYNOPSIS
    Create a staging directory and reject stale contents instead of deleting them.
    #>
    param([Parameter(Mandatory)][string] $Path)

    if ((Test-Path -LiteralPath $Path) -and
            @(Get-ChildItem -LiteralPath $Path -Force).Count -gt 0) {
        Fail "staging directory is not empty; preserve it and choose a fresh build: $Path"
    }
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

function Preserve-ReleaseDirectory {
    <# .SYNOPSIS
    Retain an existing resource directory under a verified sibling backup path.
    #>
    param([Parameter(Mandatory)][string] $Path, [Parameter(Mandatory)][hashtable] $Context)

    $root = [IO.Path]::GetFullPath($Context.ScriptDirectory).TrimEnd('\') + '\'
    $target = [IO.Path]::GetFullPath($Path)
    $backup = "$target.previous-$($Context.BuildId)"
    if (-not $target.StartsWith($root, [StringComparison]::OrdinalIgnoreCase) -or
            -not $backup.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) {
        Fail "resource paths must remain inside the packaging directory: $target"
    }
    if (Test-Path -LiteralPath $target) {
        Move-Item -LiteralPath $target -Destination $backup
    }
}

function Get-ReleaseJobs {
    param([int] $RequestedJobs)

    if ($RequestedJobs -gt 0) {
        return $RequestedJobs
    }
    if ($env:TYLOINFER_RELEASE_JOBS) {
        return [int] $env:TYLOINFER_RELEASE_JOBS
    }

    # The largest native CUDA translation units need several GiB each, so bound
    # parallelism by installed memory.
    $memoryJobs = [math]::Floor((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 5GB)
    return [math]::Max(1, [math]::Min([int] $memoryJobs, [Environment]::ProcessorCount))
}

function New-BuildContext {
    <# .SYNOPSIS
    Isolate release staging while reusing the selected native incremental cache.
    #>
    param(
        [Parameter(Mandatory)][AllowEmptyString()][string] $Venv,
        [Parameter(Mandatory)][string] $Output,
        [string] $ScriptDirectory = $PSScriptRoot,
        [string] $NativeBuildDirectory = $env:TYLOINFER_RELEASE_NATIVE_BUILD_DIR
    )

    $projectDirectory = [System.IO.Path]::GetFullPath((Join-Path $scriptDirectory ".."))
    $resourceDirectory = Join-Path $scriptDirectory "Resources"
    $sidecarDirectory = Join-Path $scriptDirectory "sidecars"
    $targetTriple = "x86_64-pc-windows-msvc"
    $cliName = "tyloinfer-cli"
    $buildId = [Guid]::NewGuid().ToString("N")
    $buildDirectory = Join-Path $scriptDirectory "build\windows-$buildId"
    $nativeBuildCache = if ([string]::IsNullOrWhiteSpace($NativeBuildDirectory)) {
        Join-Path $scriptDirectory 'build\runtime'
    } else {
        Resolve-BuildPath $NativeBuildDirectory $projectDirectory
    }

    return @{
        ScriptDirectory = $scriptDirectory
        BuildId = $buildId
        BuildDirectory = $buildDirectory
        ProjectDirectory = $projectDirectory
        StudioDirectory = Join-Path $projectDirectory "TyloInferStudio"
        PyInstallerBuildDirectory = Join-Path $buildDirectory "pyinstaller"
        SpecDirectory = Join-Path $buildDirectory "spec"
        NativeBuildDirectory = $nativeBuildCache
        SidecarDirectory = $sidecarDirectory
        ResourceDirectory = $resourceDirectory
        WindowsRuntimeDirectory = Join-Path $scriptDirectory "windows-runtime"
        TargetTriple = $targetTriple
        CliName = $cliName
        CliArtifact = Join-Path (Join-Path $resourceDirectory $cliName) "$cliName.exe"
        NativeArtifact = Join-Path $sidecarDirectory "mfq-runtime-$targetTriple.exe"
        VenvDirectory = if ($Venv) { $Venv } else { Join-Path $buildDirectory "venv" }
        OutputDirectory = $Output
        CudaRoot = $null
        Python = $null
        PyInstaller = $null
        NativeOutput = $null
        Tools = @{ Uv = $null; CMake = $null; Npm = $null }
    }
}

function Initialize-FrontendCache {
    <# .SYNOPSIS
    Use a writable build-local npm cache unless an explicit cache was selected.
    #>
    param([Parameter(Mandatory)][hashtable] $Context)

    if ([string]::IsNullOrWhiteSpace($env:npm_config_cache)) {
        $env:npm_config_cache = Join-Path $Context.BuildDirectory 'npm-cache'
    }
    New-Item -ItemType Directory -Force -Path $env:npm_config_cache | Out-Null
}

function Initialize-BuildEnvironment {
    param([Parameter(Mandatory)][hashtable] $Context)

    Add-ToPath (Join-Path ${env:ProgramFiles} "CMake\bin")
    Add-ToPath (Join-Path $env:USERPROFILE ".cargo\bin")
    Add-ToPath ([System.IO.Path]::Combine(
        [Environment]::GetFolderPath([Environment+SpecialFolder]::ProgramFilesX86), "NSIS"))
    Import-VisualStudioEnvironment

    $Context.CudaRoot = Find-CudaRoot
    $env:CUDA_PATH = $Context.CudaRoot
    Add-ToPath (Join-Path $Context.CudaRoot "bin")
    Add-ToPath (Join-Path $Context.CudaRoot "lib\x64")

    $Context.Tools.Uv = Require-Command "uv"
    $Context.Tools.CMake = Require-Command "cmake"
    $null = Require-Command "ninja"
    $null = Require-Command "cargo"
    $Context.Tools.Npm = Require-Command "npm.cmd"
    Initialize-FrontendCache $Context
    $null = Require-Command "cl"
    $null = Require-Command "nvcc"
    # Tauri downloads its pinned NSIS toolset when no local installation exists.
    Initialize-ReleasePathMapping $Context
}

function Get-ReleasePathMappings {
    param([Parameter(Mandatory)][hashtable] $Context)

    $mappings = @(
        [pscustomobject]@{ Source = $Context.ProjectDirectory; Destination = "/mfq-src" }
        [pscustomobject]@{ Source = $env:USERPROFILE; Destination = "/mfq-build/home" }
    )
    if (-not [string]::IsNullOrWhiteSpace($env:CARGO_TARGET_DIR)) {
        $mappings += [pscustomobject]@{
            Source = Resolve-BuildPath $env:CARGO_TARGET_DIR $Context.StudioDirectory
            Destination = "/mfq-build/target"
        }
    }
    if (-not ('MfqRelease.BuildPaths' -as [type])) {
        Add-Type -TypeDefinition @'
using System.Runtime.InteropServices;
using System.Text;
namespace MfqRelease {
    public static class BuildPaths {
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
        public static extern uint GetShortPathName(string path, StringBuilder output, uint size);
    }
}
'@
    }
    foreach ($mapping in $mappings) {
        $mapping
        # Native dependency generators can use Windows 8.3 path aliases.
        $shortPath = [System.Text.StringBuilder]::new(32768)
        $length = [MfqRelease.BuildPaths]::GetShortPathName($mapping.Source, $shortPath, $shortPath.Capacity)
        if ($length -gt 0 -and $length -lt $shortPath.Capacity -and $shortPath.ToString() -ne $mapping.Source) {
            [pscustomobject]@{ Source = $shortPath.ToString(); Destination = $mapping.Destination }
        }
    }
}

function Get-ReleaseNativeFlags {
    param([Parameter(Mandatory)][hashtable] $Context)

    "/experimental:deterministic"
    foreach ($mapping in @(Get-ReleasePathMappings $Context)) {
        foreach ($source in @($mapping.Source, $mapping.Source.Replace('\', '/')) | Select-Object -Unique) {
            '"/pathmap:' + $source + '=' + $mapping.Destination + '"'
        }
    }
}

function Initialize-ReleasePathMapping {
    param([Parameter(Mandatory)][hashtable] $Context)

    # Cargo's encoded form keeps a path containing spaces in a single argument.
    $separator = [char]31
    $rustArguments = @()
    if ($env:CARGO_ENCODED_RUSTFLAGS) {
        $rustArguments = @($env:CARGO_ENCODED_RUSTFLAGS.Split($separator))
    } elseif ($env:RUSTFLAGS) {
        $rustArguments = @($env:RUSTFLAGS -split '\s+' | Where-Object { $_ })
    }
    foreach ($mapping in @(Get-ReleasePathMappings $Context)) {
        foreach ($source in @($mapping.Source, $mapping.Source.Replace('\', '/')) | Select-Object -Unique) {
            $rustArguments += "--remap-path-prefix=$source=$($mapping.Destination)"
        }
    }
    $rustArguments += @("-C", "link-arg=/PDBALTPATH:tyloinfer.pdb")
    $env:CARGO_ENCODED_RUSTFLAGS = $rustArguments -join $separator
    # Native dependencies built by Rust also expand __FILE__.
    $nativeFlags = @(Get-ReleaseNativeFlags $Context) -join ' '
    $env:CFLAGS = "$env:CFLAGS $nativeFlags".Trim()
    $env:CXXFLAGS = "$env:CXXFLAGS $nativeFlags".Trim()
    $env:CC_SHELL_ESCAPED_FLAGS = "1"
}

function Get-ReleasePathCheckerSource {
    <# .SYNOPSIS
    Return the embedded Python release artifact path scanner.
    #>
    return @'
# TYLOINFER_PATH_CHECKER_BEGIN
"""Reject local build paths in release files and frozen Python code."""

from __future__ import annotations

import argparse
import marshal
from pathlib import Path
from types import CodeType
import zipfile


class PathChecker:
    """Inspect release files and embedded Python metadata for private build paths."""
    def __init__(self, prefixes: list[str]):
        """Prepare normalized path spellings and counters for one release scan."""
        self.needles: set[bytes] = set()
        for prefix in prefixes:
            normalized = prefix.rstrip("/\\").replace("\\", "/")
            for spelling in (normalized, normalized.replace("/", "\\")):
                for escaped in (spelling, spelling.replace("\\", "\\\\")):
                    for encoding in ("utf-8", "utf-16le"):
                        self.needles.add(escaped.lower().encode(encoding))
        self.overlap = max(map(len, self.needles), default=1) - 1
        self.failures: set[str] = set()
        self.files = 0
        self.python_modules = 0

    def check_bytes(self, data: bytes, label: str) -> None:
        """Record the artifact when its bytes contain a private build prefix."""
        lowered = data.lower()
        if any(needle in lowered for needle in self.needles):
            self.failures.add(label)

    def check_code(self, code: CodeType, label: str) -> None:
        """Inspect marshalled code including nested filenames and constants."""
        # Marshalled data includes nested filenames, constant tuples, and shared strings.
        self.check_bytes(marshal.dumps(code), label)

    def check_file(self, path: Path) -> None:
        """Scan a file and inspect compressed Python metadata in ZIP archives."""
        self.files += 1
        label = str(path)
        previous = b""
        with path.open("rb") as stream:
            while block := stream.read(4 * 1024 * 1024):
                data = previous + block
                self.check_bytes(data, label)
                previous = data[-self.overlap:] if self.overlap else b""
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                for name in archive.namelist():
                    data = archive.read(name)
                    member = f"{label}!{name}"
                    self.check_bytes(data, member)
                    if name.endswith(".pyc"):
                        code = marshal.loads(data[16:])
                        if isinstance(code, CodeType):
                            self.check_code(code, member)

    def check_python_cli(self, path: Path) -> None:
        """Inspect every code object embedded in a frozen PyInstaller executable."""
        from PyInstaller.archive.readers import CArchiveReader

        archive = CArchiveReader(str(path))
        for name, entry in archive.toc.items():
            if entry[-1] == "z":
                pyz = archive.open_embedded_archive(name)
                for module in pyz.toc:
                    code = pyz.extract(module)
                    self.python_modules += 1
                    if isinstance(code, CodeType):
                        self.check_code(code, f"{path.name}:{module}")
                    elif isinstance(code, bytes):
                        self.check_bytes(code, f"{path.name}:{module}")
            elif entry[-1] in ("s", "m", "M"):
                data = archive.extract(name)
                code = marshal.loads(data)
                if isinstance(code, CodeType):
                    self.check_code(code, f"{path.name}:{name}")


def main() -> int:
    """Validate supplied release artifacts and return a nonzero status on leakage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--private-prefix", action="append", required=True)
    parser.add_argument("--python-cli", type=Path)
    args = parser.parse_args()
    checker = PathChecker(args.private_prefix)
    for artifact in args.artifacts:
        if not artifact.exists():
            parser.error(f"missing release artifact: {artifact}")
        files = artifact.rglob("*") if artifact.is_dir() else (artifact,)
        for path in files:
            if path.is_file():
                checker.check_file(path)
    if args.python_cli:
        checker.check_python_cli(args.python_cli)
    if checker.failures:
        for label in sorted(checker.failures):
            print(f"Private build path in {label}")
        print(f"FAIL: {len(checker.failures)} affected files or Python modules; checked {checker.files} files and {checker.python_modules} frozen modules")
        return 1
    print(f"PASS: {checker.files} files and {checker.python_modules} frozen Python modules contain no private build prefixes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# TYLOINFER_PATH_CHECKER_END
'@
}

function Assert-ReleasePaths {
    param(
        [Parameter(Mandatory)][hashtable] $Context,
        [Parameter(Mandatory)][string[]] $Paths,
        [switch] $PythonArchive
    )

    $arguments = @('-c', (Get-ReleasePathCheckerSource)) + $Paths
    foreach ($mapping in @(Get-ReleasePathMappings $Context)) {
        $arguments += @("--private-prefix", $mapping.Source)
    }
    if ($PythonArchive) { $arguments += @("--python-cli", $Context.CliArtifact) }
    Invoke-Checked $Context.Python $arguments
}

function Initialize-ReleaseDirectories {
    param([Parameter(Mandatory)][hashtable] $Context)

    foreach ($path in @($Context.SidecarDirectory, $Context.ResourceDirectory, $Context.WindowsRuntimeDirectory)) {
        Preserve-ReleaseDirectory $path $Context
    }
    $directories = @(
        $Context.PyInstallerBuildDirectory,
        $Context.SpecDirectory,
        $Context.NativeBuildDirectory,
        $Context.SidecarDirectory,
        $Context.ResourceDirectory,
        $Context.WindowsRuntimeDirectory,
        $Context.OutputDirectory
    )
    New-Item -ItemType Directory -Force -Path $directories | Out-Null
    $temporaryDirectory = Join-Path $Context.BuildDirectory "tmp"
    New-Item -ItemType Directory -Force -Path $temporaryDirectory | Out-Null
    $env:TEMP = $temporaryDirectory
    $env:TMP = $temporaryDirectory
    $env:PYINSTALLER_CONFIG_DIR = Join-Path $Context.BuildDirectory "pyinstaller-cache"
}

function Stage-LicenseNotices {
    param([Parameter(Mandatory)][hashtable] $Context)

    $licenseDirectory = Join-Path $Context.ResourceDirectory "licenses"
    Initialize-EmptyDirectory $licenseDirectory
    Copy-Item -LiteralPath (Join-Path $Context.ProjectDirectory "LICENSE") `
        -Destination (Join-Path $licenseDirectory "MFQ-Apache-2.0.txt") -Force
    Copy-Item -LiteralPath (Join-Path $Context.ProjectDirectory "NOTICE") `
        -Destination (Join-Path $licenseDirectory "NOTICE.txt") -Force
    Get-ChildItem -LiteralPath (Join-Path $Context.ProjectDirectory "LICENSES") `
        -File -Filter "*.txt" |
        ForEach-Object {
            Copy-Item -LiteralPath $_.FullName -Destination $licenseDirectory -Force
        }
}

function Assert-ReleaseEnvironmentAvailable {
    param([Parameter(Mandatory)][hashtable] $Context)

    $python = Join-Path $Context.VenvDirectory "Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        return
    }
    $handle = $null
    try {
        $handle = [System.IO.File]::Open(
            $python, [System.IO.FileMode]::Open,
            [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
    }
    catch [System.IO.IOException], [System.UnauthorizedAccessException] {
        Fail "release Python environment is in use or not writable; choose an unused -ReleaseVenv: $($Context.VenvDirectory)"
    }
    finally {
        if ($null -ne $handle) { $handle.Dispose() }
    }
}

function Initialize-PythonEnvironment {
    param(
        [Parameter(Mandatory)][hashtable] $Context,
        [Parameter(Mandatory)][string] $Version
    )

    Assert-ReleaseEnvironmentAvailable $Context
    if (-not (Test-Path -LiteralPath (Join-Path $Context.VenvDirectory 'Scripts\python.exe') -PathType Leaf)) {
        Initialize-EmptyDirectory $Context.VenvDirectory
        Invoke-Checked $Context.Tools.Uv @('venv', '--python', $Version, $Context.VenvDirectory)
    }
    $env:VIRTUAL_ENV = $Context.VenvDirectory
    Add-ToPath (Join-Path $Context.VenvDirectory "Scripts")

    $syncArguments = @(
        "sync", "--locked", "--active", "--no-default-groups", "--no-editable",
        "--group", "release"
    )
    Invoke-Checked $Context.Tools.Uv $syncArguments

    $Context.Python = Join-Path $Context.VenvDirectory "Scripts\python.exe"
    $Context.PyInstaller = Join-Path $Context.VenvDirectory "Scripts\pyinstaller.exe"
    if (-not (Test-Path -LiteralPath $Context.Python -PathType Leaf)) {
        Fail "uv did not create the release Python interpreter"
    }
    if (-not (Test-Path -LiteralPath $Context.PyInstaller -PathType Leaf)) {
        Fail "uv did not install PyInstaller into the release environment"
    }
}

function Repair-MsvcDependencyPrefixEncoding {
    <#
    .SYNOPSIS
    Repair legacy CMake include-prefix decoding before a fresh Ninja cache is built.
    #>
    param([Parameter(Mandatory)][hashtable] $Context)

    $cmakeFiles = Join-Path $Context.NativeBuildDirectory 'CMakeFiles'
    if (-not (Test-Path -LiteralPath $cmakeFiles -PathType Container)) { return }
    $files = @((Join-Path $cmakeFiles 'rules.ninja'))
    foreach ($version in Get-ChildItem -LiteralPath $cmakeFiles -Directory) {
        if ($version.Name -notmatch '^\d+\.\d+') { continue }
        $files += Join-Path $version.FullName 'CMakeCCompiler.cmake'
        $files += Join-Path $version.FullName 'CMakeCXXCompiler.cmake'
    }
    $ansi = [Text.Encoding]::GetEncoding(
        [Text.Encoding]::GetEncoding(0).CodePage,
        [Text.EncoderExceptionFallback]::new(), [Text.DecoderExceptionFallback]::new()
    )
    $utf8 = [Text.UTF8Encoding]::new($false, $true)
    $pattern = '(?m)^(?:msvc_deps_prefix = |set\(CMAKE_(?:C|CXX)_CL_SHOWINCLUDES_PREFIX ")(?<prefix>[^"\r\n]+)'
    $changes = @()
    foreach ($file in $files) {
        if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { continue }
        $original = [IO.File]::ReadAllText($file, $utf8)
        $updated = $original
        $matches = [regex]::Matches($original, $pattern)
        for ($index = $matches.Count - 1; $index -ge 0; $index--) {
            $prefix = $matches[$index].Groups['prefix']
            try { $corrected = $utf8.GetString($ansi.GetBytes($prefix.Value)) }
            catch [Text.EncoderFallbackException] { continue }
            catch [Text.DecoderFallbackException] { continue }
            if ($corrected -cne $prefix.Value) {
                $updated = $updated.Remove($prefix.Index, $prefix.Length).Insert($prefix.Index, $corrected)
            }
        }
        if ($updated -cne $original) {
            $changes += @{ Path = $file; Contents = $updated }
        }
    }
    if ($changes.Count -eq 0) { return }
    if (Test-Path -LiteralPath (Join-Path $Context.NativeBuildDirectory '.ninja_deps') -PathType Leaf) {
        Fail 'This native cache contains incorrectly decoded MSVC include prefixes and may have missing header dependencies. Choose an unused -NativeBuildDirectory; the existing cache was preserved.'
    }
    foreach ($change in $changes) {
        $backup = "$($change.Path).before-prefix-fix-$($Context.BuildId)"
        if (Test-Path -LiteralPath $backup) { Fail "include-prefix backup already exists: $backup" }
        Copy-Item -LiteralPath $change.Path -Destination $backup -ErrorAction Stop
        [IO.File]::WriteAllText($change.Path, $change.Contents, $utf8)
    }
}

function Build-NativeSidecar {
    param(
        [Parameter(Mandatory)][hashtable] $Context,
        [Parameter(Mandatory)][string] $Architectures,
        [Parameter(Mandatory)][int] $ParallelJobs
    )

    # mfq-runtime uses the native tensor backend and must remain independent of
    # Python and LibTorch. Explicitly disable the optional A/B reference target
    # so a reused CMake cache cannot pull those dependencies back into a release.
    $nativeFlags = @(Get-ReleaseNativeFlags $Context)
    $cudaFlags = ($nativeFlags | ForEach-Object { "--compiler-options=$_" }) -join ' '
    Invoke-Checked $Context.Tools.CMake @(
        "-S", (Join-Path $Context.ProjectDirectory "cpp_runtime"),
        "-B", $Context.NativeBuildDirectory,
        "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_C_FLAGS=$env:CFLAGS",
        "-DCMAKE_CXX_FLAGS=$env:CXXFLAGS",
        "-DCMAKE_CUDA_FLAGS=$env:CUDAFLAGS $cudaFlags",
        "-DCMAKE_CUDA_COMPILER=$([System.IO.Path]::Combine($Context.CudaRoot, 'bin', 'nvcc.exe'))",
        "-DCMAKE_CUDA_ARCHITECTURES=$Architectures",
        "-DTYLOINFER_CUDA_ARCHITECTURES=$Architectures",
        "-DTYLOINFER_BUILD_RUNTIME_COMMUNICATION=ON",
        "-DTYLOINFER_BUILD_METAL_RUNTIME=OFF",
        "-DTYLOINFER_ENABLE_MFQ_COMPAT=$(if ($env:TYLOINFER_ENABLE_MFQ_COMPAT) { $env:TYLOINFER_ENABLE_MFQ_COMPAT } else { 'OFF' })",
        "-DTYLOINFER_BUILD_TORCH_REFERENCE_RUNTIME=OFF"
    )
    Repair-MsvcDependencyPrefixEncoding $Context
    Invoke-Checked $Context.Tools.CMake @(
        "--build", $Context.NativeBuildDirectory,
        "--target", "mfq-runtime", "mfq-diagnostics",
        "--config", "Release",
        "--parallel", $ParallelJobs
    )

    $Context.NativeOutput = Get-ChildItem -LiteralPath $Context.NativeBuildDirectory `
        -Recurse -File -Filter "mfq-runtime.exe" |
        Sort-Object -Property LastWriteTimeUtc -Descending |
        Select-Object -First 1
    if ($null -eq $Context.NativeOutput) {
        Fail "native build did not create mfq-runtime.exe"
    }
    Copy-Item -LiteralPath $Context.NativeOutput.FullName -Destination $Context.NativeArtifact -Force
    $diagnostics = Get-ChildItem -LiteralPath $Context.NativeBuildDirectory `
        -Recurse -File -Filter "mfq-diagnostics.exe" |
        Sort-Object -Property LastWriteTimeUtc -Descending |
        Select-Object -First 1
    if ($null -eq $diagnostics) {
        Fail "native build did not create mfq-diagnostics.exe"
    }
    Copy-Item -LiteralPath $diagnostics.FullName -Destination `
        (Join-Path $Context.SidecarDirectory "mfq-diagnostics-$($Context.TargetTriple).exe") -Force
}

function Stage-WindowsRuntime {
    param([Parameter(Mandatory)][hashtable] $Context)

    Initialize-EmptyDirectory $Context.WindowsRuntimeDirectory
    Copy-RuntimeDlls $Context.NativeOutput.DirectoryName $Context.WindowsRuntimeDirectory
    $cudaRuntimeDirectory = Find-CudaRuntimeDirectory $Context.CudaRoot
    foreach ($pattern in @("cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll")) {
        Copy-RuntimeDlls $cudaRuntimeDirectory $Context.WindowsRuntimeDirectory @($pattern)
        if (-not (Get-ChildItem -LiteralPath $Context.WindowsRuntimeDirectory `
                -File -Filter $pattern -ErrorAction SilentlyContinue)) {
            Fail "the CUDA Toolkit did not provide the native runtime DLL matching $pattern"
        }
    }
    Copy-VisualCppRuntimeDlls $Context.WindowsRuntimeDirectory
    if (-not (Get-ChildItem -LiteralPath $Context.WindowsRuntimeDirectory `
            -File -Filter "*.dll" -ErrorAction SilentlyContinue)) {
        Fail "no native runtime DLLs were staged"
    }
}

function Build-PythonCli {
    param([Parameter(Mandatory)][hashtable] $Context)

    $arguments = @(
        "--noconfirm", "--onedir",
        "--name", $Context.CliName,
        "--paths", $Context.ProjectDirectory,
        "--collect-data", "tyloinfer"
    )
    foreach ($package in @(
            "av", "fastapi", "httpx", "PIL", "pydantic", "pypdf",
            "huggingface_hub", "modelscope_hub", "uvicorn", "websockets", "tokenizers")) {
        $arguments += @("--collect-all", $package)
    }
    foreach ($module in @(
            "torch", "transformers", "safetensors",
            "scipy", "tiktoken",
            "tyloinfer.runtime",
            "tyloinfer.runtime.minicpmo45", "tyloinfer.runtime.minicpmo45_realtime",
            "minicpmo_utils", "stepaudio2_minicpmo", "torchaudio", "onnxruntime",
            "s3tokenizer", "hyperpyyaml", "librosa")) {
        $arguments += @("--exclude-module", $module)
    }
    $arguments += @(
        "--copy-metadata", "huggingface-hub",
        "--copy-metadata", "modelscope-hub"
    )
    $arguments += @(
        "--distpath", $Context.ResourceDirectory,
        "--workpath", $Context.PyInstallerBuildDirectory,
        "--specpath", $Context.SpecDirectory,
        (Join-Path $Context.ProjectDirectory "tyloinfer\cli.py")
    )
    Invoke-Checked $Context.PyInstaller $arguments

    if (-not (Test-Path -LiteralPath $Context.CliArtifact -PathType Leaf)) {
        Fail "PyInstaller did not create $($Context.CliArtifact)"
    }
    Invoke-Checked $Context.CliArtifact @("--version")
    Invoke-Checked $Context.CliArtifact @("serve", "--help")
    Write-Host "built $($Context.CliArtifact)"
    Write-Host "built $($Context.NativeArtifact)"
}

function Build-NsisInstaller {
    param([Parameter(Mandatory)][hashtable] $Context)


    $cargoTargetDirectory = if ([string]::IsNullOrWhiteSpace($env:CARGO_TARGET_DIR)) {
        Join-Path $Context.StudioDirectory "src-tauri\target"
    } else {
        Resolve-BuildPath $env:CARGO_TARGET_DIR $Context.StudioDirectory
    }
    $releaseDirectory = Join-Path $cargoTargetDirectory "$($Context.TargetTriple)\release"
    Push-Location $Context.StudioDirectory
    try {
        Invoke-Checked $Context.Tools.Npm @("ci")
        Invoke-Checked $Context.Tools.Npm @("run", "check")
        $env:CI = "true"
        Invoke-Checked $Context.Tools.Npm @(
            "run", "tauri", "--", "build",
            "--config", "src-tauri/tauri.release-windows.conf.json",
            "--target", $Context.TargetTriple,
            "--no-bundle"
        )
        Assert-ReleasePaths $Context @(
            (Join-Path $releaseDirectory "tyloinfer.exe"),
            $Context.SidecarDirectory, $Context.ResourceDirectory,
            $Context.WindowsRuntimeDirectory, (Join-Path $Context.StudioDirectory "dist")
        ) -PythonArchive
        Invoke-Checked $Context.Tools.Npm @(
            "run", "tauri", "--", "bundle",
            "--config", "src-tauri/tauri.release-windows.conf.json",
            "--target", $Context.TargetTriple,
            "--bundles", "nsis"
        )
    }
    finally {
        Pop-Location
    }

    $nsisDirectory = Join-Path $cargoTargetDirectory `
        "$($Context.TargetTriple)\release\bundle\nsis"
    $installer = Get-ChildItem -LiteralPath $nsisDirectory -File -Filter "*.exe" `
        -ErrorAction SilentlyContinue |
        Sort-Object -Property LastWriteTimeUtc -Descending |
        Select-Object -First 1
    if ($null -eq $installer) {
        Fail "Tauri did not create an NSIS installer"
    }

    $releaseInstaller = Join-Path $Context.OutputDirectory $installer.Name
    Assert-ReleasePaths $Context @($installer.FullName)
    Copy-Item -LiteralPath $installer.FullName -Destination $releaseInstaller -Force
    Write-Host "copied $releaseInstaller"
}

function Get-PortableBuilderSource {
    <# .SYNOPSIS
    Return the Python resource packer and embedded NSIS launcher template.
    #>
    return @'
"""Package the Windows release as a standalone executable or ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import zipfile
import uuid


NSIS_TEMPLATE = r'''Unicode true
RequestExecutionLevel user
SilentInstall silent
AutoCloseWindow true
InstallDir "$TEMP\TyloInferStudio"
SetCompressor /SOLID zlib
Name "TyloInfer"
Caption "TyloInfer"
OutFile "@OUTPUT@"
Icon "@ICON@"
VIProductVersion "@VERSION@.0"
VIAddVersionKey /LANG=1033 "ProductName" "TyloInfer"
VIAddVersionKey /LANG=1033 "FileDescription" "TyloInfer"
VIAddVersionKey /LANG=1033 "FileVersion" "@VERSION@"
VIAddVersionKey /LANG=1033 "LegalCopyright" "MFQ contributors"

!include "FileFunc.nsh"
!include "LogicLib.nsh"

Var Parameters
Var CacheRoot
Var CachePath
Var Stage
Var Mutex
Var WaitPid
Var Previous
Var StageSuffix

Function ReadCacheDirectoryOption
  Push $0
  Push $1
  Push $2
  Push $3
  Push $4
  Push $5
  Push $6
  Push $7
  System::Call 'kernel32::GetCommandLineW() p .r0'
  System::Call 'shell32::CommandLineToArgvW(p r0, *i .r1) p .r2'
  StrCmp $2 0 cache_option_done
  StrCpy $3 1
cache_option_loop:
  IntCmp $3 $1 cache_option_free 0 cache_option_free
  IntOp $4 $3 * ${NSIS_PTR_SIZE}
  IntOp $4 $2 + $4
  System::Call '*$4(p .r5)'
  System::Call 'kernel32::lstrcpynW(w .r6, p r5, i ${NSIS_MAX_STRLEN})'
  StrCpy $7 $6 11
  StrCmp $7 "/CACHE_DIR=" 0 cache_option_next
  StrCpy $CacheRoot $6 "" 11
cache_option_next:
  IntOp $3 $3 + 1
  Goto cache_option_loop
cache_option_free:
  System::Call 'kernel32::LocalFree(p r2)'
cache_option_done:
  Pop $7
  Pop $6
  Pop $5
  Pop $4
  Pop $3
  Pop $2
  Pop $1
  Pop $0
FunctionEnd

Section
  ${GetParameters} $Parameters
  ReadEnvStr $CacheRoot "TYLOINFER_STUDIO_RUNTIME_CACHE"
  ${If} $CacheRoot == ""
    StrCpy $CacheRoot "$LOCALAPPDATA\com.tylogi.tyloinfer\runtime"
  ${EndIf}
  Call ReadCacheDirectoryOption
  ClearErrors
  CreateDirectory "$CacheRoot"
  IfErrors failed
  GetFullPathName $0 "$CacheRoot"
  IfErrors failed
  StrCpy $CacheRoot "$0"
  StrCmp $CacheRoot "" failed
  StrCpy $CachePath "$CacheRoot\@PAYLOAD_ID@"
  System::Call 'kernel32::GetCurrentProcessId() i .r0'
  System::Call 'kernel32::GetTickCount() i .r1'
  StrCpy $StageSuffix "$0-$1"
  StrCpy $Stage "$CachePath.partial-$StageSuffix"
  StrCpy $Previous "$CachePath.previous-$StageSuffix"

  System::Call 'kernel32::CreateMutexW(p 0, i 0, w "Local\TyloInferStudio-@PAYLOAD_ID@") p .r0'
  StrCpy $Mutex $0
  ${If} $Mutex == 0
    Goto failed
  ${EndIf}
  System::Call 'kernel32::WaitForSingleObject(p $Mutex, i 120000) i .r0'
  ${If} $0 != 0
  ${AndIf} $0 != 128
    Goto failed
  ${EndIf}

  IfFileExists "$CachePath\.ready" 0 extract
@CACHE_CHECKS@
  Goto cached

extract:
  SetOutPath "$CacheRoot"
  IfFileExists "$Stage\*.*" failed
  ClearErrors
@PAYLOAD_FILES@
  FileOpen $0 "$Stage\.ready" w
  IfErrors failed
  FileWrite $0 "@PAYLOAD_ID@"
  FileClose $0
  IfErrors failed
  SetOutPath "$CacheRoot"
  IfFileExists "$CachePath\*.*" 0 publish
  IfFileExists "$Previous\*.*" failed
  ClearErrors
  Rename "$CachePath" "$Previous"
  IfErrors failed
publish:
  StrCpy $0 50
publish_retry:
  ClearErrors
  Rename "$Stage" "$CachePath"
  IfErrors 0 cached
  IntOp $0 $0 - 1
  IntCmp $0 0 restore
  ; Windows scanners can briefly lock freshly extracted executables.
  Sleep 100
  Goto publish_retry
restore:
  Rename "$Previous" "$CachePath"
  Goto failed

cached:
  System::Call 'kernel32::ReleaseMutex(p $Mutex)'
  System::Call 'kernel32::CloseHandle(p $Mutex)'
  StrCpy $Mutex 0
  ClearErrors
  ${GetOptions} $Parameters "/EXTRACT_ONLY" $0
  ${IfNot} ${Errors}
    Goto done
  ${EndIf}

  ClearErrors
  ${GetOptions} $Parameters "/WAITPID=" $WaitPid
  ${IfNot} ${Errors}
    System::Call 'kernel32::OpenProcess(i 0x100000, i 0, i $WaitPid) p .r0'
    ${If} $0 != 0
      StrCpy $1 $0
      System::Call 'kernel32::WaitForSingleObject(p r1, i 30000) i .r0'
      System::Call 'kernel32::CloseHandle(p r1)'
      ${If} $0 != 0
        Goto failed
      ${EndIf}
    ${EndIf}
  ${EndIf}

  System::Call 'kernel32::SetEnvironmentVariableW(w "TYLOINFER_STUDIO_LAUNCHER", w "$EXEPATH")'
  SetOutPath "$CachePath"
  ClearErrors
  Exec '"$CachePath\tyloinfer.exe"'
  IfErrors failed

done:
  SetErrorLevel 0
  Quit

failed:
  ${If} $Mutex != 0
    System::Call 'kernel32::ReleaseMutex(p $Mutex)'
    System::Call 'kernel32::CloseHandle(p $Mutex)'
  ${EndIf}
  MessageBox MB_OK|MB_ICONSTOP "TyloInfer could not start. Check available disk space and write access to:$\r$\n$CacheRoot" /SD IDOK
  SetErrorLevel 1
SectionEnd
'''


def payload(project: Path, executable: Path) -> dict[str, Path]:
    """Resolve and validate the resources declared by the Windows bundle."""
    tauri = project / "TyloInferStudio/src-tauri"
    bundle = json.loads((tauri / "tauri.release-windows.conf.json").read_text())["bundle"]
    files: dict[str, Path] = {}
    names: set[str] = set()

    def add(source: Path, destination: str) -> None:
        path = PurePosixPath(destination.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts or ":" in str(path):
            raise ValueError(f"Unsafe archive path: {destination}")
        name = str(path)
        if name.casefold() in names:
            raise ValueError(f"Duplicate archive path: {name}")
        if not source.is_file():
            raise FileNotFoundError(source)
        files[name] = source
        names.add(name.casefold())

    add(executable, "tyloinfer.exe")
    for source, destination in bundle["resources"].items():
        root = (tauri / source).resolve()
        if root.is_dir():
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    add(path, str(PurePosixPath(destination) / path.relative_to(root).as_posix()))
        else:
            add(root, destination)
    for source in bundle["externalBin"]:
        path = (tauri / f"{source}-x86_64-pc-windows-msvc.exe").resolve()
        add(path, Path(source).name + ".exe")
    for required in ("mfq-runtime.exe", "mfq-diagnostics.exe", "tyloinfer-cli/tyloinfer-cli.exe", "web-ui/index.html"):
        if required not in files:
            raise FileNotFoundError(f"Missing portable resource: {required}")
    return files


def build_archive(project: Path, executable: Path, output: Path) -> Path:
    """Publish a complete portable resource ZIP after validating its payload."""
    files = payload(project, executable)
    config = json.loads((project / "TyloInferStudio/src-tauri/tauri.conf.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"TyloInfer_{config['version']}_x64-portable.zip"
    partial = archive.with_name(archive.name + ".partial-" + uuid.uuid4().hex)
    total = sum(path.stat().st_size for path in files.values())
    completed = 0
    next_progress = 256 * 1024 * 1024
    print(f"Packing {len(files)} files ({total / 1024**3:.2f} GiB)", flush=True)
    try:
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as package:
            for name, path in sorted(files.items()):
                package.write(path, "TyloInfer/" + name)
                completed += path.stat().st_size
                if completed >= next_progress:
                    print(f"Packed {completed / 1024**3:.2f} / {total / 1024**3:.2f} GiB", flush=True)
                    next_progress = completed + 256 * 1024 * 1024
        partial.replace(archive)
    finally:
        partial.unlink(missing_ok=True)
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    print(f"Built {archive} ({archive.stat().st_size / 1024**2:.1f} MiB)", flush=True)
    return archive


def nsis_string(value: str) -> str:
    """Escape a literal path or filename for an NSIS quoted string."""
    return value.replace("$", "$$").replace('"', '$\\"').replace("\r", "$\\r").replace("\n", "$\\n")


def build_executable(project: Path, executable: Path, output: Path, makensis: Path, build_directory: Path | None = None) -> Path:
    """Build the launcher, ZIP and checksums from validated bundle resources."""
    files = payload(project, executable)
    version = json.loads((project / "TyloInferStudio/src-tauri/tauri.conf.json").read_text())["version"]
    if any(character in version for character in ('/', '\\', ':')) or '..' in version:
        raise ValueError('Unsafe portable version path')
    digest = hashlib.sha256()
    instructions = []
    directory = None
    for name, path in sorted(files.items()):
        digest.update(name.encode() + b"\0")
        with path.open("rb") as stream:
            digest.update(hashlib.file_digest(stream, "sha256").digest())
        target = PurePosixPath(name)
        parent = str(target.parent).replace("/", "\\")
        if parent != directory:
            instructions.append(f'  SetOutPath "$Stage\\{nsis_string(parent)}"')
            directory = parent
        instructions += [f'  File "/oname={nsis_string(target.name)}" "{nsis_string(str(path.resolve()))}"',
                         '  IfErrors failed']
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"TyloInfer_{version}_x64.exe"
    partial = destination.with_name(destination.name + ".partial-" + uuid.uuid4().hex)
    build_directory = build_directory or project / "packaging/build/standalone" / uuid.uuid4().hex
    script = build_directory / "launcher.nsi"
    script.parent.mkdir(parents=True, exist_ok=True)
    template = NSIS_TEMPLATE
    for key, value in {
        "OUTPUT": nsis_string(str(partial.resolve())), "VERSION": version,
        "ICON": nsis_string(str(project / "TyloInferStudio/src-tauri/icons/icon.ico")),
        "PAYLOAD_ID": f"{version}-{digest.hexdigest()[:16]}",
        "PAYLOAD_FILES": "\n".join(instructions),
        "CACHE_CHECKS": "\n".join(
            f'  IfFileExists "$CachePath\\{nsis_string(name.replace(chr(47), chr(92)))}" 0 extract'
            for name in sorted(files)
        ),
    }.items():
        template = template.replace(f"@{key}@", value)
    script.write_text(template, encoding="utf-8")
    print(f"Packing {len(files)} files into a standalone executable", flush=True)
    archive = output / f"TyloInfer_{version}_x64-portable.zip"
    archive_partial = archive.with_name(archive.name + '.partial-' + uuid.uuid4().hex)
    try:
        subprocess.run([str(makensis), "-INPUTCHARSET", "UTF8", "-V2", str(script)], check=True)
        with partial.open('rb') as stream:
            checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
        with zipfile.ZipFile(archive_partial, 'w', compression=zipfile.ZIP_STORED) as package:
            package.write(partial, 'TyloInfer.exe')
        with archive_partial.open('rb') as stream:
            zip_checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
        partial.replace(destination)
        archive_partial.replace(archive)
    finally:
        partial.unlink(missing_ok=True)
        archive_partial.unlink(missing_ok=True)
    destination.with_suffix(".exe.sha256").write_text(f"{checksum}  {destination.name}\n", encoding="utf-8")
    print(f"Built {destination} ({destination.stat().st_size / 1024**2:.1f} MiB)", flush=True)
    archive.with_suffix(".zip.sha256").write_text(f"{zip_checksum}  {archive.name}\n", encoding="utf-8")
    print(f"Built {archive} (one executable)", flush=True)
    return destination


def main() -> None:
    """Build a standalone launcher or archive from an explicit project root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-directory", type=Path, required=True)
    parser.add_argument("--build-directory", type=Path)
    parser.add_argument("--studio-executable", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--makensis", type=Path, help="Build a single-file app using this NSIS compiler")
    args = parser.parse_args()
    project = args.project_directory.resolve()
    if args.makensis:
        build_executable(project, args.studio_executable, args.output_directory, args.makensis, args.build_directory)
    else:
        build_archive(project, args.studio_executable, args.output_directory)


if __name__ == "__main__":
    main()

'@
}

function Invoke-PortablePackage {
    <# .SYNOPSIS
    Build a portable ZIP or launcher without rebuilding Studio.
    #>
    param(
        [Parameter(Mandatory)][string] $Python,
        [Parameter(Mandatory)][string] $ProjectDirectory,
        [Parameter(Mandatory)][string] $Executable,
        [Parameter(Mandatory)][string] $OutputDirectory,
        [string] $MakeNsis,
        [string] $BuildDirectory
    )

    $arguments = @('-c', (Get-PortableBuilderSource),
        '--project-directory', $ProjectDirectory,
        '--studio-executable', $Executable,
        '--output-directory', $OutputDirectory)
    if ($MakeNsis) { $arguments += @('--makensis', $MakeNsis) }
    if ($BuildDirectory) { $arguments += @('--build-directory', $BuildDirectory) }
    Invoke-Checked $Python $arguments
}

function Build-PortableArchive {
    param([Parameter(Mandatory)][hashtable] $Context)

    if (-not (Test-Path -LiteralPath $Context.Python -PathType Leaf)) {
        Fail 'portable packaging requires a prepared Python environment; specify -ReleaseVenv from a previous release build'
    }
    & $Context.Python -c 'from PyInstaller.archive.readers import CArchiveReader' 2>$null
    if ($LASTEXITCODE -ne 0) {
        Fail 'portable packaging requires PyInstaller for frozen archive scanning; specify -ReleaseVenv containing PyInstaller'
    }

    $cargoTargetDirectory = if ([string]::IsNullOrWhiteSpace($env:CARGO_TARGET_DIR)) {
        Join-Path $Context.StudioDirectory "src-tauri\target"
    } else {
        Resolve-BuildPath $env:CARGO_TARGET_DIR $Context.StudioDirectory
    }
    $executable = Join-Path $cargoTargetDirectory "$($Context.TargetTriple)\release\tyloinfer.exe"
    $makensis = Join-Path $cargoTargetDirectory ".tauri\NSIS\makensis.exe"
    if (-not (Test-Path -LiteralPath $makensis -PathType Leaf)) {
        $makensis = Require-Command "makensis"
    }
    Assert-ReleasePaths $Context @(
        $executable, $Context.SidecarDirectory, $Context.ResourceDirectory,
        $Context.WindowsRuntimeDirectory, (Join-Path $Context.StudioDirectory "dist")
    ) -PythonArchive
    $portableArguments = @{
        Python = $Context.Python
        ProjectDirectory = $Context.ProjectDirectory
        Executable = $executable
        OutputDirectory = $Context.OutputDirectory
        MakeNsis = $makensis
        BuildDirectory = Join-Path $Context.BuildDirectory 'standalone'
    }
    Invoke-PortablePackage @portableArguments
}

if ($MyInvocation.InvocationName -ne '.') {
    if (-not [System.OperatingSystem]::IsWindows() -or -not [Environment]::Is64BitOperatingSystem) {
        Fail "this release target must be built on 64-bit Windows"
    }

    if ($Clean) {
        Fail '-Clean is not supported; builds preserve previous files. Remove reviewed files manually.'
    }

    $PythonVersion = if ([string]::IsNullOrWhiteSpace($PythonVersion)) { "3.12" } else { $PythonVersion }
    $CudaArchitectures = if ([string]::IsNullOrWhiteSpace($CudaArchitectures)) { "80-real;86-real;89" } else { $CudaArchitectures }
    $Jobs = Get-ReleaseJobs $Jobs

    $venvDirectory = if ([string]::IsNullOrWhiteSpace($ReleaseVenv)) {
        ''
    } else {
        Resolve-BuildPath $ReleaseVenv $PSScriptRoot
    }
    $releaseOutputDirectory = if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
        Join-Path $PSScriptRoot "dist"
    } else {
        Resolve-BuildPath $OutputDirectory $PSScriptRoot
    }

    $context = New-BuildContext $venvDirectory $releaseOutputDirectory -NativeBuildDirectory $NativeBuildDirectory
    if ($PortableOnly) {
        $context.Python = if ($ReleaseVenv) {
            Join-Path $context.VenvDirectory 'Scripts\python.exe'
        } else {
            Require-Command 'python'
        }
        Build-PortableArchive $context
        exit 0
    }
    Initialize-BuildEnvironment $context
    Assert-ReleaseEnvironmentAvailable $context
    Initialize-ReleaseDirectories $context
    Stage-LicenseNotices $context

    Push-Location $context.ProjectDirectory
    try {
        Initialize-PythonEnvironment $context $PythonVersion
        Build-NativeSidecar $context $CudaArchitectures $Jobs
        Stage-WindowsRuntime $context
        Build-PythonCli $context
        Build-NsisInstaller $context
        Build-PortableArchive $context
    }
    finally {
        Pop-Location
    }
}
