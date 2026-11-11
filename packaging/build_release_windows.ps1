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
    [string] $NativeBuildDirectory = $env:TYLOINFER_RELEASE_NATIVE_BUILD_DIR,
    [string] $VisualStudioPath = $env:TYLOINFER_RELEASE_VS_PATH
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
    <# .SYNOPSIS
    Honor a selected x64 developer environment or initialize the requested VS installation.
    #>
    param([string] $InstallationPath = $env:TYLOINFER_RELEASE_VS_PATH)

    if ([string]::IsNullOrWhiteSpace($InstallationPath) -and (Get-Command cl -ErrorAction SilentlyContinue)) {
        if (-not $env:INCLUDE -or -not $env:LIB -or
                ($env:VSCMD_ARG_TGT_ARCH -and $env:VSCMD_ARG_TGT_ARCH -notin @('x64', 'amd64'))) {
            Fail 'The selected compiler requires an x64 Visual Studio developer environment with INCLUDE and LIB configured. Use an x64 Developer PowerShell or specify -VisualStudioPath.'
        }
        return
    }
    $installDirectory = if ($InstallationPath) { $InstallationPath } else { $env:VSINSTALLDIR }
    if ([string]::IsNullOrWhiteSpace($installDirectory)) {
        $vswhere = Get-Command vswhere.exe -ErrorAction SilentlyContinue
        if (-not $vswhere) {
            $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
            if (-not (Test-Path -LiteralPath $vswhere -PathType Leaf)) {
                Fail 'Visual Studio C++ tools are required. Use an x64 Developer PowerShell or specify -VisualStudioPath.'
            }
        }
        $installDirectory = & $vswhere -latest -products * `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
            -property installationPath | Select-Object -First 1
    }
    if ([string]::IsNullOrWhiteSpace($installDirectory)) {
        Fail 'Visual Studio C++ x64 tools were not found; specify -VisualStudioPath for an installation compatible with the selected CUDA Toolkit.'
    }

    $vsDevCmd = Join-Path $installDirectory.Trim() "Common7\Tools\VsDevCmd.bat"
    if (-not (Test-Path -LiteralPath $vsDevCmd -PathType Leaf)) {
        Fail "VsDevCmd.bat is missing from $installDirectory"
    }

    $originalPath = $env:PATH
    $importedPath = $null
    $environmentLines = & cmd.exe /d /s /c "call `"$vsDevCmd`" -no_logo -arch=x64 -host_arch=x64 >nul && set"
    if ($LASTEXITCODE -ne 0) {
        Fail "failed to initialize the Visual Studio x64 environment"
    }
    foreach ($line in $environmentLines) {
        $separator = $line.IndexOf('=')
        if ($separator -le 0) {
            continue
        }
        $name = $line.Substring(0, $separator)
        if ($name -ieq 'PATH') { $importedPath = $line.Substring($separator + 1); continue }
        if ($name -ine 'VSLANG') {
            Set-Item -Path "Env:$name" -Value $line.Substring($separator + 1)
        }
    }
    try {
        $env:PATH = $importedPath
        $compiler = Require-Command 'cl'
    } finally { $env:PATH = $originalPath }
    $env:PATH = (@((Split-Path $compiler -Parent)) + @($originalPath -split ';') +
        @($importedPath -split ';') | Where-Object { $_ } | Select-Object -Unique) -join ';'
}

function Find-CudaRoot {
    <# .SYNOPSIS
    Select one SDK through CUDACXX, PATH, CUDA_HOME, CUDA_PATH, then installed versions.
    #>
    $compiler = $null
    if ($env:CUDACXX) {
        if (Test-Path -LiteralPath $env:CUDACXX -PathType Leaf) {
            $compiler = [IO.Path]::GetFullPath($env:CUDACXX)
        } else {
            $command = Get-Command $env:CUDACXX -ErrorAction SilentlyContinue
            if ($command) { $compiler = $command.Source }
        }
        if (-not $compiler) { Fail 'CUDACXX must name an existing nvcc.exe executable; compiler arguments belong in CUDAFLAGS.' }
    } else {
        $command = Get-Command nvcc -ErrorAction SilentlyContinue
        if ($command) { $compiler = $command.Source }
    }
    if ($compiler) {
        $bin = Split-Path $compiler -Parent
        if ((Split-Path $compiler -Leaf) -ine 'nvcc.exe' -or (Split-Path $bin -Leaf) -ine 'bin') {
            Fail 'The selected CUDA compiler must be nvcc.exe inside a CUDA Toolkit bin directory so matching runtime DLLs can be staged.'
        }
        return Split-Path $bin -Parent
    }
    foreach ($name in @('CUDA_HOME', 'CUDA_PATH')) {
        $configured = [Environment]::GetEnvironmentVariable($name)
        if ($configured) {
            if (-not (Test-Path -LiteralPath (Join-Path $configured 'bin\nvcc.exe') -PathType Leaf)) {
                Fail "$name does not contain bin\nvcc.exe: $configured"
            }
            return [IO.Path]::GetFullPath($configured)
        }
    }

    $cudaParent = Join-Path ${env:ProgramFiles} "NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path -LiteralPath $cudaParent -PathType Container) {
        $cudaDirectory = Get-ChildItem -LiteralPath $cudaParent -Directory |
            Where-Object { $_.Name -match '^v?\d+(\.\d+){1,3}$' } |
            Sort-Object -Property @{ Expression = { [version] $_.Name.TrimStart('v') }; Descending = $true } |
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

    if ($RequestedJobs -lt 0) { Fail '-Jobs must be positive, or 0 for automatic parallelism' }
    if ($RequestedJobs -gt 0) {
        return $RequestedJobs
    }
    if ($env:TYLOINFER_RELEASE_JOBS) {
        $configuredJobs = 0
        if (-not [int]::TryParse($env:TYLOINFER_RELEASE_JOBS, [ref] $configuredJobs) -or $configuredJobs -lt 1) {
            Fail 'TYLOINFER_RELEASE_JOBS must be a positive integer'
        }
        return $configuredJobs
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
        [string] $NativeBuildDirectory = $env:TYLOINFER_RELEASE_NATIVE_BUILD_DIR,
        [switch] $Clean,
        [string] $VisualStudioPath = $env:TYLOINFER_RELEASE_VS_PATH
    )

    $projectDirectory = [System.IO.Path]::GetFullPath((Join-Path $scriptDirectory ".."))
    $resourceDirectory = Join-Path $scriptDirectory "Resources"
    $sidecarDirectory = Join-Path $scriptDirectory "sidecars"
    $targetTriple = "x86_64-pc-windows-msvc"
    $cliName = "tyloinfer-cli"
    $buildId = [Guid]::NewGuid().ToString("N")
    $buildDirectory = Join-Path $scriptDirectory "build\windows-$buildId"
    $nativeBuildCache = if ([string]::IsNullOrWhiteSpace($NativeBuildDirectory)) {
        if ($Clean) { Join-Path $buildDirectory 'runtime' } else { Join-Path $scriptDirectory 'build\runtime' }
    } else {
        Resolve-BuildPath $NativeBuildDirectory $projectDirectory
    }
    $cargoTargetDirectory = if ($env:CARGO_TARGET_DIR) {
        Resolve-BuildPath $env:CARGO_TARGET_DIR (Join-Path $projectDirectory 'TyloInferStudio')
    } elseif ($Clean) { Join-Path $buildDirectory 'cargo-target' } else { $null }
    if ($Clean) {
        foreach ($cache in @($nativeBuildCache, $cargoTargetDirectory)) {
            if ((Test-Path -LiteralPath $cache) -and @(Get-ChildItem -LiteralPath $cache -Force).Count -gt 0) {
                Fail '-Clean requires empty explicitly selected cache directories. Choose fresh -NativeBuildDirectory/CARGO_TARGET_DIR paths or omit those overrides; existing caches were preserved.'
            }
        }
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
        CargoTargetDirectory = $cargoTargetDirectory
        VisualStudioPath = $VisualStudioPath
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

function Initialize-BuildEnvironment {
    param([Parameter(Mandatory)][hashtable] $Context)

    if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) { Add-ToPath (Join-Path ${env:ProgramFiles} 'CMake\bin') }
    if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) { Add-ToPath (Join-Path $env:USERPROFILE '.cargo\bin') }
    if (-not (Get-Command makensis -ErrorAction SilentlyContinue)) {
        Add-ToPath ([System.IO.Path]::Combine(
            [Environment]::GetFolderPath([Environment+SpecialFolder]::ProgramFilesX86), 'NSIS'))
    }
    Import-VisualStudioEnvironment -InstallationPath $Context.VisualStudioPath
    if (-not (Get-Command ninja -ErrorAction SilentlyContinue) -and $env:VSINSTALLDIR) {
        Add-ToPath (Join-Path $env:VSINSTALLDIR 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja')
    }
    if ($Context.CargoTargetDirectory) { $env:CARGO_TARGET_DIR = $Context.CargoTargetDirectory }

    $Context.CudaRoot = Find-CudaRoot
    $env:CUDA_PATH = $Context.CudaRoot
    Add-ToPath (Join-Path $Context.CudaRoot "bin")
    Add-ToPath (Join-Path $Context.CudaRoot "lib\x64")

    $Context.Tools.Uv = Require-Command "uv"
    $Context.Tools.CMake = Require-Command "cmake"
    $null = Require-Command "ninja"
    $null = Require-Command "cargo"
    $Context.Tools.Npm = Require-Command "npm.cmd"
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
    if ($Context.ContainsKey('NativeBuildDirectory') -and $Context.NativeBuildDirectory) {
        $mappings += [pscustomobject]@{
            Source = Resolve-BuildPath $Context.NativeBuildDirectory $Context.ProjectDirectory
            Destination = '/mfq-build/native'
        }
    }
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
        self.ascii_needles: set[bytes] = set()
        self.unicode_prefixes: set[str] = set()
        for prefix in prefixes:
            normalized = prefix.rstrip("/\\").replace("\\", "/")
            for spelling in (normalized, normalized.replace("/", "\\")):
                for escaped in (spelling, spelling.replace("\\", "\\\\")):
                    for encoding in ("utf-8", "utf-16le"):
                        self.needles.add(escaped.encode(encoding))
                        if escaped.isascii():
                            self.ascii_needles.add(escaped.lower().encode(encoding))
                    if not escaped.isascii():
                        self.unicode_prefixes.add(escaped.casefold())
        self.overlap = max(
            [len(needle) for needle in self.needles]
            + [4 * len(prefix) for prefix in self.unicode_prefixes], default=1,
        ) - 1
        self.failures: set[str] = set()
        self.files = 0
        self.python_modules = 0

    def check_bytes(self, data: bytes, label: str) -> None:
        """Record the artifact when its bytes contain a private build prefix."""
        if any(needle in data for needle in self.needles):
            self.failures.add(label)
            return
        if self.ascii_needles:
            lowered = data.lower()
            if any(needle in lowered for needle in self.ascii_needles):
                self.failures.add(label)
                return
        if self.unicode_prefixes:
            for encoding, offset in (("utf-8", 0), ("utf-16le", 0), ("utf-16le", 1)):
                text = data[offset:].decode(encoding, errors="replace").casefold()
                if any(prefix in text for prefix in self.unicode_prefixes):
                    self.failures.add(label)
                    return

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
    try { Invoke-Checked $Context.Tools.CMake @(
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
    ) } catch {
        Fail "Native configuration failed with the selected compiler/CUDA Toolkit. Use a compatible x64 developer environment or -VisualStudioPath, and a fresh -NativeBuildDirectory when changing toolchains. $($_.Exception.Message)"
    }
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


def file_sha256(path: Path) -> bytes:
    """Hash a file in bounded chunks on every supported Python version."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.digest()


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
    digest = file_sha256(archive).hex()
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
        digest.update(file_sha256(path))
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
        checksum = file_sha256(partial).hex()
        with zipfile.ZipFile(archive_partial, 'w', compression=zipfile.ZIP_STORED) as package:
            package.write(partial, 'TyloInfer.exe')
        zip_checksum = file_sha256(archive_partial).hex()
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

function Get-CheckoutBuildLock {
    <# .SYNOPSIS
    Hold an exclusive per-checkout file handle until the caller disposes it.
    #>
    param([string] $ScriptDirectory = $PSScriptRoot)

    $directory = Join-Path ([IO.Path]::GetFullPath($ScriptDirectory)) 'build'
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
    try {
        return [IO.File]::Open((Join-Path $directory '.release.lock'),
            [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    } catch {
        Fail "Cannot acquire the checkout release lock. Another release may be active; wait for it or use a separate checkout. $($_.Exception.Message)"
    }
}

if ($MyInvocation.InvocationName -ne '.') {
    $originalEnvironment = [Environment]::GetEnvironmentVariables('Process')
    $checkoutLock = $null
    try {
    if (-not [System.OperatingSystem]::IsWindows() -or -not [Environment]::Is64BitOperatingSystem) {
        Fail "this release target must be built on 64-bit Windows"
    }

    if ($Clean -and $PortableOnly) { Fail '-Clean cannot be combined with -PortableOnly' }
    $checkoutLock = Get-CheckoutBuildLock

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

    $context = New-BuildContext $venvDirectory $releaseOutputDirectory -NativeBuildDirectory $NativeBuildDirectory `
        -Clean:$Clean -VisualStudioPath $VisualStudioPath
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
    } finally {
        if ($checkoutLock) { $checkoutLock.Dispose() }
        foreach ($name in [Environment]::GetEnvironmentVariables('Process').Keys) {
            if (-not $originalEnvironment.Contains($name)) {
                Remove-Item -LiteralPath "Env:$name"
            }
        }
        foreach ($name in $originalEnvironment.Keys) {
            Set-Item -LiteralPath "Env:$name" -Value ([string] $originalEnvironment[$name])
        }
    }
}
