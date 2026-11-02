# Build the native CUDA server, Studio NSIS installer and single-file portable package. Use Python only for build-time validation.

#requires -Version 7
[CmdletBinding()]
param(
    [string] $BuildPython = $env:TYLOINFER_RELEASE_BUILD_PYTHON,
    [string] $NativePrefix = $env:TYLOINFER_RELEASE_NATIVE_PREFIX,
    [string[]] $RuntimeDirectories = @(),
    [string] $OutputDirectory = $env:TYLOINFER_RELEASE_OUTPUT_DIR,
    [string] $CudaArchitectures = $env:TYLOINFER_RELEASE_CUDA_ARCHITECTURES,
    [int] $Jobs = 0,
    [switch] $PortableOnly,
    [switch] $Clean
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

<#
.SYNOPSIS
Stop the build with a release-specific error message.
#>
function Fail {
    param([Parameter(Mandatory)][string] $Message)

    throw "error: $Message"
}

<#
.SYNOPSIS
Resolve an absolute build input relative to its documented base directory.
#>
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

<#
.SYNOPSIS
Add an existing tool directory to this build process without duplicating entries.
#>
function Add-ToPath {
    param([Parameter(Mandatory)][string] $Directory)

    if (-not (Test-Path -LiteralPath $Directory -PathType Container)) {
        return
    }
    if (($env:PATH -split [System.IO.Path]::PathSeparator) -notcontains $Directory) {
        $env:PATH = "$Directory$([System.IO.Path]::PathSeparator)$env:PATH"
    }
}

<#
.SYNOPSIS
Find a required build tool or fail before producing release artifacts.
#>
function Require-Command {
    param([Parameter(Mandatory)][string] $Name)

    $command = Get-Command $Name -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $command) {
        Fail "$Name is required; install it and make it available to this release script"
    }
    return $command.Source
}

<#
.SYNOPSIS
Run a build command and reject any nonzero exit status.
#>
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

<#
.SYNOPSIS
Import the Visual Studio 2022 x64 toolchain into this build process.
#>
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
        Fail "Visual Studio C++ x64 tools are required"
    }

    $vsDevCmd = Join-Path $installDirectory.Trim() "Common7\Tools\VsDevCmd.bat"
    if (-not (Test-Path -LiteralPath $vsDevCmd -PathType Leaf)) {
        Fail "VsDevCmd.bat is missing from $installDirectory"
    }

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
}

<#
.SYNOPSIS
Find the installed CUDA Toolkit containing the native CUDA compiler.
#>
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

<#
.SYNOPSIS
Locate the CUDA Toolkit runtime DLL directory.
#>
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

<#
.SYNOPSIS
Copy matching redistributable DLLs into the isolated server staging directory.
#>
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

<#
.SYNOPSIS
Stage the official Visual C++ and available OpenMP redistributable libraries.
#>
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

    # Bundle the VS OpenMP library so user machines do not depend on the build environment.
    Get-ChildItem -LiteralPath $x64Directory -Directory |
        Where-Object { $_.Name -match '^Microsoft\.VC\d+\.OpenMP$' } |
        ForEach-Object {
            Copy-RuntimeDlls $_.FullName $DestinationDirectory
        }
}

<#
.SYNOPSIS
Choose bounded native build parallelism based on the requested jobs and host memory.
#>
function Get-ReleaseJobs {
    param([int] $RequestedJobs)

    if ($RequestedJobs -gt 0) {
        return $RequestedJobs
    }
    if ($env:TYLOINFER_RELEASE_JOBS) {
        return [int] $env:TYLOINFER_RELEASE_JOBS
    }

    # Large CUDA translation units consume several GiB each; limit parallelism based on physical memory.
    $memoryJobs = [math]::Floor((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 5GB)
    return [math]::Max(1, [math]::Min([int] $memoryJobs, [Environment]::ProcessorCount))
}

<#
.SYNOPSIS
Creates isolated build paths while keeping the canonical resource layout.
#>
function New-BuildContext {
    param(
        [Parameter(Mandatory)][string] $Output,
        [string] $ScriptDirectory = $PSScriptRoot
    )

    $projectDirectory = [System.IO.Path]::GetFullPath((Join-Path $ScriptDirectory '..'))
    $resourceDirectory = Join-Path $ScriptDirectory 'Resources'
    $buildId = [guid]::NewGuid().ToString('N')
    return @{
        ScriptDirectory = $ScriptDirectory
        PathChecker = Join-Path $ScriptDirectory 'check_release_paths.py'
        ProjectDirectory = $projectDirectory
        StudioDirectory = Join-Path $projectDirectory 'TyloInferStudio'
        NativeBuildDirectory = Join-Path $ScriptDirectory "build/native-server/$buildId"
        ServerStageDirectory = Join-Path $ScriptDirectory "build/stage/$buildId/tyloinfer-server"
        TemporaryDirectory = Join-Path $ScriptDirectory "build/tmp/$buildId"
        ServerDirectory = Join-Path $resourceDirectory 'tyloinfer-server'
        ResourceDirectory = $resourceDirectory
        TargetTriple = 'x86_64-pc-windows-msvc'
        OutputDirectory = $Output
        CudaRoot = $null
        Python = $null
        NativeOutput = $null
        Tools = @{ CMake = $null; Npm = $null }
    }
}

<#
.SYNOPSIS
Locate existing native, CUDA and Studio tools and configure release path remapping.
#>
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

    $Context.Tools.CMake = Require-Command "cmake"
    Add-ToPath (Join-Path $env:VSINSTALLDIR 'Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja')
    $null = Require-Command 'ninja'
    $null = Require-Command 'dumpbin'
    $null = Require-Command "cargo"
    $Context.Tools.Npm = Require-Command "npm.cmd"
    $null = Require-Command "cl"
    $null = Require-Command "nvcc"
    # Let Tauri download the pinned NSIS version when it is not installed locally.
    Initialize-ReleasePathMapping $Context
}

<#
.SYNOPSIS
Enumerate source and home path spellings that must not leak into release binaries.
#>
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
        # Native dependency generators may also use Windows 8.3 short paths.
        $shortPath = [System.Text.StringBuilder]::new(32768)
        $length = [MfqRelease.BuildPaths]::GetShortPathName($mapping.Source, $shortPath, $shortPath.Capacity)
        if ($length -gt 0 -and $length -lt $shortPath.Capacity -and $shortPath.ToString() -ne $mapping.Source) {
            [pscustomobject]@{ Source = $shortPath.ToString(); Destination = $mapping.Destination }
        }
    }
}

<#
.SYNOPSIS
Return MSVC flags that remap diagnostic source paths reproducibly.
#>
function Get-ReleaseNativeFlags {
    param([Parameter(Mandatory)][hashtable] $Context)

    "/experimental:deterministic"
    foreach ($mapping in @(Get-ReleasePathMappings $Context)) {
        foreach ($source in @($mapping.Source, $mapping.Source.Replace('\', '/')) | Select-Object -Unique) {
            '"/pathmap:' + $source + '=' + $mapping.Destination + '"'
        }
    }
}

<#
.SYNOPSIS
Apply Rust and native path remapping while preserving encoded argument boundaries.
#>
function Initialize-ReleasePathMapping {
    param([Parameter(Mandatory)][hashtable] $Context)

    # Encode arguments to preserve argument boundaries for paths containing spaces.
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

<#
.SYNOPSIS
Reject private build paths in the supplied release artifacts.
#>
function Assert-ReleasePaths {
    param(
        [Parameter(Mandatory)][hashtable] $Context,
        [Parameter(Mandatory)][string[]] $Paths
    )

    $arguments = @($Context.PathChecker) + $Paths
    foreach ($mapping in @(Get-ReleasePathMappings $Context)) {
        $arguments += @("--private-prefix", $mapping.Source)
    }
    Invoke-Checked $Context.Python $arguments
}

<#
.SYNOPSIS
Creates new build directories without deleting previous release artifacts.
#>
function Initialize-ReleaseDirectories {
    param([Parameter(Mandatory)][hashtable] $Context)

    New-Item -ItemType Directory -Force -Path @(
        $Context.NativeBuildDirectory, $Context.ServerStageDirectory,
        $Context.ResourceDirectory, $Context.OutputDirectory, $Context.TemporaryDirectory
    ) | Out-Null
    $env:TEMP = $Context.TemporaryDirectory
    $env:TMP = $Context.TemporaryDirectory
    # Default release builds to this worktree's npm cache because the global cache may belong to another installation or be unwritable.
    if ([string]::IsNullOrWhiteSpace($env:npm_config_cache)) {
        $env:npm_config_cache = Join-Path $Context.ScriptDirectory 'build/npm-cache'
    }
}

<#
.SYNOPSIS
Copy project and third-party notices into the mapped license resource directory.
#>
function Stage-LicenseNotices {
    param([Parameter(Mandatory)][hashtable] $Context)

    $licenseDirectory = Join-Path $Context.ResourceDirectory "licenses"
    New-Item -ItemType Directory -Force -Path $licenseDirectory | Out-Null
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

<#
.SYNOPSIS
Selects a build-only Python interpreter without modifying any environment.
#>
function Initialize-BuildPython {
    param([Parameter(Mandatory)][hashtable] $Context, [string] $Executable)

    $Context.Python = if ([string]::IsNullOrWhiteSpace($Executable)) {
        Require-Command 'python'
    } else {
        (Get-Command $Executable -ErrorAction Stop).Source
    }
    Invoke-Checked $Context.Python @('-c', 'import sys; assert sys.version_info >= (3, 11), "Build tools require Python 3.11+"')
}

<#
.SYNOPSIS
Builds the public CUDA server from a fresh cache; no Python runtime is packaged.
#>
function Build-NativeServer {
    param(
        [Parameter(Mandatory)][hashtable] $Context,
        [Parameter(Mandatory)][string] $Architectures,
        [Parameter(Mandatory)][int] $ParallelJobs,
        [string] $Prefix
    )

    $nativeFlags = @(Get-ReleaseNativeFlags $Context)
    $cudaFlags = ($nativeFlags | ForEach-Object { "--compiler-options=$_" }) -join ' '
    $arguments = @(
        '-S', (Join-Path $Context.ProjectDirectory 'cpp_runtime'),
        '-B', $Context.NativeBuildDirectory, '-G', 'Ninja',
        '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_TESTING=OFF',
        "-DCMAKE_C_FLAGS=$env:CFLAGS", "-DCMAKE_CXX_FLAGS=$env:CXXFLAGS",
        "-DCMAKE_CUDA_FLAGS=$env:CUDAFLAGS $cudaFlags",
        "-DCMAKE_CUDA_COMPILER=$([System.IO.Path]::Combine($Context.CudaRoot, 'bin', 'nvcc.exe'))",
        "-DCMAKE_CUDA_ARCHITECTURES=$Architectures",
        "-DTYLOINFER_CUDA_ARCHITECTURES=$Architectures",
        '-DTYLOINFER_BUILD_SERVER=ON', '-DTYLOINFER_BUILD_CUDA_RUNTIME=ON',
        '-DTYLOINFER_BUILD_HIP_RUNTIME=OFF', '-DTYLOINFER_BUILD_METAL_RUNTIME=OFF',
        '-DTYLOINFER_BUILD_RUNTIME_COMMUNICATION=ON',
        "-DTYLOINFER_ENABLE_MFQ_COMPAT=$(if ($env:TYLOINFER_ENABLE_MFQ_COMPAT) { $env:TYLOINFER_ENABLE_MFQ_COMPAT } else { 'OFF' })",
        '-DTYLOINFER_BUILD_TORCH_REFERENCE_RUNTIME=OFF'
    )
    if ($Prefix) { $arguments += "-DCMAKE_PREFIX_PATH=$Prefix" }
    Invoke-Checked $Context.Tools.CMake $arguments
    Invoke-Checked $Context.Tools.CMake @(
        '--build', $Context.NativeBuildDirectory, '--target', 'tyloinfer-server',
        '--config', 'Release', '--parallel', $ParallelJobs
    )
    $Context.NativeOutput = Join-Path $Context.NativeBuildDirectory 'server/tyloinfer-server.exe'
    if (-not (Test-Path -LiteralPath $Context.NativeOutput -PathType Leaf)) {
        Fail 'native build did not create server/tyloinfer-server.exe'
    }
}

<#
.SYNOPSIS
Collects transitive DLLs and verifies the staged server with a system-only PATH.
#>
function Stage-NativeServer {
    param(
        [Parameter(Mandatory)][hashtable] $Context,
        [string[]] $SearchDirectories = @()
    )

    $stage = $Context.ServerStageDirectory
    Copy-Item -LiteralPath $Context.NativeOutput -Destination (Join-Path $stage 'tyloinfer-server.exe')
    Copy-VisualCppRuntimeDlls $stage
    $directories = @((Split-Path -Parent $Context.NativeOutput), $stage) + $SearchDirectories
    $cudaDirectory = Find-CudaRuntimeDirectory $Context.CudaRoot
    $directories += $cudaDirectory
    foreach ($pattern in @('cudart64_*.dll', 'cublas64_*.dll', 'cublasLt64_*.dll')) {
        $matches = @(Get-ChildItem -LiteralPath $cudaDirectory -File -Filter $pattern)
        if ($matches.Count -ne 1) { Fail "expected one CUDA runtime DLL matching $pattern" }
        Copy-Item -LiteralPath $matches[0].FullName -Destination $stage
    }
    foreach ($directory in $directories) {
        if (-not (Test-Path -LiteralPath $directory -PathType Container)) {
            Fail "DLL search directory does not exist: $directory"
        }
    }
    Invoke-Checked $Context.Tools.CMake @(
        "-DSERVER=$stage/tyloinfer-server.exe", "-DDESTINATION=$stage",
        "-DSEARCH_DIRECTORIES=$($directories -join ';')",
        "-DDUMPBIN=$(Require-Command 'dumpbin')",
        '-P', (Join-Path $Context.ScriptDirectory 'collect_windows_runtime.cmake')
    )
    $savedPath = $env:PATH
    try {
        $env:PATH = "$stage;$env:SystemRoot/System32;$env:SystemRoot"
        Invoke-Checked (Join-Path $stage 'tyloinfer-server.exe') @('--version')
        Invoke-Checked (Join-Path $stage 'tyloinfer-server.exe') @('--help')
    } finally { $env:PATH = $savedPath }
    Assert-ReleasePaths $Context @($stage)
}

<#
.SYNOPSIS
Publishes verified resources and preserves the previous directory for manual cleanup.
#>
function Publish-NativeServer {
    param([Parameter(Mandatory)][hashtable] $Context)

    $destination = [IO.Path]::GetFullPath($Context.ServerDirectory)
    $expected = [IO.Path]::GetFullPath((Join-Path $Context.ScriptDirectory 'Resources/tyloinfer-server'))
    $source = [IO.Path]::GetFullPath($Context.ServerStageDirectory)
    $stageRoot = [IO.Path]::GetFullPath((Join-Path $Context.ScriptDirectory 'build/stage')) + [IO.Path]::DirectorySeparatorChar
    if ($destination -ne $expected -or -not $source.StartsWith($stageRoot, [StringComparison]::OrdinalIgnoreCase)) {
        Fail 'refusing to move server resources outside the release staging paths'
    }
    $previous = "$destination.previous-$([guid]::NewGuid().ToString('N'))"
    if (Test-Path -LiteralPath $destination) {
        if ((Get-Item -LiteralPath $destination).Attributes -band [IO.FileAttributes]::ReparsePoint) {
            Fail 'refusing to replace a linked server resource directory'
        }
        Move-Item -LiteralPath $destination -Destination $previous
        Write-Host "preserved previous server resources: $previous"
    }
    try { Move-Item -LiteralPath $source -Destination $destination }
    catch {
        if (Test-Path -LiteralPath $previous) { Move-Item -LiteralPath $previous -Destination $destination }
        throw
    }
}

<#
.SYNOPSIS
Build Studio, validate its mapped resources and publish its NSIS installer.
#>
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
            $Context.ServerDirectory, (Join-Path $Context.ResourceDirectory 'licenses'),
            (Join-Path $Context.StudioDirectory "dist")
        )
        Assert-PortableResources $Context (Join-Path $releaseDirectory 'tyloinfer.exe')
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
    $partial = "$releaseInstaller.partial"
    try {
        Copy-Item -LiteralPath $installer.FullName -Destination $partial -Force
        [IO.File]::Move($partial, $releaseInstaller, $true)
    } finally {
        if (Test-Path -LiteralPath $partial -PathType Leaf) { Remove-Item -LiteralPath $partial }
    }
    $checksum = (Get-FileHash -LiteralPath $releaseInstaller -Algorithm SHA256).Hash.ToLowerInvariant()
    [IO.File]::WriteAllText("$releaseInstaller.sha256", "$checksum  $($installer.Name)`n")
    Write-Host "copied $releaseInstaller"
}

<#
.SYNOPSIS
Validates exactly the resources mapped by the release configuration.
#>
function Assert-PortableResources {
    param([Parameter(Mandatory)][hashtable] $Context, [Parameter(Mandatory)][string] $Executable)
    Invoke-Checked $Context.Python @(
        (Join-Path $Context.ScriptDirectory 'build_portable_windows.py'),
        '--studio-executable', $Executable, '--validate-only'
    )
}

<#
.SYNOPSIS
Validate an existing Studio release and build the standalone portable executable.
#>
function Build-PortableArchive {
    param([Parameter(Mandatory)][hashtable] $Context)

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
        $executable, $Context.ServerDirectory, (Join-Path $Context.ResourceDirectory 'licenses'),
        (Join-Path $Context.StudioDirectory "dist")
    )
    Invoke-Checked $Context.Python @(
        (Join-Path $Context.ScriptDirectory "build_portable_windows.py"),
        "--studio-executable", $executable,
        "--output-directory", $Context.OutputDirectory,
        "--makensis", $makensis
    )
}

if (-not [System.OperatingSystem]::IsWindows() -or -not [Environment]::Is64BitOperatingSystem) {
    Fail "this release target must be built on 64-bit Windows"
}

if ($Clean) {
    Fail '-Clean no longer deletes release trees; choose a new output directory and manually remove obsolete artifacts'
}

$CudaArchitectures = if ([string]::IsNullOrWhiteSpace($CudaArchitectures)) { '80-real;86-real;89' } else { $CudaArchitectures }
$releaseOutputDirectory = if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    Join-Path $PSScriptRoot 'dist'
} else {
    Resolve-BuildPath $OutputDirectory $PSScriptRoot
}
$context = New-BuildContext $releaseOutputDirectory
Initialize-BuildPython $context $BuildPython
if ($PortableOnly) {
    Add-ToPath (Join-Path ${env:ProgramFiles(x86)} 'NSIS')
    Build-PortableArchive $context
    exit 0
}
$Jobs = Get-ReleaseJobs $Jobs
Initialize-BuildEnvironment $context
Initialize-ReleaseDirectories $context

Push-Location $context.ProjectDirectory
try {
    Build-NativeServer $context $CudaArchitectures $Jobs $NativePrefix
    $searchDirectories = @($RuntimeDirectories)
    if ($env:TYLOINFER_RELEASE_RUNTIME_DIRS) { $searchDirectories += $env:TYLOINFER_RELEASE_RUNTIME_DIRS -split ';' }
    foreach ($prefix in ($NativePrefix -split ';' | Where-Object { $_ })) {
        $searchDirectories += Join-Path $prefix 'bin'
    }
    Stage-NativeServer $context $searchDirectories
    Publish-NativeServer $context
    Stage-LicenseNotices $context
    Build-NsisInstaller $context
    Build-PortableArchive $context
} finally { Pop-Location }
