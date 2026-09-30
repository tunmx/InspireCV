param(
    [string]$BuildDirectory,
    [ValidateSet('Release', 'Debug')][string]$Configuration = 'Release',
    [ValidateSet('Object', 'Static', 'Shared')][string]$LibraryType = 'Object',
    [switch]$RunTests,
    [switch]$Examples,
    [ValidateRange(1, 64)][int]$Jobs = 4,
    [string[]]$CMakeOptions = @()
)

$ErrorActionPreference = 'Stop'
if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
    throw 'This build entry point requires Windows.'
}
if (![Environment]::Is64BitProcess -or $env:VSCMD_ARG_TGT_ARCH -ne 'x64' -or
    !(Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    throw 'Run from an x64 Visual Studio Native Tools environment using 64-bit PowerShell.'
}

# Use the optional CMake/Ninja components from the active Visual Studio
# installation. Toolchain installation and environment setup are external.
if ($env:VSINSTALLDIR) {
    $bundledTools = Join-Path $env:VSINSTALLDIR 'Common7\IDE\CommonExtensions\Microsoft\CMake'
    foreach ($relative in @('CMake\bin', 'Ninja')) {
        $directory = Join-Path $bundledTools $relative
        if (Test-Path $directory) { $env:PATH = "$directory;$env:PATH" }
    }
}
foreach ($tool in @('cmake.exe', 'ctest.exe', 'ninja.exe')) {
    if (!(Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "$tool is required on PATH. Install CMake 3.20 or newer and Ninja."
    }
}
$cmakeVersionOutput = & cmake.exe --version
if ($LASTEXITCODE -ne 0) { throw "Could not read the CMake version: $LASTEXITCODE" }
$versionMatch = [regex]::Match(($cmakeVersionOutput -join "`n"), 'cmake version (\d+\.\d+\.\d+)')
if (!$versionMatch.Success -or [version]$versionMatch.Groups[1].Value -lt [version]'3.20.0') {
    throw 'CMake 3.20 or newer is required.'
}

$repo = Split-Path $PSScriptRoot -Parent
$Configuration = if ($Configuration -eq 'Debug') { 'Debug' } else { 'Release' }
$kind = $LibraryType.ToLowerInvariant()
if (!$BuildDirectory) {
    $BuildDirectory = Join-Path $repo "build\windows-x64-$Configuration-$kind"
}
$buildProvider = $null
$buildDrive = $null
$BuildDirectory = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath(
    $BuildDirectory, [ref]$buildProvider, [ref]$buildDrive)
if ($buildProvider.Name -ne 'FileSystem') {
    throw 'BuildDirectory must be a filesystem path.'
}
$installDirectory = Join-Path $BuildDirectory 'install'
$objectLibrary = if ($LibraryType -eq 'Object') { 'ON' } else { 'OFF' }
$sharedLibrary = if ($LibraryType -eq 'Shared') { 'ON' } else { 'OFF' }
$buildTests = if ($RunTests) { 'ON' } else { 'OFF' }
$buildExamples = if ($Examples) { 'ON' } else { 'OFF' }

# Apply the script's named options last so tests, installation and the output
# directory agree with the requested configuration even in a reused cache.
$options = @($CMakeOptions) + @(
    '-S', $repo, '-B', $BuildDirectory, '-G', 'Ninja',
    "-DCMAKE_BUILD_TYPE=$Configuration",
    "-DCMAKE_INSTALL_PREFIX=$installDirectory",
    "-DINSPIRECV_BUILD_OBJECT_LIBS=$objectLibrary",
    "-DINSPIRECV_BUILD_SHARED_LIBS=$sharedLibrary",
    "-DINSPIRECV_BUILD_TESTS=$buildTests",
    "-DBUILD_TESTING=$buildTests",
    "-DINSPIRECV_BUILD_EXAMPLES=$buildExamples",
    '-DINSPIRECV_INSTALL=ON'
)

Write-Output "Configuring InspireCV in $BuildDirectory"
& cmake.exe @options
if ($LASTEXITCODE -ne 0) { throw "CMake configuration failed: $LASTEXITCODE" }

Write-Output 'Building InspireCV'
& cmake.exe --build $BuildDirectory --config $Configuration --parallel $Jobs
if ($LASTEXITCODE -ne 0) { throw "Build failed: $LASTEXITCODE" }

if ($RunTests) {
    & ctest.exe --test-dir $BuildDirectory -C $Configuration --output-on-failure --no-tests=error
    if ($LASTEXITCODE -ne 0) { throw "Tests failed: $LASTEXITCODE" }
}

Write-Output "Installing InspireCV to $installDirectory"
& cmake.exe --install $BuildDirectory --config $Configuration
if ($LASTEXITCODE -ne 0) { throw "Installation failed: $LASTEXITCODE" }
Write-Output "Build completed successfully: $installDirectory"
