$ErrorActionPreference = 'Stop'
if ($env:OS -ne 'Windows_NT' -or ![Environment]::Is64BitProcess) {
    throw 'Run Windows CI setup in a 64-bit Windows PowerShell process.'
}

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (!(Test-Path $vswhere)) { throw 'Visual Studio 2022 C++ Build Tools and vswhere are required.' }
$installation = & $vswhere -latest -products '*' -version '[17.0,18.0)' `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if ($LASTEXITCODE -ne 0 -or !$installation) { throw 'Visual Studio 2022 with the x64 C++ tools was not found.' }
$developerCommand = Join-Path $installation 'Common7\Tools\VsDevCmd.bat'
if (!(Test-Path $developerCommand)) { throw 'Visual Studio developer environment script was not found.' }

# Keep compiler setup separate from the build entry point. Only toolchain
# variables are persisted to subsequent Actions steps.
$developerEnvironment = & $env:ComSpec /d /s /c "`"`"$developerCommand`" -no_logo -arch=x64 -host_arch=x64 >nul && set`""
if ($LASTEXITCODE -ne 0) { throw 'Failed to initialize the x64 Visual Studio environment.' }
$toolchainNames = @('PATH', 'INCLUDE', 'LIB', 'LIBPATH', 'VSINSTALLDIR',
    'VCINSTALLDIR', 'VCToolsInstallDir', 'VCToolsVersion', 'VisualStudioVersion',
    'DevEnvDir', 'VSCMD_ARG_TGT_ARCH', 'VSCMD_ARG_HOST_ARCH',
    'WindowsSdkDir', 'WindowsSDKVersion', 'WindowsSDKLibVersion',
    'WindowsSdkBinPath', 'WindowsSdkVerBinPath', 'UniversalCRTSdkDir',
    'UCRTVersion', 'ExtensionSdkDir')
foreach ($line in $developerEnvironment) {
    if ($line -match '^([^=]+)=(.*)$' -and $matches[1] -in $toolchainNames) {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
    }
}
if ($env:VSCMD_ARG_TGT_ARCH -ne 'x64') { throw 'Visual Studio did not select the x64 target.' }

$bundledTools = Join-Path $installation 'Common7\IDE\CommonExtensions\Microsoft\CMake'
foreach ($relative in @('CMake\bin', 'Ninja')) {
    $directory = Join-Path $bundledTools $relative
    if (Test-Path $directory) { $env:PATH = "$directory;$env:PATH" }
}
foreach ($tool in @('cl.exe', 'cmake.exe', 'ctest.exe', 'ninja.exe')) {
    if (!(Get-Command $tool -ErrorAction SilentlyContinue)) { throw "$tool is required for Windows CI." }
}
$cmakeVersion = & cmake.exe --version
if ($LASTEXITCODE -ne 0 -or $cmakeVersion[0] -notmatch '^cmake version (\d+\.\d+\.\d+)') {
    throw 'Could not determine the CMake version.'
}
if ([version]$matches[1] -lt [version]'3.20.0') { throw 'CMake 3.20 or newer is required.' }

if ($env:GITHUB_ENV) {
    foreach ($name in $toolchainNames) {
        $value = [Environment]::GetEnvironmentVariable($name, 'Process')
        if ($null -ne $value) {
            if ($value -match '[\r\n]') { throw "Unexpected newline in environment variable $name" }
            "$name=$value" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
        }
    }
}
Write-Output "Visual Studio: $installation"
Write-Output $cmakeVersion[0]
& ninja.exe --version
if ($LASTEXITCODE -ne 0) { throw 'Ninja is not usable.' }
Write-Output 'Windows x64 build environment ready.'
