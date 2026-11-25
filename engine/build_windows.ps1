# OSSXC Native C++/CUDA Engine Build Script (Windows MSVC + NVCC)
# Usage:
#   powershell -ExecutionPolicy Bypass -File engine\build_windows.ps1
#   powershell -ExecutionPolicy Bypass -File engine\build_windows.ps1 -Arch sm_86
param(
    [string]$Arch = "sm_75"
)

$ErrorActionPreference = "Stop"

$EngineDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$CuSrc     = Join-Path $EngineDir "ossxc_engine.cu"
$OutDll    = Join-Path $EngineDir "ossxc_engine.dll"
$PtxasLog  = Join-Path $EngineDir "ptxas_engine_report.txt"

# 1. Auto-detect Visual Studio vcvars64.bat (via vswhere or common paths)
$VcVars = $null
$VsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $VsWhere) {
    $vsPath = & $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
    if ($vsPath) {
        $cand = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
        if (Test-Path $cand) { $VcVars = $cand }
    }
}
if (-not $VcVars) {
    $VcVars = @(
        "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    ) | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $VcVars) {
    throw "Could not find Visual Studio 2022/2019 vcvars64.bat. Install Visual Studio Build Tools with C++ Desktop Development."
}

# 2. Auto-detect CUDA nvcc.exe (CUDA_PATH, PATH, or default toolkit folders)
$Nvcc = $null
if ($env:CUDA_PATH) {
    $cand = Join-Path $env:CUDA_PATH "bin\nvcc.exe"
    if (Test-Path $cand) { $Nvcc = $cand }
}
if (-not $Nvcc) {
    $cmd = Get-Command nvcc.exe -ErrorAction SilentlyContinue
    if ($cmd) { $Nvcc = $cmd.Source }
}
if (-not $Nvcc) {
    $Nvcc = Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin\nvcc.exe" -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending | Select-Object -First 1 -ExpandProperty FullName
}
if (-not $Nvcc) {
    throw "Could not find nvcc.exe. Install NVIDIA CUDA Toolkit 12.x and ensure CUDA_PATH is set."
}

$ComputeArch = $Arch -replace "^sm_", "compute_"
Write-Host "Using vcvars64: $VcVars"
Write-Host "Using nvcc:     $Nvcc"
Write-Host "Target arch:    $Arch ($ComputeArch)"

$CmdLine = "`"$VcVars`" && `"$Nvcc`" -std=c++17 -O3 -gencode arch=$ComputeArch,code=$Arch --ptxas-options=-v --shared -lcublas -lcudart -Xcompiler `"/MD /O2 /EHsc /arch:AVX2`" -o `"$OutDll`" `"$CuSrc`" 2>&1"
$Output = & cmd.exe /c $CmdLine
$ExitCode = $LASTEXITCODE

$Output | Out-File -FilePath $PtxasLog -Encoding utf8
$Output | ForEach-Object { Write-Host $_ }

if ($ExitCode -ne 0) {
    throw "nvcc compilation failed with exit code $ExitCode"
}

Write-Host "Successfully built: $OutDll"
Write-Host "Saved ptxas resource report: $PtxasLog"
