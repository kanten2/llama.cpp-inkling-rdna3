# Run from the VS 2022 x64 developer environment with CMake and Ninja on PATH.
param(
    [Parameter(Mandatory)][string]$DevelopmentSdk,
    [Parameter(Mandatory)][string]$BuildDir,
    [int]$Jobs = 8
)
$ErrorActionPreference = 'Stop'
$src = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path.Replace('\','/')
$dev = (Resolve-Path -LiteralPath $DevelopmentSdk).Path.Replace('\','/')
$llvm = "$dev/lib/llvm/bin"
$env:HIP_PATH = $dev
$env:ROCM_PATH = $dev
$env:HIP_DEVICE_LIB_PATH = "$dev/lib/llvm/amdgcn/bitcode"
$env:PATH = "$llvm;$dev/bin;$env:PATH"
$map = '"-ffile-prefix-map=' + $src + '=." "-ffile-prefix-map=' + $dev + '=rocm-sdk"'
cmake -S $src -B $BuildDir -G Ninja -DCMAKE_BUILD_TYPE=Release `
    "-DCMAKE_C_COMPILER=$llvm/clang.exe" "-DCMAKE_CXX_COMPILER=$llvm/clang++.exe" `
    "-DCMAKE_PREFIX_PATH=$dev" "-DCMAKE_C_FLAGS=$map -Wno-error=incompatible-pointer-types" `
    "-DCMAKE_CXX_FLAGS=$map" -DBUILD_SHARED_LIBS=ON -DGGML_HIP=ON -DGGML_HIP_GRAPHS=ON `
    -DGGML_HIP_NO_VMM=ON -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF -DAMDGPU_TARGETS=gfx1100 `
    -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
if ($LASTEXITCODE) { exit $LASTEXITCODE }
cmake --build $BuildDir --target llama ggml-hip --parallel $Jobs
exit $LASTEXITCODE
