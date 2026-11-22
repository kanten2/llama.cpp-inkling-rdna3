param([Parameter(Mandatory)][ValidateSet('Chat','Bench')][string]$Mode)
$ErrorActionPreference = 'Stop'
$bin = Join-Path (Split-Path $PSScriptRoot) 'runtime/bin'
$server = Join-Path $bin 'llama-server.exe'
$process = $null
try {
    if (!$env:INKLING_MODEL -or !(Test-Path -LiteralPath $env:INKLING_MODEL -PathType Leaf)) {
        throw "Drop an Inkling 2 Mini .gguf model onto $($Mode.ToLower()).cmd."
    }
    $model = Get-Item -LiteralPath $env:INKLING_MODEL
    if ($model.Extension -ne '.gguf' -or $model.Name -like 'mmproj*') {
        throw 'Select the text model .gguf. Rename Mainline_Support .gguf_file to .gguf first.'
    }
    Set-Location -LiteralPath $bin
    $env:PATH = "$bin;$env:SystemRoot\System32;$env:SystemRoot"
    foreach ($key in @('HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','GGML_BACKEND_PATH',
                       'GGML_CUDA_DISABLE_GRAPHS','GGML_CUDA_DISABLE_FUSION','GGML_HIP_DISABLE_ROW_CLIP',
                       'LLAMA_INKLING_DISABLE_MIX_REUSE','GGML_HIP_DISABLE_CONV_RESIDUAL',
                       'GGML_HIP_DISABLE_CONCAT_HISTORY','GGML_IQ4_PROFILE_STRIPE','GGML_IQ4_PROFILE_STRIDE',
                       'ROCBLAS_TENSILE_LIBPATH','HIPBLASLT_TENSILE_LIBPATH')) {
        [Environment]::SetEnvironmentVariable($key, $null, 'Process')
    }
    $env:GGML_HIP_IQ4_GATE4 = '1'
    $env:GGML_HIP_DISABLE_SCALE_ADD_FUSION = '1'
    $devices = & $server --list-devices
    if ($LASTEXITCODE) { throw 'Cannot load the runtime. Check the AMD driver and x64 VC++ runtime.' }
    $device = $devices | Select-String '^\s*ROCm(\d+):\s*(AMD Radeon RX 7900 XTX)\s*\(' | Select-Object -First 1
    if (!$device) { throw 'RX 7900 XTX was not found by the bundled ROCm backend.' }
    $env:HIP_VISIBLE_DEVICES = $device.Matches[0].Groups[1].Value
    Write-Host "GPU: $($device.Matches[0].Groups[2].Value)"
    Write-Host "Model: $($model.Name)"

    if ($Mode -eq 'Bench') {
        Write-Host 'Running tg768 (warm-up and one timed run)...'
        $json = & (Join-Path $bin 'llama-bench.exe') -m $model.FullName -dev ROCm0 -sm none -mg 0 -ngl 99 -t 16 `
            -b 2048 -ub 512 -fa auto -d 0 -p 0 -n 768 -r 1 -o json
        if ($LASTEXITCODE) { throw "llama-bench exited with code $LASTEXITCODE." }
        $rows = @(($json -join [Environment]::NewLine) | ConvertFrom-Json | ForEach-Object { $_ })
        if ($rows.Count -ne 1 -or $rows[0].n_prompt -ne 0 -or $rows[0].n_gen -ne 768 -or
            $rows[0].devices -ne 'ROCm0' -or $rows[0].gpu_info -notmatch 'RX 7900 XTX' -or $rows[0].avg_ts -le 0) {
            throw 'The benchmark did not return the requested GPU and workload.'
        }
        Write-Host ('tg768: {0:F2} tok/s' -f $rows[0].avg_ts)
    } else {
        # Pick a free loopback port so an existing chat can keep its own port.
        $listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, 0)
        $listener.Start()
        try { $port = $listener.LocalEndpoint.Port } finally { $listener.Stop() }
        $url = "http://127.0.0.1:$port"
        $arguments = @('-m', ('"' + $model.FullName + '"'), '--alias', 'inkling-2-mini', '--host', '127.0.0.1',
                       '--port', $port, '-c', 4096, '-np', 1, '-ngl', 99, '--device', 'ROCm0', '--jinja', '--offline')
        $process = Start-Process -FilePath $server -WorkingDirectory $bin -ArgumentList $arguments -NoNewWindow -PassThru
        $ready = $false
        $timer = [Diagnostics.Stopwatch]::StartNew()
        while (!$process.HasExited -and $timer.Elapsed.TotalMinutes -lt 5) {
            try {
                $health = Invoke-RestMethod "$url/health" -TimeoutSec 2
                if ($health.status -eq 'ok') { $ready = $true; break }
            } catch [Net.WebException] {}
            Start-Sleep -Milliseconds 400
        }
        if (!$ready -or $process.HasExited) { throw "Server failed to become ready. Check free VRAM and the messages above." }
        Write-Host "Local chat: $url"
        Write-Host 'Press Ctrl+C here to stop the server before benchmarking.'
        Start-Process -FilePath $url
        $process.WaitForExit()
        if ($process.ExitCode) { throw "Server exited with code $($process.ExitCode)." }
    }
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
} finally {
    if ($process) {
        if (!$process.HasExited) { $process.Kill(); $process.WaitForExit() }
        $process.Dispose()
    }
}
