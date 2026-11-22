param([Parameter(Mandatory)][ValidateSet('Chat','Bench')][string]$Mode)
$ErrorActionPreference = 'Stop'
$bin = Join-Path (Split-Path $PSScriptRoot) 'runtime/bin'
$server = Join-Path $bin 'llama-server.exe'
$process = $null
$language = $env:INKLING_LANG
if (!$language) {
    # Windows PowerShell's own UI language can differ from the user's Windows display language.
    Add-Type -Namespace Inkling -Name WindowsLanguage -MemberDefinition @'
        [System.Runtime.InteropServices.DllImport("kernel32.dll")]
        public static extern ushort GetUserDefaultUILanguage();
'@
    try { $language = [Globalization.CultureInfo]::GetCultureInfo([Inkling.WindowsLanguage]::GetUserDefaultUILanguage()).Name }
    catch [Globalization.CultureNotFoundException] { $language = (Get-UICulture).Name }
}
$language = switch -Regex ($language.Replace('_','-')) {
    '^ja(?:-|$)' { 'ja'; break }
    '^zh-Hans(?:-|$)' { 'zh-Hans'; break }
    '^zh-(Hant|TW|HK|MO)(?:-|$)' { 'zh-Hant'; break }
    '^zh(?:-|$)' { 'zh-Hans'; break }
    default { 'en' }
}
$catalog = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'messages.json') -Encoding UTF8 -Raw | ConvertFrom-Json
$msg = $catalog.$language
$encodingBefore = [Console]::OutputEncoding
$failed = $false
try {
    [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
    if (!$env:INKLING_MODEL -or !(Test-Path -LiteralPath $env:INKLING_MODEL -PathType Leaf)) {
        throw ($msg.drop -f $Mode.ToLowerInvariant())
    }
    $model = Get-Item -LiteralPath $env:INKLING_MODEL
    if ($model.Extension -ne '.gguf' -or $model.Name -like 'mmproj*') {
        throw $msg.select
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
    if (!$env:INKLING_PP_FUSE) { $env:INKLING_PP_FUSE = '1' }
    $devices = & $server --list-devices
    if ($LASTEXITCODE) { throw $msg.runtime }
    $device = $devices | Select-String '^\s*ROCm(\d+):\s*(AMD Radeon RX 7900 XTX)\s*\(' | Select-Object -First 1
    if (!$device) { throw $msg.gpuMissing }
    $env:HIP_VISIBLE_DEVICES = $device.Matches[0].Groups[1].Value
    Write-Host ($msg.gpu -f $device.Matches[0].Groups[2].Value)
    Write-Host ($msg.model -f $model.Name)

    if ($Mode -eq 'Bench') {
        Write-Host $msg.warmup
        $json = & (Join-Path $bin 'llama-bench.exe') -m $model.FullName -dev ROCm0 -sm none -mg 0 -ngl 99 -t 16 `
            -b 2048 -ub 512 -fa auto -d 0 -p 0 -n 768 -r 1 -o json
        if ($LASTEXITCODE) { throw ($msg.benchExit -f $LASTEXITCODE) }
        $rows = @(($json -join [Environment]::NewLine) | ConvertFrom-Json | ForEach-Object { $_ })
        if ($rows.Count -ne 1 -or $rows[0].n_prompt -ne 0 -or $rows[0].n_gen -ne 768 -or
            $rows[0].devices -ne 'ROCm0' -or $rows[0].gpu_info -notmatch 'RX 7900 XTX' -or $rows[0].avg_ts -le 0) {
            throw $msg.mismatch
        }
        Write-Host ($msg.rate -f $rows[0].avg_ts)
    } else {
        # Pick a free loopback port so an existing chat can keep its own port.
        $listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, 0)
        $listener.Start()
        try { $port = $listener.LocalEndpoint.Port } finally { $listener.Stop() }
        $url = "http://127.0.0.1:$port"
        $arguments = @('-m', ('"' + $model.FullName + '"'), '--alias', 'inkling-2-mini', '--host', '127.0.0.1',
                       '--port', $port, '-c', 4096, '-np', 1, '-ngl', 99, '--device', 'ROCm0', '--jinja', '--offline')
        Write-Host $msg.loading
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
        if (!$ready -or $process.HasExited) { throw $msg.notReady }
        Write-Host ($msg.url -f $url)
        Write-Host $msg.stop
        Start-Process -FilePath $url
        $process.WaitForExit()
        if ($process.ExitCode) { throw ($msg.serverExit -f $process.ExitCode) }
    }
} catch {
    $failed = $true
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
} finally {
    if ($process) {
        if (!$process.HasExited) { $process.Kill(); $process.WaitForExit() }
        $process.Dispose()
    }
    if (!$env:INKLING_NO_PAUSE -and ($failed -or $Mode -eq 'Bench')) { Write-Host $msg.pause }
    [Console]::OutputEncoding = $encodingBefore
}
