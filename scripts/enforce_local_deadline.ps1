param([Parameter(Mandatory=$true)][string]$Session)
$ErrorActionPreference = 'Stop'
$sessionPath = [System.IO.Path]::GetFullPath($Session)
$policyPath = Join-Path $sessionPath 'runtime_deadline.json'
while (-not (Test-Path -LiteralPath (Join-Path $sessionPath 'pipeline_completed.json'))) {
    $policy = Get-Content -LiteralPath $policyPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $seconds = [double]$policy.stop_at - [DateTimeOffset]::UtcNow.ToUnixTimeSeconds() - 15
    if ($seconds -le 0) {
        # Only this session's supervisor/worker processes, including Python's
        # Windows launcher children. Other kernels and training sessions stay intact.
        $processes = @(Get-CimInstance Win32_Process -Filter "name='python.exe'" | Where-Object {
            $_.CommandLine -and $_.CommandLine.Contains($sessionPath) -and
            ($_.CommandLine.Contains('scripts\run_local_pipeline.py') -or $_.CommandLine.Contains('-m spacelab.worker'))
        })
        foreach ($process in $processes) {
            & taskkill /PID $process.ProcessId /T /F 2>$null | Out-Null
        }
        $record = @{ stopped_at = [DateTimeOffset]::Now.ToString('o'); reason = 'User requested maximum elapsed time'; processes = @($processes | ForEach-Object { $_.ProcessId }) }
        $record | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $sessionPath 'deadline_reached.json') -Encoding UTF8
        break
    }
    Start-Sleep -Milliseconds ([int][Math]::Min(5000, [Math]::Max(100, $seconds*1000)))
}
