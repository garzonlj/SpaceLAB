param([string]$Session = 'reports/experiments/local_gpu_20261107')
$ErrorActionPreference = 'Stop'
$repoPath = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$sessionPath = if ([System.IO.Path]::IsPathRooted($Session)) { $Session } else { Join-Path $repoPath $Session }
Write-Host 'Monitor local. Ctrl+C detiene solo este monitor, no el entrenamiento.'
while ($true) {
    $currentPath = Join-Path $sessionPath 'current_action.json'
    if (Test-Path -LiteralPath $currentPath) {
        try {
            $current = Get-Content -LiteralPath $currentPath -Raw -Encoding UTF8 | ConvertFrom-Json
            Write-Host ("`n" + (Get-Date -Format 'HH:mm:ss') + ' | ' + $current.key + ' | ' + $current.action)
            $policyPath = Join-Path $sessionPath 'runtime_deadline.json'
            if (Test-Path -LiteralPath $policyPath) {
                $policy = Get-Content -LiteralPath $policyPath -Raw -Encoding UTF8 | ConvertFrom-Json
                $remaining = [Math]::Max(0, [Math]::Ceiling(([double]$policy.stop_at - [DateTimeOffset]::UtcNow.ToUnixTimeSeconds())/60))
                Write-Host ('Plazo maximo: ' + $policy.deadline + ' | faltan ' + $remaining + ' minutos')
            }
            $budget = Get-Content -LiteralPath (Join-Path $sessionPath 'budget.json') -Raw -Encoding UTF8 | ConvertFrom-Json
            $latest = $budget.jobs | Select-Object -Last 1
            Write-Host ('Estado: ' + $latest.status)
            if ($latest.run) {
                $log = Join-Path $latest.run 'worker.log'
                if (Test-Path -LiteralPath $log) { Get-Content -LiteralPath $log -Encoding UTF8 -Tail 5 }
            }
        } catch { Write-Host 'Actualizando estado...' }
    }
    if (Test-Path -LiteralPath (Join-Path $sessionPath 'pipeline_completed.json')) {
        Write-Host 'Pipeline terminado. Revisar comparison.csv y los resultados finales.'
        break
    }
    Start-Sleep -Seconds 5
}
