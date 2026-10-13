param(
    [Parameter(Mandatory = $true)]
    [string]$JobId
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot '.venv\Scripts\python.exe'
$worker = Join-Path $PSScriptRoot 'local_gpu_worker.py'

if (-not (Test-Path -LiteralPath $python)) {
    throw 'BLACKHOLE virtual environment is missing. Use the project .venv; global Python is not permitted for controlled execution.'
}
if (-not (Test-Path -LiteralPath $worker)) {
    throw 'BLACKHOLE local worker script is missing.'
}

$localRoot = Join-Path $env:LOCALAPPDATA 'BLACKHOLE_GPU_WORKER'

Write-Host 'BLACKHOLE local CUDA preflight...' -ForegroundColor Cyan
& $python $worker --check-only --local-root $localRoot
if ($LASTEXITCODE -ne 0) {
    throw 'BLACKHOLE local CUDA preflight failed.'
}

Write-Host 'Executing one already-authorized BLACKHOLE local job...' -ForegroundColor Cyan
& $python $worker --execute-training --once --job-id $JobId --local-root $localRoot
if ($LASTEXITCODE -ne 0) {
    throw 'BLACKHOLE authorized local job did not complete successfully.'
}

Write-Host 'BLACKHOLE authorized local job complete.' -ForegroundColor Green
