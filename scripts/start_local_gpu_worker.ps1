param(
    [switch]$CheckOnly,
    [switch]$ExecuteTraining,
    [switch]$Once,
    [string]$JobId,
    [int]$PollSeconds = 15
)

$ErrorActionPreference = 'Stop'

if ($CheckOnly -eq $ExecuteTraining) {
    throw 'Exactly one of -CheckOnly or -ExecuteTraining is required.'
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'BLACKHOLE virtual environment is missing. Complete project setup first.'
}

$mode = if ($CheckOnly) { '--check-only' } else { '--execute-training' }
$localRoot = Join-Path $env:LOCALAPPDATA 'BLACKHOLE_GPU_WORKER'
$arguments = @(
    (Join-Path $PSScriptRoot 'local_gpu_worker.py'),
    $mode,
    '--poll-seconds', $PollSeconds,
    '--local-root', $localRoot
)
if ($Once) {
    $arguments += '--once'
}
if ($JobId) {
    if (-not $Once) {
        throw '-JobId requires -Once so the command is bound to one authorized job.'
    }
    $arguments += @('--job-id', $JobId)
}
& $python $arguments
exit $LASTEXITCODE
