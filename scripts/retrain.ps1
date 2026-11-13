# RecoSphere weekly retrain for Windows Task Scheduler (Phase 7f, the no-Airflow option).
#
# One-time setup (PowerShell, from the repo root):
#   schtasks /Create /TN "RecoSphere retrain" /SC WEEKLY /D MON /ST 03:00 `
#     /TR "powershell -NoProfile -ExecutionPolicy Bypass -File `"$PWD\scripts\retrain.ps1`""
# Run now:   schtasks /Run /TN "RecoSphere retrain"
# Remove:    schtasks /Delete /TN "RecoSphere retrain" /F
#
# What it does: lock (no two runs at once) -> python -m pipeline.run (all steps) -> log file per run ->
# exit 0 = promoted (the API hot-swaps within MODEL_WATCH_SECONDS), 2 = gate said no (production unchanged),
# anything else = failure (Task Scheduler shows "Last Run Result" != 0x0).
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Python = Join-Path $Repo ".reco\Scripts\python.exe"           # your venv
$LogDir = Join-Path $Repo "logs"; New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$Log = Join-Path $LogDir "retrain-$Stamp.log"
$Lock = Join-Path $LogDir "retrain.lock"

if (Test-Path $Lock) {
    $age = (Get-Date) - (Get-Item $Lock).LastWriteTime
    if ($age.TotalHours -lt 6) { "another retrain is running (lock $Lock)" | Tee-Object -FilePath $Log; exit 3 }
    Remove-Item $Lock                                              # stale lock from a crashed run
}
New-Item -ItemType File -Path $Lock | Out-Null
try {
    $env:MLFLOW_DISABLE_AGENT_HINT = "1"
    $ErrorActionPreference = "Continue"     # PowerShell 5.1 turns python's stderr (mlflow logs) into errors under "Stop"
    & $Python -m pipeline.run *>&1 | Tee-Object -FilePath $Log
    $code = $LASTEXITCODE
    switch ($code) {
        0 { "RESULT: new model promoted" | Tee-Object -FilePath $Log -Append }
        2 { "RESULT: gate failed, production unchanged" | Tee-Object -FilePath $Log -Append }
        default { "RESULT: pipeline FAILED (exit $code)" | Tee-Object -FilePath $Log -Append }
    }
    exit $code
}
finally {
    Remove-Item $Lock -ErrorAction SilentlyContinue
}
