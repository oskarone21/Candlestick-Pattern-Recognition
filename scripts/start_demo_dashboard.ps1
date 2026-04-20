param(
    [string]$RunName = "demo_balanced_v2_20260420",
    [int]$Port = 3000,
    [switch]$Rebuild
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$dashboardRoot = Join-Path $repoRoot "web-dashboard"
$buildDir = Join-Path $dashboardRoot "build"

Write-Host "Preparing dashboard snapshot for run '$RunName'..."
python (Join-Path $repoRoot "scripts/build_results_dashboard.py") `
    --config (Join-Path $repoRoot "configs/config.yaml") `
    --run-name $RunName `
    --output-root (Join-Path $dashboardRoot ".generated") `
    --data-only

Push-Location $dashboardRoot
try {
    if ($Rebuild -or -not (Test-Path $buildDir)) {
        Write-Host "Building dashboard bundle..."
        npm.cmd run build
    }

    $env:DASHBOARD_RUN_NAME = $RunName
    $env:PORT = "$Port"
    Write-Host "Starting demo dashboard at http://localhost:$Port"
    npm.cmd run start
}
finally {
    Pop-Location
}
