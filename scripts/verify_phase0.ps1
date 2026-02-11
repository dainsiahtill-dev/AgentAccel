param(
  [string]$ProjectRoot = "."
)

$ErrorActionPreference = "Stop"
$project = (Resolve-Path $ProjectRoot).Path

Write-Host "Running Phase 0 verification for agent-accel at $project"

python -m accel.cli init --project $project --output json | Out-Null
python -m accel.cli doctor --project $project --output json | Out-Null
python -m accel.cli index build --project $project --full --output json | Out-Null

Write-Host "Phase 0 verification passed."
exit 0
