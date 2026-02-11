param(
  [string]$ProjectRoot = "..",
  [switch]$IncludeBuild,
  [switch]$RunVerify
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$accelRoot = (Resolve-Path (Join-Path $scriptDir "..")).Path
$project = (Resolve-Path $ProjectRoot).Path

Write-Host "[phase0] accel root: $accelRoot"
Write-Host "[phase0] project root: $project"

Push-Location $accelRoot

python -m accel.cli doctor --project $project --output json
python -m accel.cli index build --project $project --full --output json
python -m accel.cli context --project $project --task "phase0 baseline for harborpilot refactor" --out "$project\.harborpilot\logs\context_pack_phase0.json" --output json

$baselineArgs = @(
  "scripts/collect_harborpilot_phase0_baseline.py",
  "--project", $project,
  "--json-out", "$project\.harborpilot\logs\phase0_baseline_latest.json",
  "--analysis-out", "$project\.harborpilot\logs\analysis_last.json",
  "--md-out", "$project\.harborpilot\docs\reports\phase0_baseline_latest.md"
)

if ($IncludeBuild) {
  $baselineArgs += "--include-build"
}

python @baselineArgs

if ($RunVerify) {
  python -m accel.cli verify --project $project --changed-files "agent-accel/scripts/collect_harborpilot_phase0_baseline.py" --output json
}

Pop-Location
Write-Host "[phase0] complete"
