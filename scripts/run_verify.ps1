param(
  [string]$ProjectRoot = "..",
  [string[]]$ChangedFiles = @(),
  [ValidateSet("json", "text")]
  [string]$Output = "json",
  [switch]$EvidenceRun,
  [switch]$FastLoop
)

$ErrorActionPreference = "Stop"

# Always execute from agent-accel root so relative paths are stable.
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$agentAccelRoot = Resolve-Path (Join-Path $scriptRoot "..")
Set-Location $agentAccelRoot

# Avoid single-quote inline command patterns that may break shell wrappers.
$env:PYTHONPATH = "."

if ($EvidenceRun -and -not $FastLoop) {
  $env:ACCEL_VERIFY_FAIL_FAST = "0"
  $env:ACCEL_VERIFY_CACHE_ENABLED = "0"
} elseif ($FastLoop -and -not $EvidenceRun) {
  $env:ACCEL_VERIFY_FAIL_FAST = "1"
  $env:ACCEL_VERIFY_CACHE_ENABLED = "1"
}

$args = @("-m", "accel.cli", "verify", "--project", $ProjectRoot)
if ($ChangedFiles.Count -gt 0) {
  $args += "--changed-files"
  $args += $ChangedFiles
}
$args += @("--output", $Output)

& python @args
exit $LASTEXITCODE
