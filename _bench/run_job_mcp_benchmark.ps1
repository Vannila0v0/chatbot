param(
    [string]$Workspace = "",
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"

$repo = Resolve-Path (Join-Path $PSScriptRoot "..")
$python = Join-Path $repo ".akashic-workspace\mcp\feed-mcp\.venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    $python = Join-Path $repo ".venv\Scripts\python.exe"
}
if (-not (Test-Path $python)) {
    throw "Cannot find Python executable."
}

$argsList = @(
    (Join-Path $repo "_bench\job_mcp_benchmark.py")
)
if ($Workspace.Trim()) {
    $argsList += @("--workspace", $Workspace)
}
if ($OutputDir.Trim()) {
    $argsList += @("--output-dir", $OutputDir)
}

& $python @argsList
