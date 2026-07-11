param(
    [string]$Config = "",
    [string]$Workspace = "",
    [string]$OutputDir = "",
    [int]$PreloadCapacity = 5,
    [int]$TopK = 5
)

$ErrorActionPreference = "Stop"

$repo = Resolve-Path (Join-Path $PSScriptRoot "..")
$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = Join-Path $repo ".akashic-workspace\mcp\feed-mcp\.venv\Scripts\python.exe"
}
if (-not (Test-Path $python)) {
    throw "Cannot find Python executable."
}

$argsList = @(
    (Join-Path $repo "_bench\tool_governance_benchmark.py"),
    "--preload-capacity", "$PreloadCapacity",
    "--top-k", "$TopK"
)

if ($Config.Trim()) {
    $argsList += @("--config", $Config)
}
if ($Workspace.Trim()) {
    $argsList += @("--workspace", $Workspace)
}
if ($OutputDir.Trim()) {
    $argsList += @("--output-dir", $OutputDir)
}

& $python @argsList
