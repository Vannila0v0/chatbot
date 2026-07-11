param(
    [string]$Profile = "tool-selection-runner",
    [int]$SyntheticCount = 60,
    [int]$MaxSteps = 4,
    [int]$MaxTokens = 512,
    [int]$Limit = 0,
    [string]$Model = "",
    [string]$Benchmark = "/app/_bench/tool_selection_benchmark_sample.json",
    [switch]$ConnectMcp
)

$ErrorActionPreference = "Stop"

$repo = Resolve-Path (Join-Path $PSScriptRoot "..")
$compose = Join-Path $repo "docker\debug\docker-compose.yml"

if (-not (Test-Path $compose)) {
    throw "Cannot find docker compose file: $compose"
}

$config = Join-Path $repo "config.toml"
if (-not (Test-Path $config)) {
    throw "Cannot find config.toml. This benchmark needs a real LLM config."
}

$env:AKASHIC_DEBUG_PROFILE = $Profile

$argsList = @(
    "compose", "-f", $compose,
    "run", "--rm", "akashic-debug",
    "python", "_bench/tool_selection_runner.py",
    "--config", "/app/config.toml",
    "--benchmark", $Benchmark,
    "--workspace", "/sandbox/workspace",
    "--output-dir", "/app/_bench/results",
    "--synthetic-count", "$SyntheticCount",
    "--max-steps", "$MaxSteps",
    "--max-tokens", "$MaxTokens"
)

if ($Limit -gt 0) {
    $argsList += @("--limit", "$Limit")
}

if ($Model.Trim()) {
    $argsList += @("--model", $Model)
}

if ($ConnectMcp) {
    $argsList += @("--connect-mcp", "--mcp-config", "/app/.akashic-workspace/mcp_servers.json")
}

docker @argsList
