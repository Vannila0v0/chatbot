param(
    [string]$Profile = "tool-search-pressure",
    [int]$SyntheticCount = 60,
    [int]$TopK = 5,
    [int]$PreloadCapacity = 5
)

$ErrorActionPreference = "Stop"

$repo = Resolve-Path (Join-Path $PSScriptRoot "..")
$compose = Join-Path $repo "docker\debug\docker-compose.yml"

if (-not (Test-Path $compose)) {
    throw "Cannot find docker compose file: $compose"
}

$env:AKASHIC_DEBUG_PROFILE = $Profile

docker compose -f $compose build akashic-debug
docker compose -f $compose run --rm akashic-debug `
    python _bench/tool_search_pressure.py `
    --config /app/config.example.toml `
    --workspace /sandbox/workspace `
    --output-dir /app/_bench/results `
    --synthetic-count $SyntheticCount `
    --top-k $TopK `
    --preload-capacity $PreloadCapacity
