param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Command
)

$ErrorActionPreference = "Stop"

if (-not $Command -or $Command.Count -eq 0) {
    $Command = @("bash")
}

# Match the bash helper behavior: prefer default compose (with GPU reservation)
# when NVIDIA runtime is present, otherwise use the no-GPU override file.
$hasNvidia = docker info --format '{{json .Runtimes}}' 2>$null | Select-String -Pattern '"nvidia"' -Quiet

if ($hasNvidia) {
    Write-Host "NVIDIA runtime detected. Running with GPU configuration."
    docker compose run --rm app @Command
} else {
    Write-Host "No NVIDIA runtime detected. Falling back to CPU override."
    docker compose -f docker-compose.cpu.yml run --rm app @Command
}
