param(
    [string]$HostName = "eruin@192.168.0.3",
    [string]$RemoteDir = "/home/eruin/kingoGPT",
    [switch]$IncludeState
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Archive = Join-Path $env:TEMP "kingogpt-agent.tar.gz"
$Excludes = @(
    "--exclude=.git",
    "--exclude=.venv",
    "--exclude=__pycache__",
    "--exclude=*.pyc",
    "--exclude=.pytest_cache",
    "--exclude=secrets"
)

if (-not $IncludeState) {
    $Excludes += "--exclude=state"
}

if (Test-Path $Archive) {
    Remove-Item -LiteralPath $Archive -Force
}

Push-Location $RepoRoot
try {
    & tar -czf $Archive @Excludes .
}
finally {
    Pop-Location
}

& ssh $HostName "mkdir -p '$RemoteDir'"
& scp $Archive "${HostName}:$RemoteDir/kingogpt-agent.tar.gz"
& ssh $HostName "cd '$RemoteDir' && tar -xzf kingogpt-agent.tar.gz && rm kingogpt-agent.tar.gz && chmod +x deploy/remote_bootstrap.sh"

Write-Host "Synced to $($HostName):$RemoteDir"
Write-Host "Next on server: cd $RemoteDir && ./deploy/remote_bootstrap.sh"
