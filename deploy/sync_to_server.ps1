param(
    [string]$HostName = "eruin@192.168.0.3",
    [string]$RemoteDir = "/home/eruin/kingoGPT",
    [switch]$IncludeState
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Archive = Join-Path $env:TEMP "kingogpt-agent.tar.gz"
$Excludes = @(
    "--exclude=__pycache__",
    "--exclude=*.pyc",
    "--exclude=.pytest_cache",
    "--exclude=state/pip_tmp"
)

if (-not $IncludeState) {
    $Excludes += "--exclude=state"
}

$ArchiveItems = @(
    "deploy",
    "internal_agent",
    "scripts",
    "tests",
    "Dockerfile",
    "docker-compose.yml",
    "requirements.txt",
    "kingogpt_api_solver.py",
    "kingogpt_token_capture.py"
)

if ($IncludeState) {
    $ArchiveItems += "state"
}

if (Test-Path $Archive) {
    Remove-Item -LiteralPath $Archive -Force
}

Push-Location $RepoRoot
try {
    & tar -czf $Archive @Excludes @ArchiveItems
}
finally {
    Pop-Location
}

$KnownHosts = Join-Path $RepoRoot "state\ssh_known_hosts"
$SshOptions = @(
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=$KnownHosts"
)

& ssh @SshOptions $HostName "mkdir -p '$RemoteDir'"
& scp @SshOptions $Archive "${HostName}:$RemoteDir/kingogpt-agent.tar.gz"
& ssh @SshOptions $HostName "cd '$RemoteDir' && tar -xzf kingogpt-agent.tar.gz && rm kingogpt-agent.tar.gz && chmod +x deploy/remote_bootstrap.sh"

Write-Host "Synced to $($HostName):$RemoteDir"
Write-Host "Next on server: cd $RemoteDir && ./deploy/remote_bootstrap.sh"
