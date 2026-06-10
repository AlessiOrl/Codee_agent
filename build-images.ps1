param(
    [string]$Platform = "linux/amd64",
    [string]$RegistryUser = "alessiorl",
    [string]$BetaTag = "beta",
    [string]$VersionTag = "v2",
    [switch]$NoPush,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

$Images = @(
    @{
        Name = "codee_bot"
        Context = Join-Path $Root "telegram-bot"
    },
    @{
        Name = "codee_agent"
        Context = Join-Path $Root "agent-api"
    }
)

foreach ($Image in $Images) {
    $ImageName = "$RegistryUser/$($Image.Name)"
    $Command = @(
        "docker", "buildx", "build",
        "--platform", $Platform,
        "-t", "${ImageName}:${BetaTag}",
        "-t", "${ImageName}:${VersionTag}"
    )

    if (-not $NoPush) {
        $Command += "--push"
    }

    $Command += $Image.Context
    Write-Host ($Command -join " ")

    if (-not $DryRun) {
        $Executable = $Command[0]
        $Arguments = $Command[1..($Command.Count - 1)]
        & $Executable @Arguments
    }
}
