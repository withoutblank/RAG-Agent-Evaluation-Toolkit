[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = "Low")]
param(
    [Parameter(Mandatory)]
    [string] $Owner,

    [Parameter()]
    [string] $Repo = "rag-agent-eval-toolkit",

    [Parameter()]
    [string] $PythonCommand = "python"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "github/common.ps1")

Assert-SafeRepositoryName -Owner $Owner -Repo $Repo

$projectRoot = Get-ProjectRoot -ScriptDirectory $PSScriptRoot
Push-Location $projectRoot
try {
    Assert-GitHubPrerequisites -ExpectedOwner $Owner -PythonCommand $PythonCommand
    Assert-RemoteRepositoryIdentity -Owner $Owner -Repo $Repo
    Assert-CleanWorkingTree
    Assert-OnMainBranch

    if (-not $PSCmdlet.ShouldProcess("main", "fast-forward from origin and run weekly checks")) {
        return
    }

    Invoke-CheckedCommand `
        -Command "git" `
        -Arguments @("pull", "--ff-only", "origin", "main")
    Invoke-LocalQualityChecks -PythonCommand $PythonCommand
    Invoke-CheckedCommand `
        -Command "gh" `
        -Arguments @("pr", "list", "--repo", "$Owner/$Repo")
    Invoke-CheckedCommand `
        -Command "gh" `
        -Arguments @("issue", "list", "--repo", "$Owner/$Repo")
    Invoke-CheckedCommand `
        -Command "gh" `
        -Arguments @("run", "list", "--repo", "$Owner/$Repo", "--limit", "10")
}
finally {
    Pop-Location
}
