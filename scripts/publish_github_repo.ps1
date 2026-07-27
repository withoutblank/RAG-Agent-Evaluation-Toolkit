[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = "High")]
param(
    [Parameter(Mandatory)]
    [string] $Owner,

    [Parameter()]
    [string] $Repo = "rag-agent-eval-toolkit",

    [Parameter()]
    [string] $PythonCommand = "python",

    [Parameter()]
    [switch] $ConfirmPublic,

    [Parameter()]
    [switch] $ConfirmSampleResultsTruthful,

    [Parameter()]
    [switch] $ConfirmScreenshotsSanitized
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "github/common.ps1")

Assert-SafeRepositoryName -Owner $Owner -Repo $Repo

$projectRoot = Get-ProjectRoot -ScriptDirectory $PSScriptRoot
Push-Location $projectRoot
try {
    Assert-GitHubPrerequisites -ExpectedOwner $Owner -PythonCommand $PythonCommand
    if (-not (Test-GitHubRepositoryExists -FullName "$Owner/$Repo")) {
        throw "GitHub repository '$Owner/$Repo' does not exist."
    }

    $visibility = Get-GitHubRepositoryVisibility -Owner $Owner -Repo $Repo
    if ($visibility -eq "PUBLIC") {
        Write-Host "Repository '$Owner/$Repo' is already public. Nothing was changed."
        return
    }
    if ($visibility -ne "PRIVATE") {
        throw "Expected a private repository, but '$Owner/$Repo' has visibility '$visibility'."
    }
    if (-not $ConfirmPublic) {
        throw "Changing repository visibility requires the explicit -ConfirmPublic flag."
    }

    Assert-PublicReleaseAudit `
        -Owner $Owner `
        -Repo $Repo `
        -PythonCommand $PythonCommand `
        -ConfirmSampleResultsTruthful:$ConfirmSampleResultsTruthful `
        -ConfirmScreenshotsSanitized:$ConfirmScreenshotsSanitized

    if (-not $PSCmdlet.ShouldProcess(
            "$Owner/$Repo",
            "change repository visibility from private to public"
        )) {
        return
    }

    Invoke-CheckedCommand `
        -Command "gh" `
        -Arguments @(
            "repo", "edit", "$Owner/$Repo",
            "--visibility", "public",
            "--accept-visibility-change-consequences"
        )

    $verifiedVisibility = Get-GitHubRepositoryVisibility -Owner $Owner -Repo $Repo
    if ($verifiedVisibility -ne "PUBLIC") {
        throw "Visibility-change command completed, but verification returned '$verifiedVisibility'."
    }
    Write-Host "Repository '$Owner/$Repo' is now public."
}
finally {
    Pop-Location
}
