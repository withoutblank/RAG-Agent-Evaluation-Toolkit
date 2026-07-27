[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string] $Owner,

    [Parameter(Mandatory)]
    [string] $Repo,

    [Parameter()]
    [string] $PythonCommand = "python",

    [Parameter()]
    [switch] $ConfirmSampleResultsTruthful,

    [Parameter()]
    [switch] $ConfirmScreenshotsSanitized
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "github/common.ps1")

$projectRoot = Get-ProjectRoot -ScriptDirectory $PSScriptRoot
Push-Location $projectRoot
try {
    Assert-PublicReleaseAudit `
        -Owner $Owner `
        -Repo $Repo `
        -PythonCommand $PythonCommand `
        -ConfirmSampleResultsTruthful:$ConfirmSampleResultsTruthful `
        -ConfirmScreenshotsSanitized:$ConfirmScreenshotsSanitized
    Write-Host "Public-release audit passed. No repository settings were changed."
}
finally {
    Pop-Location
}
