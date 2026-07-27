[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = "Low")]
param(
    [Parameter(Mandatory)]
    [ValidateSet("feat", "fix", "docs", "test", "chore", "ci")]
    [string] $Type,

    [Parameter(Mandatory)]
    [ValidatePattern("^[a-z0-9]+(?:-[a-z0-9]+)*$")]
    [string] $Slug,

    [Parameter()]
    [ValidateRange(1, 2147483647)]
    [int] $Issue
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "github/common.ps1")

$projectRoot = Get-ProjectRoot -ScriptDirectory $PSScriptRoot
Push-Location $projectRoot
try {
    Assert-CommandAvailable -Name "git"
    Assert-CleanWorkingTree
    Assert-OnMainBranch

    $branchSuffix = if ($PSBoundParameters.ContainsKey("Issue")) {
        "$Issue-$Slug"
    }
    else {
        $Slug
    }
    $branchName = "$Type/$branchSuffix"

    $localBranches = Invoke-CheckedCommand `
        -Command "git" `
        -Arguments @("branch", "--format=%(refname:short)") `
        -CaptureOutput
    if (($localBranches -split "`r?`n") -ccontains $branchName) {
        throw "Local branch '$branchName' already exists."
    }

    $remoteBranch = @(& git ls-remote --exit-code --heads origin $branchName 2>&1)
    $remoteBranchExit = $LASTEXITCODE
    if ($remoteBranchExit -eq 0) {
        throw "Remote branch '$branchName' already exists."
    }
    if ($remoteBranchExit -ne 2) {
        throw "Unable to determine whether remote branch '$branchName' exists."
    }

    if (-not $PSCmdlet.ShouldProcess($branchName, "fast-forward main and create task branch")) {
        return
    }

    Invoke-CheckedCommand `
        -Command "git" `
        -Arguments @("pull", "--ff-only", "origin", "main")
    Invoke-CheckedCommand `
        -Command "git" `
        -Arguments @("switch", "-c", $branchName)

    Write-Host "Created branch '$branchName'."
    Write-Host "Next: make focused changes, run python scripts/check_all.py, then use scripts/open_pr.ps1."
}
finally {
    Pop-Location
}
