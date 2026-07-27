[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = "High")]
param(
    [Parameter()]
    [string] $Owner = $env:GITHUB_OWNER,

    [Parameter()]
    [string] $Repo = "rag-agent-eval-toolkit",

    [Parameter()]
    [string] $Description = "A lightweight Python toolkit for building and benchmarking cited RAG agents.",

    [Parameter()]
    [string] $PythonCommand = "python",

    [Parameter()]
    [switch] $ConfirmCreate
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "github/common.ps1")

if (-not $ConfirmCreate) {
    throw "Repository creation requires the explicit -ConfirmCreate flag."
}
Assert-SafeRepositoryName -Owner $Owner -Repo $Repo

$projectRoot = Get-ProjectRoot -ScriptDirectory $PSScriptRoot
Push-Location $projectRoot
try {
    Assert-GitHubPrerequisites -ExpectedOwner $Owner -PythonCommand $PythonCommand

    $fullName = "$Owner/$Repo"
    if (Test-GitHubRepositoryExists -FullName $fullName) {
        throw "GitHub repository '$fullName' already exists. Nothing was changed."
    }

    if (Test-Path -LiteralPath ".git" -PathType Container) {
        $remotes = Invoke-CheckedCommand `
            -Command "git" `
            -Arguments @("remote") `
            -CaptureOutput
        if (-not [string]::IsNullOrWhiteSpace($remotes)) {
            throw "A Git remote already exists. This script will not overwrite or add to existing remotes."
        }
    }

    Invoke-LocalQualityChecks -PythonCommand $PythonCommand

    if (-not $PSCmdlet.ShouldProcess(
            $fullName,
            "initialise Git, create a private GitHub repository, and push main"
        )) {
        return
    }

    if (-not (Test-Path -LiteralPath ".git" -PathType Container)) {
        Invoke-CheckedCommand -Command "git" -Arguments @("init")
        Invoke-CheckedCommand -Command "git" -Arguments @("branch", "-M", "main")
    }
    else {
        $null = @(& git rev-parse --verify HEAD 2>&1)
        $hasHead = $LASTEXITCODE -eq 0
        if ($hasHead -and (Get-CurrentGitBranch) -cne "main") {
            throw "An existing repository must already be on 'main'; branch renaming is refused."
        }
        if (-not $hasHead) {
            Invoke-CheckedCommand -Command "git" -Arguments @("branch", "-M", "main")
        }
    }

    Assert-NoObviousSecrets -IncludeUntracked
    Invoke-CheckedCommand -Command "git" -Arguments @("add", "--all")
    Assert-NoTrackedEnvironmentFile
    Assert-NoObviousSecrets -Cached

    Write-Host "Files staged for the bootstrap commit:"
    Invoke-CheckedCommand -Command "git" -Arguments @("diff", "--cached", "--name-status")

    $staged = Invoke-CheckedCommand `
        -Command "git" `
        -Arguments @("diff", "--cached", "--name-only") `
        -CaptureOutput
    $null = @(& git rev-parse --verify HEAD 2>&1)
    $hasHead = $LASTEXITCODE -eq 0
    if (-not [string]::IsNullOrWhiteSpace($staged)) {
        Invoke-CheckedCommand `
            -Command "git" `
            -Arguments @("commit", "-m", "chore: bootstrap RAG agent evaluation toolkit")
    }
    elseif (-not $hasHead) {
        throw "No files are staged, so an initial commit cannot be created."
    }

    Invoke-CheckedCommand `
        -Command "gh" `
        -Arguments @(
            "repo", "create", $fullName,
            "--private",
            "--source", ".",
            "--remote", "origin",
            "--push",
            "--description", $Description
        )

    & (Join-Path $PSScriptRoot "setup_github_project.ps1") `
        -Owner $Owner `
        -Repo $Repo `
        -Apply `
        -Confirm:$false
    if ($LASTEXITCODE -ne 0) {
        throw "The private repository was created, but project metadata setup failed."
    }

    $visibility = Get-GitHubRepositoryVisibility -Owner $Owner -Repo $Repo
    if ($visibility -ne "PRIVATE") {
        throw "Repository creation completed with unexpected visibility '$visibility'."
    }
    Write-Host "Created and configured private repository '$fullName'."
}
finally {
    Pop-Location
}
