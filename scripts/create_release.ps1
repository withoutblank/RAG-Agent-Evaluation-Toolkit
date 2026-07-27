[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = "High")]
param(
    [Parameter(Mandatory)]
    [ValidatePattern("^v?(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-[0-9A-Za-z.-]+)?$")]
    [string] $Version,

    [Parameter(Mandatory)]
    [string] $Owner,

    [Parameter()]
    [string] $Repo = "rag-agent-eval-toolkit",

    [Parameter()]
    [string] $PythonCommand = "python",

    [Parameter()]
    [switch] $ConfirmRelease,

    [Parameter()]
    [switch] $ConfirmSampleResultsTruthful,

    [Parameter()]
    [switch] $ConfirmScreenshotsSanitized
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "github/common.ps1")

if (-not $ConfirmRelease) {
    throw "Creating and publishing a release requires the explicit -ConfirmRelease flag."
}
Assert-SafeRepositoryName -Owner $Owner -Repo $Repo

$tag = if ($Version.StartsWith("v", [System.StringComparison]::Ordinal)) {
    $Version
}
else {
    "v$Version"
}
$semanticVersion = $tag.Substring(1)

$projectRoot = Get-ProjectRoot -ScriptDirectory $PSScriptRoot
Push-Location $projectRoot
try {
    Assert-GitHubPrerequisites -ExpectedOwner $Owner -PythonCommand $PythonCommand
    Assert-RemoteRepositoryIdentity -Owner $Owner -Repo $Repo

    $releaseLookup = @(
        & gh release view $tag --repo "$Owner/$Repo" --json tagName,url 2>&1 |
            ForEach-Object { "$_" }
    )
    $releaseLookupExit = $LASTEXITCODE
    if ($releaseLookupExit -eq 0) {
        Write-Host "Release '$tag' already exists. Nothing was changed."
        return
    }
    $releaseLookupMessage = $releaseLookup -join "`n"
    if ($releaseLookupMessage -notmatch "(?i)release not found|http 404|not found") {
        throw "Unable to determine whether release '$tag' exists."
    }

    $visibility = Get-GitHubRepositoryVisibility -Owner $Owner -Repo $Repo
    if ($visibility -ne "PUBLIC") {
        throw "Release creation requires a public repository; current visibility is '$visibility'."
    }

    Assert-PublicReleaseAudit `
        -Owner $Owner `
        -Repo $Repo `
        -PythonCommand $PythonCommand `
        -ConfirmSampleResultsTruthful:$ConfirmSampleResultsTruthful `
        -ConfirmScreenshotsSanitized:$ConfirmScreenshotsSanitized

    $changelogPath = "CHANGELOG.md"
    if (-not (Test-Path -LiteralPath $changelogPath -PathType Leaf)) {
        throw "CHANGELOG.md must be updated before release."
    }
    $changelog = Get-Content -Raw -Encoding UTF8 $changelogPath
    $escapedVersion = [regex]::Escape($semanticVersion)
    if ($changelog -notmatch "(?m)^## \[$escapedVersion\](?:\s|$)") {
        throw "CHANGELOG.md must contain a release heading for [$semanticVersion]."
    }

    $null = @(& git show-ref --verify --quiet "refs/tags/$tag" 2>&1)
    $localTagExit = $LASTEXITCODE
    if ($localTagExit -eq 0) {
        throw "Local tag '$tag' already exists; refusing to replace it."
    }
    if ($localTagExit -ne 1) {
        throw "Unable to inspect local tag '$tag'."
    }

    $null = @(& git ls-remote --exit-code --tags origin "refs/tags/$tag" 2>&1)
    $remoteTagExit = $LASTEXITCODE
    if ($remoteTagExit -eq 0) {
        throw "Remote tag '$tag' already exists; refusing to replace it."
    }
    if ($remoteTagExit -ne 2) {
        throw "Unable to inspect remote tag '$tag'."
    }

    if (-not $PSCmdlet.ShouldProcess(
            "$Owner/$Repo $tag",
            "create annotated tag, push it, and create a GitHub release"
        )) {
        return
    }

    Invoke-CheckedCommand `
        -Command "git" `
        -Arguments @(
            "tag", "-a", $tag,
            "-m", "$tag — Portfolio MVP"
        )
    Invoke-CheckedCommand `
        -Command "git" `
        -Arguments @("push", "origin", "refs/tags/$tag")
    Invoke-CheckedCommand `
        -Command "gh" `
        -Arguments @(
            "release", "create", $tag,
            "--repo", "$Owner/$Repo",
            "--generate-notes",
            "--title", "$tag — Portfolio MVP"
        )

    $releaseJson = Invoke-CheckedCommand `
        -Command "gh" `
        -Arguments @(
            "release", "view", $tag,
            "--repo", "$Owner/$Repo",
            "--json", "isDraft,tagName,url"
        ) `
        -CaptureOutput
    $release = $releaseJson | ConvertFrom-Json
    if ($release.tagName -cne $tag -or $release.isDraft) {
        throw "Release verification failed for '$tag'."
    }
    Write-Host "Created release '$tag': $($release.url)"
}
finally {
    Pop-Location
}
