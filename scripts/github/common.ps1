Set-StrictMode -Version Latest

$ErrorActionPreference = "Stop"

function Get-ProjectRoot {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string] $ScriptDirectory
    )

    return (Resolve-Path (Join-Path $ScriptDirectory "..")).Path
}

function Assert-SafeRepositoryName {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string] $Owner,

        [Parameter(Mandatory)]
        [string] $Repo
    )

    if ($Owner -notmatch "^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$") {
        throw "Owner must be a valid GitHub user or organisation name."
    }
    if ($Repo -notmatch "^[A-Za-z0-9._-]{1,100}$") {
        throw "Repo must contain only letters, numbers, dots, underscores, or hyphens."
    }
}

function Assert-CommandAvailable {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string] $Name
    )

    if ($null -eq (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found on PATH."
    }
}

function Invoke-CheckedCommand {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string] $Command,

        [Parameter()]
        [string[]] $Arguments = @(),

        [Parameter()]
        [switch] $CaptureOutput
    )

    $output = @(& $Command @Arguments 2>&1 | ForEach-Object { "$_" })
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "Command '$Command' failed with exit code $exitCode."
    }

    if ($CaptureOutput) {
        return ($output -join "`n").Trim()
    }

    foreach ($line in $output) {
        Write-Host $line
    }
}

function Get-AuthenticatedGitHubOwner {
    [CmdletBinding()]
    param()

    $login = Invoke-CheckedCommand `
        -Command "gh" `
        -Arguments @("api", "user", "--jq", ".login") `
        -CaptureOutput
    if ([string]::IsNullOrWhiteSpace($login)) {
        throw "GitHub CLI returned an empty authenticated login."
    }
    return $login.Trim()
}

function Assert-GitHubPrerequisites {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string] $ExpectedOwner,

        [Parameter()]
        [string] $PythonCommand = "python"
    )

    foreach ($command in @("git", "gh", $PythonCommand)) {
        Assert-CommandAvailable -Name $command
    }

    Invoke-CheckedCommand -Command "git" -Arguments @("--version")
    Invoke-CheckedCommand -Command "gh" -Arguments @("--version")
    Invoke-CheckedCommand -Command "gh" -Arguments @("auth", "status")
    Invoke-CheckedCommand -Command $PythonCommand -Arguments @("--version")

    $detectedOwner = Get-AuthenticatedGitHubOwner
    if ($detectedOwner -cne $ExpectedOwner) {
        throw "Authenticated GitHub owner mismatch. Expected '$ExpectedOwner'; detected '$detectedOwner'."
    }
}

function Test-GitHubRepositoryExists {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string] $FullName
    )

    $output = @(& gh repo view $FullName --json nameWithOwner 2>&1 | ForEach-Object { "$_" })
    $exitCode = $LASTEXITCODE
    if ($exitCode -eq 0) {
        return $true
    }

    $message = $output -join "`n"
    if ($message -match "(?i)could not resolve to a repository|http 404|not found") {
        return $false
    }
    throw "Unable to determine whether GitHub repository '$FullName' exists (exit $exitCode)."
}

function Assert-CleanWorkingTree {
    [CmdletBinding()]
    param()

    $status = Invoke-CheckedCommand `
        -Command "git" `
        -Arguments @("status", "--porcelain=v1", "--untracked-files=normal") `
        -CaptureOutput
    if (-not [string]::IsNullOrWhiteSpace($status)) {
        throw "Working tree must be clean before this operation."
    }
}

function Get-CurrentGitBranch {
    [CmdletBinding()]
    param()

    $branch = Invoke-CheckedCommand `
        -Command "git" `
        -Arguments @("branch", "--show-current") `
        -CaptureOutput
    if ([string]::IsNullOrWhiteSpace($branch)) {
        throw "A named Git branch is required; detached HEAD is not supported."
    }
    return $branch.Trim()
}

function Assert-OnMainBranch {
    [CmdletBinding()]
    param()

    $branch = Get-CurrentGitBranch
    if ($branch -cne "main") {
        throw "This operation requires branch 'main'; current branch is '$branch'."
    }
}

function Assert-NotMainBranch {
    [CmdletBinding()]
    param()

    $branch = Get-CurrentGitBranch
    if ($branch -ceq "main") {
        throw "This operation refuses to run on 'main'. Create a task branch first."
    }
    return $branch
}

function Assert-RemoteRepositoryIdentity {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string] $Owner,

        [Parameter(Mandatory)]
        [string] $Repo
    )

    $remote = Invoke-CheckedCommand `
        -Command "git" `
        -Arguments @("remote", "get-url", "origin") `
        -CaptureOutput
    $expectedPath = [regex]::Escape("$Owner/$Repo")
    if ($remote -notmatch "(?i)github\.com[:/]$expectedPath(?:\.git)?/?$") {
        throw "Remote 'origin' does not match expected repository '$Owner/$Repo'."
    }
}

function Assert-NoTrackedEnvironmentFile {
    [CmdletBinding()]
    param()

    $tracked = Invoke-CheckedCommand `
        -Command "git" `
        -Arguments @("ls-files") `
        -CaptureOutput
    $unsafe = @(
        $tracked -split "`r?`n" |
            Where-Object {
                $leaf = Split-Path $_ -Leaf
                $leaf -eq ".env" -or ($leaf -like ".env.*" -and $leaf -ne ".env.example")
            }
    )
    if ($unsafe.Count -gt 0) {
        throw "A tracked environment file was detected. Remove it from the Git index before continuing."
    }
}

function Assert-NoSuspiciousTrackedFileNames {
    [CmdletBinding()]
    param()

    $tracked = Invoke-CheckedCommand `
        -Command "git" `
        -Arguments @("ls-files") `
        -CaptureOutput
    $unsafe = @(
        $tracked -split "`r?`n" |
            Where-Object {
                $_ -match "(?i)(^|/)(credentials?|secrets?|tokens?)([._/-]|$)"
            }
    )
    if ($unsafe.Count -gt 0) {
        throw "A tracked filename suggests credentials, secrets, or tokens. Review tracked files."
    }
}

function Assert-NoObviousSecrets {
    [CmdletBinding()]
    param(
        [Parameter()]
        [switch] $Cached,

        [Parameter()]
        [switch] $IncludeUntracked
    )

    if ($Cached -and $IncludeUntracked) {
        throw "Cached and IncludeUntracked secret scans must run separately."
    }

    $pattern = "sk-[A-Za-z0-9_-]{20,}|OPENAI_API_KEY[[:space:]]*=[[:space:]]*[^[:space:]]+|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
    $arguments = @("grep", "-n", "-I", "-E")
    if ($Cached) {
        $arguments += "--cached"
    }
    if ($IncludeUntracked) {
        $arguments += @("--untracked", "--exclude-standard")
    }
    $arguments += @("--", $pattern)

    $null = @(& git @arguments 2>&1)
    $exitCode = $LASTEXITCODE
    if ($exitCode -eq 0) {
        throw "The secret scan found one or more possible credentials. Matches are intentionally not printed."
    }
    if ($exitCode -ne 1) {
        throw "The secret scan could not complete (git grep exit code $exitCode)."
    }
}

function Assert-NoObviousSecretsInHistory {
    [CmdletBinding()]
    param()

    $pattern = "sk-[A-Za-z0-9_-]{20,}|OPENAI_API_KEY[[:space:]]*=[[:space:]]*[^[:space:]]+|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
    $revisions = Invoke-CheckedCommand `
        -Command "git" `
        -Arguments @("rev-list", "--all") `
        -CaptureOutput
    foreach ($revision in @($revisions -split "`r?`n")) {
        if ([string]::IsNullOrWhiteSpace($revision)) {
            continue
        }
        $null = @(& git grep -I -E -- $pattern $revision 2>&1)
        $exitCode = $LASTEXITCODE
        if ($exitCode -eq 0) {
            throw "The Git history scan found a possible credential. Matches are intentionally not printed."
        }
        if ($exitCode -ne 1) {
            throw "The Git history scan could not inspect revision '$revision'."
        }
    }
}

function Invoke-LocalQualityChecks {
    [CmdletBinding()]
    param(
        [Parameter()]
        [string] $PythonCommand = "python"
    )

    Invoke-CheckedCommand `
        -Command $PythonCommand `
        -Arguments @("scripts/check_all.py")
}

function Assert-ReadmeReleaseReady {
    [CmdletBinding()]
    param()

    if (-not (Test-Path -LiteralPath "README.md" -PathType Leaf)) {
        throw "README.md is required before publication."
    }
    $readme = Get-Content -Raw -Encoding UTF8 "README.md"
    $placeholderPattern = "(?im)\b(TODO|TBD|FIXME|PLACEHOLDER)\b|^## Planned quick start\s*$|Development status:.*being built"
    if ($readme -match $placeholderPattern) {
        throw "README.md still contains placeholder or pre-release wording."
    }
}

function Assert-TruthfulSampleReport {
    [CmdletBinding()]
    param(
        [Parameter()]
        [switch] $Confirmed
    )

    $path = "results/sample_benchmark_report.md"
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "The curated sample report '$path' is required before publication."
    }

    $tracked = Invoke-CheckedCommand `
        -Command "git" `
        -Arguments @("ls-files", "--error-unmatch", "--", $path) `
        -CaptureOutput
    if ([string]::IsNullOrWhiteSpace($tracked)) {
        throw "The curated sample report must be tracked."
    }

    $report = Get-Content -Raw -Encoding UTF8 $path
    if ($report -notmatch "(?i)corpus.*[a-f0-9]{64}" -or
        $report -notmatch "(?i)(dataset|evaluation).*[a-f0-9]{64}" -or
        $report -notmatch "(?i)fake-provider|fake provider") {
        throw "The sample report must identify corpus and dataset checksums and the fake provider."
    }
    if (-not $Confirmed) {
        throw "Re-run the benchmark, inspect the generated sample, then pass -ConfirmSampleResultsTruthful."
    }
}

function Assert-SanitizedScreenshots {
    [CmdletBinding()]
    param(
        [Parameter()]
        [switch] $Confirmed
    )

    $screenshots = Invoke-CheckedCommand `
        -Command "git" `
        -Arguments @("ls-files", "docs/screenshots") `
        -CaptureOutput
    if ([string]::IsNullOrWhiteSpace($screenshots)) {
        return
    }
    if (-not $Confirmed) {
        throw "Tracked screenshots require manual privacy review and -ConfirmScreenshotsSanitized."
    }
}

function Assert-SuccessfulMainCi {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string] $Owner,

        [Parameter(Mandatory)]
        [string] $Repo
    )

    $headSha = Invoke-CheckedCommand `
        -Command "git" `
        -Arguments @("rev-parse", "HEAD") `
        -CaptureOutput
    $runJson = Invoke-CheckedCommand `
        -Command "gh" `
        -Arguments @(
            "run", "list",
            "--repo", "$Owner/$Repo",
            "--branch", "main",
            "--workflow", "ci.yml",
            "--limit", "1",
            "--json", "conclusion,headSha,status,url"
        ) `
        -CaptureOutput
    if ([string]::IsNullOrWhiteSpace($runJson)) {
        throw "No CI run was found for main."
    }

    $runs = @($runJson | ConvertFrom-Json)
    if ($runs.Count -ne 1) {
        throw "Unable to identify exactly one latest CI run for main."
    }
    $run = $runs[0]
    if ($run.status -ne "completed" -or $run.conclusion -ne "success") {
        throw "The latest CI run for main has not completed successfully."
    }
    if ($run.headSha -cne $headSha.Trim()) {
        throw "The latest successful CI run does not match the local main HEAD."
    }
}

function Get-GitHubRepositoryVisibility {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string] $Owner,

        [Parameter(Mandatory)]
        [string] $Repo
    )

    $visibility = Invoke-CheckedCommand `
        -Command "gh" `
        -Arguments @("repo", "view", "$Owner/$Repo", "--json", "visibility", "--jq", ".visibility") `
        -CaptureOutput
    return $visibility.Trim().ToUpperInvariant()
}

function Assert-PublicReleaseAudit {
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

    Assert-SafeRepositoryName -Owner $Owner -Repo $Repo
    Assert-GitHubPrerequisites -ExpectedOwner $Owner -PythonCommand $PythonCommand
    Assert-RemoteRepositoryIdentity -Owner $Owner -Repo $Repo
    Assert-CleanWorkingTree
    Assert-OnMainBranch
    Assert-NoTrackedEnvironmentFile
    Assert-NoSuspiciousTrackedFileNames
    Assert-NoObviousSecrets -Cached
    Assert-NoObviousSecrets -IncludeUntracked
    Assert-NoObviousSecretsInHistory
    Invoke-LocalQualityChecks -PythonCommand $PythonCommand
    Assert-ReadmeReleaseReady
    Assert-TruthfulSampleReport -Confirmed:$ConfirmSampleResultsTruthful
    Assert-SanitizedScreenshots -Confirmed:$ConfirmScreenshotsSanitized
    Assert-SuccessfulMainCi -Owner $Owner -Repo $Repo
}
