[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = "Medium")]
param(
    [Parameter(Mandatory)]
    [ValidateNotNullOrEmpty()]
    [string] $Title,

    [Parameter()]
    [ValidateRange(1, 2147483647)]
    [int] $Issue,

    [Parameter()]
    [string] $PythonCommand = "python",

    [Parameter()]
    [switch] $Draft
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "github/common.ps1")

$projectRoot = Get-ProjectRoot -ScriptDirectory $PSScriptRoot
Push-Location $projectRoot
try {
    foreach ($command in @("git", "gh", $PythonCommand)) {
        Assert-CommandAvailable -Name $command
    }
    Invoke-CheckedCommand -Command "gh" -Arguments @("auth", "status")
    Assert-CleanWorkingTree
    $branch = Assert-NotMainBranch
    Assert-NoTrackedEnvironmentFile
    Assert-NoObviousSecrets -Cached
    Invoke-LocalQualityChecks -PythonCommand $PythonCommand

    $existingPr = Invoke-CheckedCommand `
        -Command "gh" `
        -Arguments @(
            "pr", "list",
            "--head", $branch,
            "--state", "open",
            "--limit", "1",
            "--json", "url",
            "--jq", ".[0].url"
        ) `
        -CaptureOutput
    if (-not [string]::IsNullOrWhiteSpace($existingPr)) {
        Write-Host "An open pull request already exists for '$branch': $existingPr"
        return
    }

    if (-not $PSCmdlet.ShouldProcess($branch, "push branch and open pull request")) {
        return
    }

    $templatePath = ".github/pull_request_template.md"
    if (-not (Test-Path -LiteralPath $templatePath -PathType Leaf)) {
        throw "Required pull request template '$templatePath' was not found."
    }
    $body = Get-Content -Raw -Encoding UTF8 $templatePath
    $body += @'

## Automated check evidence

```text
python scripts/check_all.py
exit code: 0
```
'@
    if ($PSBoundParameters.ContainsKey("Issue")) {
        $body += "`n`nCloses #$Issue`n"
    }

    $bodyFile = [System.IO.Path]::GetTempFileName()
    try {
        Set-Content -LiteralPath $bodyFile -Value $body -Encoding UTF8
        Invoke-CheckedCommand `
            -Command "git" `
            -Arguments @("push", "--set-upstream", "origin", $branch)

        $arguments = @(
            "pr", "create",
            "--base", "main",
            "--head", $branch,
            "--title", $Title,
            "--body-file", $bodyFile
        )
        if ($Draft) {
            $arguments += "--draft"
        }
        Invoke-CheckedCommand -Command "gh" -Arguments $arguments
    }
    finally {
        Remove-Item -LiteralPath $bodyFile -Force -ErrorAction SilentlyContinue
    }
}
finally {
    Pop-Location
}
