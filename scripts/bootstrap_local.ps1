[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$PythonCommand = "python",
    [switch]$InstallDependencies,
    [switch]$RunChecks,
    [switch]$InitializeGit
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VirtualEnvironment = Join-Path $ProjectRoot ".venv"
$VirtualEnvironmentPython = Join-Path $VirtualEnvironment "Scripts\python.exe"

Push-Location $ProjectRoot
try {
    if (-not (Test-Path -LiteralPath $VirtualEnvironmentPython -PathType Leaf)) {
        $ResolvedPython = Get-Command $PythonCommand -ErrorAction Stop
        if ($PSCmdlet.ShouldProcess($VirtualEnvironment, "Create Python virtual environment")) {
            & $ResolvedPython.Source -m venv $VirtualEnvironment
            if ($LASTEXITCODE -ne 0) {
                throw "Virtual environment creation failed with exit code $LASTEXITCODE."
            }
        }
    }

    if (-not (Test-Path -LiteralPath $VirtualEnvironmentPython -PathType Leaf)) {
        throw "Virtual environment Python was not found at $VirtualEnvironmentPython."
    }

    $VersionOutput = & $VirtualEnvironmentPython --version
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to run virtual environment Python."
    }
    $VersionMatch = [regex]::Match($VersionOutput, "Python (\d+)\.(\d+)")
    if (-not $VersionMatch.Success) {
        throw "Unable to parse Python version from: $VersionOutput"
    }
    $MajorVersion = [int]$VersionMatch.Groups[1].Value
    $MinorVersion = [int]$VersionMatch.Groups[2].Value
    if (($MajorVersion -lt 3) -or (($MajorVersion -eq 3) -and ($MinorVersion -lt 11))) {
        throw "Python 3.11 or newer is required; found $VersionOutput."
    }
    Write-Host "Using $VersionOutput"

    if ($InstallDependencies) {
        if ($PSCmdlet.ShouldProcess($ProjectRoot, "Install project and development dependencies")) {
            & $VirtualEnvironmentPython -m pip install --upgrade pip
            if ($LASTEXITCODE -ne 0) {
                throw "pip upgrade failed with exit code $LASTEXITCODE."
            }
            & $VirtualEnvironmentPython -m pip install --editable "${ProjectRoot}[dev]"
            if ($LASTEXITCODE -ne 0) {
                throw "Dependency installation failed with exit code $LASTEXITCODE."
            }
        }
    }
    else {
        Write-Host "Dependencies were not installed. Re-run with -InstallDependencies when ready."
    }

    if ($RunChecks) {
        & $VirtualEnvironmentPython (Join-Path $ProjectRoot "scripts\check_all.py")
        if ($LASTEXITCODE -ne 0) {
            throw "Local checks failed with exit code $LASTEXITCODE."
        }
    }

    if ($InitializeGit) {
        if (Test-Path -LiteralPath (Join-Path $ProjectRoot ".git")) {
            Write-Host "Git is already initialized; no repository state was changed."
        }
        else {
            $GitCommand = Get-Command git -ErrorAction Stop
            if ($PSCmdlet.ShouldProcess($ProjectRoot, "Initialize local Git repository on main")) {
                & $GitCommand.Source init
                if ($LASTEXITCODE -ne 0) {
                    throw "git init failed with exit code $LASTEXITCODE."
                }
                & $GitCommand.Source branch -M main
                if ($LASTEXITCODE -ne 0) {
                    throw "Unable to name the initial branch main."
                }
            }
        }
    }

    Write-Host "Local bootstrap completed. No remote repository action was performed."
}
finally {
    Pop-Location
}
