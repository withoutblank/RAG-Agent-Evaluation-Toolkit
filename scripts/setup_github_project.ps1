[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = "Medium")]
param(
    [Parameter(Mandatory)]
    [string] $Owner,

    [Parameter()]
    [string] $Repo = "rag-agent-eval-toolkit",

    [Parameter()]
    [switch] $Apply
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$previousConsoleOutputEncoding = [Console]::OutputEncoding
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$script:OutputEncoding = $utf8NoBom
[Console]::OutputEncoding = $utf8NoBom

. (Join-Path $PSScriptRoot "github/common.ps1")

function New-IssueBody {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string] $Objective,

        [Parameter(Mandatory)]
        [string] $ImplementationNotes,

        [Parameter(Mandatory)]
        [string] $AcceptanceCriteria,

        [Parameter(Mandatory)]
        [string] $TestsRequired,

        [Parameter(Mandatory)]
        [string] $Dependencies,

        [Parameter(Mandatory)]
        [string] $ScopeExclusions
    )

    return @"
## Objective

$Objective

## Implementation notes

$ImplementationNotes

## Acceptance criteria

$AcceptanceCriteria

## Tests required

$TestsRequired

## Dependencies

$Dependencies

## Scope exclusions

$ScopeExclusions
"@
}

if (-not $Apply) {
    throw "Remote project setup requires the explicit -Apply flag."
}
Assert-SafeRepositoryName -Owner $Owner -Repo $Repo

$projectRoot = Get-ProjectRoot -ScriptDirectory $PSScriptRoot
Push-Location $projectRoot
try {
    Assert-CommandAvailable -Name "git"
    Assert-CommandAvailable -Name "gh"
    Invoke-CheckedCommand -Command "gh" -Arguments @("auth", "status")
    $detectedOwner = Get-AuthenticatedGitHubOwner
    if ($detectedOwner -cne $Owner) {
        throw "Authenticated GitHub owner mismatch. Expected '$Owner'; detected '$detectedOwner'."
    }

    $fullName = "$Owner/$Repo"
    Assert-RemoteRepositoryIdentity -Owner $Owner -Repo $Repo
    if (-not (Test-GitHubRepositoryExists -FullName $fullName)) {
        throw "GitHub repository '$fullName' does not exist."
    }
    if (-not $PSCmdlet.ShouldProcess(
            $fullName,
            "configure metadata, labels, milestone, and implementation issues"
        )) {
        return
    }

    $repoEditHelp = Invoke-CheckedCommand `
        -Command "gh" `
        -Arguments @("repo", "edit", "--help") `
        -CaptureOutput
    foreach ($requiredFlag in @(
            "--enable-issues",
            "--enable-wiki",
            "--delete-branch-on-merge",
            "--add-topic"
        )) {
        if ($repoEditHelp -notmatch [regex]::Escape($requiredFlag)) {
            throw "Installed GitHub CLI does not support required flag '$requiredFlag'."
        }
    }

    $description = "A lightweight Python toolkit for building and benchmarking cited RAG agents."
    $topics = @(
        "rag",
        "llm",
        "ai-agents",
        "evaluation",
        "benchmarking",
        "python",
        "openai",
        "retrieval-augmented-generation",
        "streamlit"
    )
    $editArguments = @(
        "repo", "edit", $fullName,
        "--enable-issues=true",
        "--enable-wiki=false",
        "--delete-branch-on-merge=false",
        "--description", $description
    )
    foreach ($topic in $topics) {
        $editArguments += @("--add-topic", $topic)
    }
    Invoke-CheckedCommand -Command "gh" -Arguments $editArguments

    $labels = @(
        [pscustomobject]@{ Name = "type:feature"; Color = "1D76DB"; Description = "New capability" },
        [pscustomobject]@{ Name = "type:bug"; Color = "D73A4A"; Description = "Defect" },
        [pscustomobject]@{ Name = "type:docs"; Color = "0075CA"; Description = "Documentation" },
        [pscustomobject]@{ Name = "type:test"; Color = "5319E7"; Description = "Tests" },
        [pscustomobject]@{ Name = "type:ci"; Color = "0E8A16"; Description = "Automation" },
        [pscustomobject]@{ Name = "area:ingestion"; Color = "C5DEF5"; Description = "Document pipeline" },
        [pscustomobject]@{ Name = "area:retrieval"; Color = "C5DEF5"; Description = "Embeddings and search" },
        [pscustomobject]@{ Name = "area:rag"; Color = "C5DEF5"; Description = "Answer generation" },
        [pscustomobject]@{ Name = "area:agent"; Color = "C5DEF5"; Description = "Agent tools" },
        [pscustomobject]@{ Name = "area:evaluation"; Color = "C5DEF5"; Description = "Metrics and experiments" },
        [pscustomobject]@{ Name = "area:ui"; Color = "C5DEF5"; Description = "Streamlit" },
        [pscustomobject]@{ Name = "priority:high"; Color = "B60205"; Description = "Required for MVP" },
        [pscustomobject]@{ Name = "priority:medium"; Color = "FBCA04"; Description = "Useful after core" },
        [pscustomobject]@{ Name = "good first issue"; Color = "7057FF"; Description = "Small external contribution" },
        [pscustomobject]@{ Name = "help wanted"; Color = "008672"; Description = "Open contribution opportunity" },
        [pscustomobject]@{ Name = "dependencies"; Color = "0366D6"; Description = "Dependency updates" }
    )
    foreach ($label in $labels) {
        Invoke-CheckedCommand `
            -Command "gh" `
            -Arguments @(
                "label", "create", $label.Name,
                "--repo", $fullName,
                "--color", $label.Color,
                "--description", $label.Description,
                "--force"
            )
    }

    $milestoneTitle = "v0.1.0 $([char]0x2014) Portfolio MVP"
    $milestonePagesJson = Invoke-CheckedCommand `
        -Command "gh" `
        -Arguments @(
            "api",
            "--paginate",
            "--slurp",
            "repos/$fullName/milestones?state=all&per_page=100"
        ) `
        -CaptureOutput
    $milestonePages = $milestonePagesJson | ConvertFrom-Json
    $matchingMilestones = @(
        foreach ($page in $milestonePages) {
            foreach ($milestone in @($page)) {
                if ($milestone.title -ceq $milestoneTitle) {
                    $milestone
                }
            }
        }
    )
    if ($matchingMilestones.Count -gt 1) {
        throw "Multiple milestones named '$milestoneTitle' exist; resolve the duplicate before setup."
    }
    if ($matchingMilestones.Count -eq 1) {
        $milestoneNumber = "$($matchingMilestones[0].number)"
    }
    else {
        $milestoneNumber = Invoke-CheckedCommand `
            -Command "gh" `
            -Arguments @(
                "api",
                "--method", "POST",
                "repos/$fullName/milestones",
                "-f", "title=$milestoneTitle",
                "-f", "description=Tracks the complete v0.1.0 portfolio MVP acceptance scope.",
                "--jq", ".number"
            ) `
            -CaptureOutput
    }

    $issues = @(
        [pscustomobject]@{
            Title = "Bootstrap Python package and quality tooling"
            Objective = "Create the installable Python package, typed configuration, CLI skeleton, and local quality gate."
            Notes = "Use the src layout, Python 3.11+, Typer, strict mypy, Ruff, pytest, coverage, and pip-audit."
            Acceptance = "The package imports, CLI help works, a trivial test passes, and scripts/check_all.py runs the required checks."
            Tests = "Unit smoke tests for imports, configuration precedence, and CLI registration."
            Dependencies = "None."
            Exclusions = "No retrieval, generation, UI, remote publication, or cloud deployment."
            Labels = @("type:feature", "priority:high")
        },
        [pscustomobject]@{
            Title = "Implement document ingestion and stable metadata"
            Objective = "Load Markdown, text, and text-based PDF files while preserving stable source metadata and checksums."
            Notes = "Validate extensions, paths, encodings, empty content, page metadata, and duplicate checksums."
            Acceptance = "All eight fictional documents load and retain source name, relative path, page where available, and checksum."
            Tests = "Unit fixtures for all supported formats plus empty, unreadable, unsupported, duplicate, and path-normalisation cases."
            Dependencies = "Bootstrap Python package and quality tooling."
            Exclusions = "OCR, web crawling, private documents, and live APIs."
            Labels = @("type:feature", "area:ingestion", "priority:high")
        },
        [pscustomobject]@{
            Title = "Implement deterministic chunking"
            Objective = "Create deterministic character-based chunking with overlap and stable chunk identifiers."
            Notes = "Preserve all source metadata and reject overlap greater than or equal to chunk size."
            Acceptance = "Identical content and configuration produce identical non-empty chunks and IDs."
            Tests = "Boundary, overlap, short-document, repeated-heading, invalid-configuration, and page-propagation tests."
            Dependencies = "Implement document ingestion and stable metadata."
            Exclusions = "Semantic chunking, external tokenisers, and model calls."
            Labels = @("type:feature", "area:ingestion", "priority:high")
        },
        [pscustomobject]@{
            Title = "Implement embedding providers and NumPy vector index"
            Objective = "Add injectable fake and OpenAI embeddings plus a local NumPy cosine-similarity index."
            Notes = "Persist vectors and a manifest containing dimensions, model, corpus checksum, and chunking configuration."
            Acceptance = "Fake rankings are deterministic, reload is equivalent, and incompatible dimensions or manifests fail clearly."
            Tests = "Provider injection, ranking, ties, empty index, persistence, top-k bounds, and dimension mismatch."
            Dependencies = "Implement deterministic chunking."
            Exclusions = "Hosted vector databases, Redis, and live API calls in tests."
            Labels = @("type:feature", "area:retrieval", "priority:high")
        },
        [pscustomobject]@{
            Title = "Implement cited direct RAG pipeline"
            Objective = "Retrieve ranked evidence and generate context-constrained answers with resolvable citations."
            Notes = "Use injectable fake/OpenAI generation providers and explicit insufficient-evidence behavior."
            Acceptance = "Every answer citation maps to a retrieved chunk and answers retain ranked evidence and timing fields."
            Tests = "Prompt isolation, citation parsing and validation, abstention, fake injection, and offline integration."
            Dependencies = "Implement embedding providers and NumPy vector index."
            Exclusions = "Uncited answers, web context, hidden filesystem access, and paid CI calls."
            Labels = @("type:feature", "area:rag", "priority:high")
        },
        [pscustomobject]@{
            Title = "Implement retrieval tools and agent runner"
            Objective = "Expose typed search_corpus and get_source_metadata tools through a constrained agent runner."
            Notes = "Capture tool traces and validate final citations against retrieved evidence."
            Acceptance = "The agent uses both typed tools where appropriate, returns citations, and exposes no arbitrary shell or filesystem tool."
            Tests = "Tool schemas, validation, outputs, trace capture, citation enforcement, and fake model behavior."
            Dependencies = "Implement cited direct RAG pipeline."
            Exclusions = "Autonomous browsing, arbitrary code execution, and general-purpose tools."
            Labels = @("type:feature", "area:agent", "priority:high")
        },
        [pscustomobject]@{
            Title = "Create evaluation dataset and deterministic metrics"
            Objective = "Create at least 20 labelled questions and deterministic retrieval, answer, citation, abstention, and latency metrics."
            Notes = "Include answerable, multi-source, difficult, and unanswerable questions; keep LLM judging separate and optional."
            Acceptance = "Hand-calculated fixtures validate every metric and dataset labels never enter retrieval input."
            Tests = "Hit Rate@k, MRR, Source Recall@k, fact coverage, citation validity/precision, unanswerable accuracy, and latency."
            Dependencies = "Implement cited direct RAG pipeline."
            Exclusions = "Fabricated results and mandatory non-deterministic judging."
            Labels = @("type:feature", "area:evaluation", "priority:high")
        },
        [pscustomobject]@{
            Title = "Implement configuration benchmark runner"
            Objective = "Run the four required chunk-size, overlap, and top-k configurations against one immutable corpus and dataset."
            Notes = "Record configuration, seed, model versions, corpus checksum, evaluation checksum, failures, and timings."
            Acceptance = "The fake-provider benchmark is reproducible and writes JSON, CSV, and a truthful Markdown report."
            Tests = "Four-config execution, duplicate-name rejection, repeatability, serialization, checksums, and partial failure."
            Dependencies = "Create evaluation dataset and deterministic metrics."
            Exclusions = "Unsupported best-model claims, hidden configuration changes, and committed live result dumps."
            Labels = @("type:feature", "area:evaluation", "priority:high")
        },
        [pscustomobject]@{
            Title = "Build Streamlit demonstration"
            Objective = "Provide a compact reviewer-facing UI for corpus status, direct RAG, agent mode, evidence, and benchmark comparison."
            Notes = "Use the same application services as the CLI and display fictional-data and API-usage disclaimers."
            Acceptance = "The app starts, missing API keys are actionable, and no secret or absolute local path is rendered."
            Tests = "Import smoke test and mocked interactions for direct, agent, missing-key, evidence, and benchmark states."
            Dependencies = "Implement configuration benchmark runner and retrieval tools and agent runner."
            Exclusions = "Authentication, cloud deployment, multi-page polish, and private screenshots."
            Labels = @("type:feature", "area:ui", "priority:medium")
        },
        [pscustomobject]@{
            Title = "Add CI, security checks, and Dependabot"
            Objective = "Run offline quality, coverage, dependency, and secret checks with least-privilege GitHub workflows."
            Notes = "Test Python 3.11 and 3.12; schedule a fake-provider health benchmark; never auto-merge dependency updates."
            Acceptance = "CI, security, and weekly health workflows are read-only and Dependabot monitors pip and Actions weekly."
            Tests = "Static workflow validation plus successful remote CI without paid API calls."
            Dependencies = "Bootstrap Python package and quality tooling; benchmark command for scheduled health."
            Exclusions = "Write permissions for test jobs, automatic PR approval, and paid model calls."
            Labels = @("type:ci", "priority:high")
        },
        [pscustomobject]@{
            Title = "Write public README and architecture documentation"
            Objective = "Create concise reviewer-facing setup, architecture, evaluation, limitations, security, and contribution documentation."
            Notes = "Use only tested commands, fictional data, relative paths, and generated evidence."
            Acceptance = "A fresh reviewer can install, run tests, inspect the sample benchmark, and understand limitations."
            Tests = "Run documented PowerShell setup and inspect links, paths, screenshots, and sample provenance."
            Dependencies = "Core implementation and benchmark runner."
            Exclusions = "Unverified badges, fabricated performance claims, personal contact data, and unapproved licence text."
            Labels = @("type:docs", "priority:high")
        },
        [pscustomobject]@{
            Title = "Run public-release audit and publish v0.1.0"
            Objective = "Audit the completed private repository, explicitly approve publication, and create the v0.1.0 release."
            Notes = "Require clean main, passing local and remote checks, secret scan, truthful sample results, and reviewed screenshots."
            Acceptance = "Public visibility changes only with ConfirmPublic and the release tag is created only after all acceptance gates pass."
            Tests = "Execute the release audit, verify repository settings and CI, and confirm release metadata after creation."
            Dependencies = "All previous portfolio MVP issues."
            Exclusions = "Automatic publication, force-push, repository deletion, and release claims without evidence."
            Labels = @("type:ci", "type:docs", "priority:high")
        }
    )

    $existingTitlesRaw = Invoke-CheckedCommand `
        -Command "gh" `
        -Arguments @(
            "issue", "list",
            "--repo", $fullName,
            "--state", "all",
            "--limit", "1000",
            "--json", "title",
            "--jq", ".[].title"
        ) `
        -CaptureOutput
    $existingTitles = @($existingTitlesRaw -split "`r?`n")

    foreach ($issue in $issues) {
        if ($existingTitles -ccontains $issue.Title) {
            Write-Host "Issue already exists: $($issue.Title)"
            continue
        }

        $body = New-IssueBody `
            -Objective $issue.Objective `
            -ImplementationNotes $issue.Notes `
            -AcceptanceCriteria $issue.Acceptance `
            -TestsRequired $issue.Tests `
            -Dependencies $issue.Dependencies `
            -ScopeExclusions $issue.Exclusions
        $bodyFile = [System.IO.Path]::GetTempFileName()
        try {
            Set-Content -LiteralPath $bodyFile -Value $body -Encoding UTF8
            $issueArguments = @(
                "issue", "create",
                "--repo", $fullName,
                "--title", $issue.Title,
                "--body-file", $bodyFile,
                "--milestone", $milestoneTitle
            )
            foreach ($labelName in $issue.Labels) {
                $issueArguments += @("--label", $labelName)
            }
            Invoke-CheckedCommand -Command "gh" -Arguments $issueArguments
        }
        finally {
            Remove-Item -LiteralPath $bodyFile -Force -ErrorAction SilentlyContinue
        }
    }

    $metadataJson = Invoke-CheckedCommand `
        -Command "gh" `
        -Arguments @(
            "repo", "view", $fullName,
            "--json",
            "deleteBranchOnMerge,description,hasIssuesEnabled,hasWikiEnabled,repositoryTopics"
        ) `
        -CaptureOutput
    $metadata = $metadataJson | ConvertFrom-Json
    if ($metadata.deleteBranchOnMerge -or
        -not $metadata.hasIssuesEnabled -or
        $metadata.hasWikiEnabled -or
        $metadata.description -cne $description) {
        throw "Repository metadata verification failed after setup."
    }
    $verifiedTopics = @($metadata.repositoryTopics | ForEach-Object { $_.name })
    foreach ($topic in $topics) {
        if ($verifiedTopics -notcontains $topic) {
            throw "Repository topic '$topic' was not present after setup."
        }
    }
    Write-Host "Repository metadata, labels, milestone, and implementation issues are configured; automatic branch deletion is disabled."
}
finally {
    Pop-Location
    [Console]::OutputEncoding = $previousConsoleOutputEncoding
}
