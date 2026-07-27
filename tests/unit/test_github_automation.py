"""Static contract tests for the safe GitHub automation scaffolding."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
WORKFLOWS = ROOT / ".github" / "workflows"

EXPECTED_ACTION_PINS = {
    "actions/checkout": (
        "de0fac2e4500dabe0009e67214ff5f5447ce83dd",
        "v6.0.2",
    ),
    "actions/dependency-review-action": (
        "2031cfc080254a8a887f58cffee85186f0e49e48",
        "v4.9.0",
    ),
    "actions/setup-python": (
        "a309ff8b426b58ec0e2a45f0f869d46889d02405",
        "v6.2.0",
    ),
    "gitleaks/gitleaks-action": (
        "ff98106e4c7b2bc287b24eaf42907196329070c7",
        "v2.3.9",
    ),
}
ACTION_REFERENCE_PATTERN = re.compile(
    r"^\s*uses:\s+(?P<action>[^@\s]+)@(?P<sha>[0-9a-f]{40})"
    r"\s+#\s+(?P<version>v\d+\.\d+\.\d+)\s*$",
    flags=re.MULTILINE,
)


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_external_workflow_actions_use_reviewed_immutable_pins() -> None:
    """Every external action uses its reviewed release SHA and readable version comment."""

    observed: dict[str, set[tuple[str, str]]] = {}
    total_references = 0
    workflow_paths = sorted({*WORKFLOWS.glob("*.yml"), *WORKFLOWS.glob("*.yaml")})

    for workflow_path in workflow_paths:
        workflow = workflow_path.read_text(encoding="utf-8")
        uses_lines = re.findall(r"^\s*uses:\s+(\S.*)$", workflow, flags=re.MULTILINE)
        matches = list(ACTION_REFERENCE_PATTERN.finditer(workflow))

        assert len(matches) == len(uses_lines), (
            f"{workflow_path.name} contains an unpinned or uncommented action reference"
        )
        total_references += len(matches)
        for match in matches:
            observed.setdefault(match["action"], set()).add((match["sha"], match["version"]))

    assert total_references == 10
    assert observed == {action: {pin} for action, pin in EXPECTED_ACTION_PINS.items()}


def test_required_github_automation_scripts_exist() -> None:
    """Phase 7 exposes each manual operation as a separate PowerShell script."""

    required = {
        "audit_public_release.ps1",
        "bootstrap_github_repo.ps1",
        "create_release.ps1",
        "open_pr.ps1",
        "publish_github_repo.ps1",
        "run_github_maintenance.ps1",
        "setup_github_project.ps1",
        "start_task.ps1",
    }
    assert required <= {path.name for path in SCRIPTS.glob("*.ps1")}
    assert (SCRIPTS / "github" / "common.ps1").is_file()


def test_repository_bootstrap_is_private_and_requires_explicit_confirmation() -> None:
    """Bootstrap refuses implicit creation, verifies the owner, and selects private visibility."""

    script = _read("scripts/bootstrap_github_repo.ps1")
    assert "$ConfirmCreate" in script
    assert "Assert-GitHubPrerequisites -ExpectedOwner $Owner" in script
    assert "Test-GitHubRepositoryExists" in script
    assert '"--private"' in script
    assert '"--remote", "origin"' in script
    assert "Get-GitHubRepositoryVisibility" in script
    assert '"PRIVATE"' in script
    assert "--public" not in script


def test_project_setup_contains_exact_mvp_labels_milestone_and_issues() -> None:
    """Idempotent setup retains MVP and automation labels, milestone, and issue titles."""

    script = _read("scripts/setup_github_project.ps1")
    expected_labels = {
        "type:feature",
        "type:bug",
        "type:docs",
        "type:test",
        "type:ci",
        "area:ingestion",
        "area:retrieval",
        "area:rag",
        "area:agent",
        "area:evaluation",
        "area:ui",
        "priority:high",
        "priority:medium",
        "good first issue",
        "help wanted",
        "dependencies",
    }
    expected_issues = {
        "Bootstrap Python package and quality tooling",
        "Implement document ingestion and stable metadata",
        "Implement deterministic chunking",
        "Implement embedding providers and NumPy vector index",
        "Implement cited direct RAG pipeline",
        "Implement retrieval tools and agent runner",
        "Create evaluation dataset and deterministic metrics",
        "Implement configuration benchmark runner",
        "Build Streamlit demonstration",
        "Add CI, security checks, and Dependabot",
        "Write public README and architecture documentation",
        "Run public-release audit and publish v0.1.0",
    }

    assert all(f'"{label}"' in script for label in expected_labels)
    assert all(f'Title = "{title}"' in script for title in expected_issues)
    assert '"v0.1.0 $([char]0x2014) Portfolio MVP"' in script
    assert "[Console]::OutputEncoding = $utf8NoBom" in script
    assert "[Console]::OutputEncoding = $previousConsoleOutputEncoding" in script
    assert '"--force"' in script
    assert "$existingTitles -ccontains $issue.Title" in script
    edit_arguments = script[
        script.index("$editArguments = @(") : script.index("foreach ($topic in $topics)")
    ]
    assert '"--delete-branch-on-merge=false"' in edit_arguments
    assert (
        re.search(
            r'^\s*"--delete-branch-on-merge",?\s*$',
            edit_arguments,
            flags=re.MULTILINE,
        )
        is None
    )
    assert "if ($metadata.deleteBranchOnMerge -or" in script
    assert '"--slurp"' in script
    assert "$milestonePagesJson | ConvertFrom-Json" in script
    assert "$matchingMilestones.Count -gt 1" in script
    for heading in (
        "## Objective",
        "## Implementation notes",
        "## Acceptance criteria",
        "## Tests required",
        "## Dependencies",
        "## Scope exclusions",
    ):
        assert heading in script


def test_project_setup_creates_every_label_used_by_dependabot() -> None:
    """Dependabot never requests a label omitted from idempotent project setup."""

    script = _read("scripts/setup_github_project.ps1")
    dependabot = _read(".github/dependabot.yml")
    expected_dependabot_labels = {"dependencies", "area:evaluation", "type:ci"}

    for label in expected_dependabot_labels:
        assert f"- {label}" in dependabot
        assert f'Name = "{label}"' in script


def test_task_and_pull_request_helpers_enforce_branch_safety() -> None:
    """Task and PR helpers require clean branches, fast-forward updates, and real check output."""

    start_task = _read("scripts/start_task.ps1")
    assert 'ValidateSet("feat", "fix", "docs", "test", "chore", "ci")' in start_task
    assert "Assert-CleanWorkingTree" in start_task
    assert "Assert-OnMainBranch" in start_task
    assert '"pull", "--ff-only", "origin", "main"' in start_task
    assert '"switch", "-c", $branchName' in start_task

    open_pr = _read("scripts/open_pr.ps1")
    assert "Assert-NotMainBranch" in open_pr
    assert "Invoke-LocalQualityChecks" in open_pr
    assert ".github/pull_request_template.md" in open_pr
    assert '"push", "--set-upstream", "origin", $branch' in open_pr
    assert '"pr", "create"' in open_pr
    assert "Closes #$Issue" in open_pr
    assert "exit code: 0" in open_pr


def test_publication_and_release_require_audits_and_confirmation() -> None:
    """Visibility and release writes stay behind explicit flags and verified release gates."""

    publish = _read("scripts/publish_github_repo.ps1")
    assert "$ConfirmPublic" in publish
    assert "Assert-PublicReleaseAudit" in publish
    assert '"--visibility", "public"' in publish
    assert '"--accept-visibility-change-consequences"' in publish
    assert '"PUBLIC"' in publish
    assert "already public. Nothing was changed." in publish

    release = _read("scripts/create_release.ps1")
    assert "$ConfirmRelease" in release
    assert "ValidatePattern(" in release
    assert "Assert-PublicReleaseAudit" in release
    assert "CHANGELOG.md must contain a release heading" in release
    assert '"tag", "-a", $tag' in release
    assert '"push", "origin", "refs/tags/$tag"' in release
    assert '"release", "create", $tag' in release


def test_shared_release_audit_covers_local_remote_and_privacy_gates() -> None:
    """The reusable audit checks quality, secrets, evidence, identity, and matching remote CI."""

    common = _read("scripts/github/common.ps1")
    required_calls = {
        "Assert-GitHubPrerequisites",
        "Assert-RemoteRepositoryIdentity",
        "Assert-CleanWorkingTree",
        "Assert-OnMainBranch",
        "Assert-NoTrackedEnvironmentFile",
        "Assert-NoSuspiciousTrackedFileNames",
        "Assert-NoObviousSecrets -Cached",
        "Assert-NoObviousSecrets -IncludeUntracked",
        "Assert-NoObviousSecretsInHistory",
        "Invoke-LocalQualityChecks",
        "Assert-ReadmeReleaseReady",
        "Assert-TruthfulSampleReport",
        "Assert-SanitizedScreenshots",
        "Assert-SuccessfulMainCi",
    }
    audit_function = common[common.index("function Assert-PublicReleaseAudit") :]
    assert all(call in audit_function for call in required_calls)
    assert "Matches are intentionally not printed." in common
    assert '"run", "list"' in common
    assert "$run.headSha -cne $headSha.Trim()" in common


def test_scripts_do_not_expose_destructive_repository_operations() -> None:
    """Automation contains no repository deletion, hard reset, force-push, or main push."""

    combined = "\n".join(path.read_text(encoding="utf-8") for path in SCRIPTS.rglob("*.ps1"))
    forbidden = {
        "gh repo delete",
        "git reset --hard",
        '"push", "--force"',
        '"push", "-f"',
        '"push", "origin", "main"',
    }
    assert all(operation not in combined for operation in forbidden)


def test_manual_maintenance_uses_fast_forward_and_read_only_github_lists() -> None:
    """The documented weekly routine updates only by fast-forward and reports remote state."""

    script = _read("scripts/run_github_maintenance.ps1")
    assert "Assert-CleanWorkingTree" in script
    assert "Assert-OnMainBranch" in script
    assert '"pull", "--ff-only", "origin", "main"' in script
    assert "Invoke-LocalQualityChecks" in script
    assert '"pr", "list"' in script
    assert '"issue", "list"' in script
    assert '"run", "list"' in script


def test_weekly_health_is_read_only_offline_and_verifies_four_artifacts() -> None:
    """The scheduled health workflow uses fakes and fails if benchmark outputs are absent."""

    workflow = _read(".github/workflows/weekly-health.yml")
    assert "schedule:" in workflow
    assert "workflow_dispatch:" in workflow
    assert "permissions:\n  contents: read" in workflow
    assert "RAG_PROVIDER: fake" in workflow
    assert "--provider fake" in workflow
    assert "python -m pytest" in workflow
    assert "test -s results/summary.csv" in workflow
    assert "test -s results/report.md" in workflow
    assert "\"$(find results -maxdepth 1 -type f -name '*.json' | wc -l)\" -eq 4" in workflow
    assert "${{ secrets." not in workflow
    assert "openai" not in workflow.lower()
