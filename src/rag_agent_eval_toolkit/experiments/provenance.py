"""Read-only Git provenance detection for benchmark results."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

GIT_WORKTREE_STATE_FIELD = "git_worktree_state"
GIT_DIRTY_WARNING = (
    "Git worktree has tracked or untracked changes; git_commit was omitted because "
    "HEAD does not fully identify this run."
)
GIT_UNAVAILABLE_WARNING = "Git commit and worktree state could not be determined for this run."
GIT_UNVERIFIED_WARNING = (
    "Git worktree state is unverified because git_commit was supplied explicitly."
)
LEGACY_GIT_PROVENANCE_WARNING = (
    "Git worktree state was not recorded; git_commit provenance is unverified."
)

GitWorktreeState = Literal["clean", "dirty", "unavailable", "unverified"]
GIT_WORKTREE_STATES = frozenset({"clean", "dirty", "unavailable", "unverified"})

_GIT_COMMIT_PATTERN = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


@dataclass(frozen=True, slots=True)
class GitProvenance:
    """A commit that identifies the run inputs, plus its verification state."""

    commit: str | None
    worktree_state: GitWorktreeState


def detect_git_provenance(repo_dir: Path) -> GitProvenance:
    """Return a commit only when HEAD exists and the complete worktree is clean."""

    resolved = Path(repo_dir).resolve()
    commit_process = _run_git(resolved, "rev-parse", "--verify", "HEAD")
    if commit_process is None or commit_process.returncode != 0:
        return GitProvenance(commit=None, worktree_state="unavailable")

    commit = commit_process.stdout.strip().lower()
    if _GIT_COMMIT_PATTERN.fullmatch(commit) is None:
        return GitProvenance(commit=None, worktree_state="unavailable")

    status_process = _run_git(
        resolved,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    if status_process is None or status_process.returncode != 0:
        return GitProvenance(commit=None, worktree_state="unavailable")
    if status_process.stdout:
        return GitProvenance(commit=None, worktree_state="dirty")
    return GitProvenance(commit=commit, worktree_state="clean")


def detect_git_commit(repo_dir: Path) -> str | None:
    """Return the full local Git commit only when it identifies a clean worktree."""

    return detect_git_provenance(repo_dir).commit


def _run_git(
    repo_dir: Path,
    *arguments: str,
) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            [
                "git",
                "--no-optional-locks",
                "-c",
                f"safe.directory={repo_dir.as_posix()}",
                *arguments,
            ],
            cwd=repo_dir,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None


__all__ = [
    "GIT_DIRTY_WARNING",
    "GIT_UNAVAILABLE_WARNING",
    "GIT_UNVERIFIED_WARNING",
    "GIT_WORKTREE_STATE_FIELD",
    "GIT_WORKTREE_STATES",
    "LEGACY_GIT_PROVENANCE_WARNING",
    "GitProvenance",
    "GitWorktreeState",
    "detect_git_commit",
    "detect_git_provenance",
]
