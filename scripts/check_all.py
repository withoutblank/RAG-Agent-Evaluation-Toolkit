"""Run the complete local quality gate using the active Python interpreter."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
RUN_STATE_DIR: Final = PROJECT_ROOT / ".rag_eval" / f"check-all-{os.getpid()}"
PYTEST_TEMP_DIR: Final = RUN_STATE_DIR / "pytest-temp"
PYTEST_CACHE_DIR: Final = RUN_STATE_DIR / "pytest-cache"
PIP_AUDIT_CACHE_DIR: Final = RUN_STATE_DIR / "pip-audit-cache"


@dataclass(frozen=True, slots=True)
class Check:
    """One named validation command."""

    name: str
    arguments: tuple[str, ...]


CHECKS: Final = (
    Check(
        "pytest",
        (
            "-m",
            "pytest",
            "--basetemp",
            str(PYTEST_TEMP_DIR),
            "-o",
            f"cache_dir={PYTEST_CACHE_DIR}",
        ),
    ),
    Check(
        "pytest with coverage",
        (
            "-m",
            "pytest",
            "--basetemp",
            str(PYTEST_TEMP_DIR),
            "-o",
            f"cache_dir={PYTEST_CACHE_DIR}",
            "--cov=rag_agent_eval_toolkit",
            "--cov-report=term-missing",
        ),
    ),
    Check("Ruff lint", ("-m", "ruff", "check", ".")),
    Check("Ruff format", ("-m", "ruff", "format", "--check", ".")),
    Check("mypy", ("-m", "mypy", "src")),
    Check(
        "pip-audit",
        (
            "-m",
            "pip_audit",
            "--cache-dir",
            str(PIP_AUDIT_CACHE_DIR),
        ),
    ),
)


def main() -> int:
    """Run every configured check and return zero only when all pass."""
    failures: list[str] = []
    RUN_STATE_DIR.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault("PYTHONUTF8", "1")
    environment["TEMP"] = str(RUN_STATE_DIR)
    environment["TMP"] = str(RUN_STATE_DIR)
    environment["TMPDIR"] = str(RUN_STATE_DIR)
    for check in CHECKS:
        command = (sys.executable, *check.arguments)
        print(f"\n[{check.name}] {' '.join(command)}", flush=True)
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            env=environment,
        )
        if completed.returncode != 0:
            failures.append(check.name)

    if failures:
        print(f"\nFailed checks: {', '.join(failures)}", file=sys.stderr)
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
