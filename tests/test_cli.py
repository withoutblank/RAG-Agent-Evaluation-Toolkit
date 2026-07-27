"""CLI registration smoke tests."""

from typer.testing import CliRunner

from rag_agent_eval_toolkit.cli import app

runner = CliRunner()


def test_cli_exposes_required_commands() -> None:
    """The top-level help lists every command required by the specification."""
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in (
        "ingest",
        "ask",
        "agent-ask",
        "evaluate",
        "benchmark",
        "report",
        "doctor",
    ):
        assert command in result.stdout


def test_cli_reports_package_version() -> None:
    """The eager version option works without loading a provider or making a network call."""
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "0.1.0"


def test_evaluate_command_runs_one_offline_configuration() -> None:
    """The evaluation command executes deterministically without a paid provider."""
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--config",
            "configs/benchmark.yaml",
            "--experiment",
            "small-k3",
            "--provider",
            "fake",
        ],
    )

    assert result.exit_code == 0
    assert "Configuration: small-k3" in result.stdout
    assert "Questions: 20 successful, 0 failed" in result.stdout
