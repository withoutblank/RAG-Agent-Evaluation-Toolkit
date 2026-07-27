"""Typer command-line interface for the toolkit."""

import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated, NoReturn, cast

import typer

from rag_agent_eval_toolkit import __version__
from rag_agent_eval_toolkit.application import (
    build_and_save_index,
    create_agent_runner,
    load_runtime,
)
from rag_agent_eval_toolkit.benchmarking import (
    ProviderName,
    execute_benchmark,
    execute_evaluation,
)
from rag_agent_eval_toolkit.config import Settings, load_settings
from rag_agent_eval_toolkit.exceptions import ConfigurationError, RagEvalError
from rag_agent_eval_toolkit.experiments import load_experiment_result
from rag_agent_eval_toolkit.logging_config import configure_logging
from rag_agent_eval_toolkit.reporting import write_markdown_report

app = typer.Typer(
    name="rag-eval",
    help="Build and benchmark cited retrieval-augmented AI agents.",
    invoke_without_command=True,
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    rich_markup_mode=None,
)

ConfigOption = Annotated[
    Path | None,
    typer.Option(
        "--config",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Optional YAML configuration file.",
    ),
]


@app.callback()
def main(
    version_requested: Annotated[
        bool,
        typer.Option("--version", help="Show the package version and exit.", is_eager=True),
    ] = False,
) -> None:
    """Run the RAG Agent Evaluation Toolkit command-line interface."""
    if version_requested:
        typer.echo(__version__)
        raise typer.Exit()


@app.command()
def ingest(config: ConfigOption = None) -> None:
    """Ingest documents and build a local index."""
    settings = _command_settings(config)
    try:
        built = build_and_save_index(settings)
    except RagEvalError as exc:
        _fail(str(exc))
    typer.echo(f"Indexed {built.document_count} document(s) into {built.chunk_count} chunk(s).")
    typer.echo(f"Corpus checksum: {built.corpus_checksum}")
    typer.echo(f"Index directory: {settings.index_dir}")
    if built.duplicate_group_count:
        typer.echo(
            f"Warning: detected {built.duplicate_group_count} duplicate checksum group(s).",
            err=True,
        )


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question to answer from indexed evidence.")],
    config: ConfigOption = None,
) -> None:
    """Generate a direct cited RAG answer."""
    settings = _command_settings(config)
    try:
        answer = load_runtime(settings).rag_pipeline.ask(question)
    except RagEvalError as exc:
        _fail(str(exc))
    typer.echo(answer.answer)
    if answer.retrieval_results:
        typer.echo("\nRetrieved evidence:")
        for result in answer.retrieval_results:
            typer.echo(f"{result.rank}. {result.citation_label} (score={result.score:.4f})")
    for warning in answer.warnings:
        typer.echo(f"Warning: {warning}", err=True)


@app.command("agent-ask")
def agent_ask(
    question: Annotated[str, typer.Argument(help="Question for the tool-using agent.")],
    config: ConfigOption = None,
) -> None:
    """Generate a cited answer with the constrained retrieval agent."""
    settings = _command_settings(config)
    try:
        runtime = load_runtime(settings)
        answer = create_agent_runner(settings, runtime).run(question)
    except RagEvalError as exc:
        _fail(str(exc))
    typer.echo(answer.answer)
    if answer.tool_calls:
        typer.echo("\nTool trace:")
        for call in answer.tool_calls:
            status = "error" if call.error is not None else "ok"
            typer.echo(f"- {call.tool_name} [{status}]: {call.result_summary}")
    for warning in answer.warnings:
        typer.echo(f"Warning: {warning}", err=True)


@app.command()
def evaluate(
    config: ConfigOption = None,
    experiment: Annotated[
        str | None,
        typer.Option(help="Configuration name; defaults to the first benchmark entry."),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option(help="Override the configured provider with 'fake' or 'openai'."),
    ] = None,
) -> None:
    """Evaluate one configuration against the labelled dataset."""
    _evaluate(config=config, experiment_name=experiment, provider=provider)


@app.command()
def benchmark(
    config: ConfigOption = None,
    provider: Annotated[
        str | None,
        typer.Option(help="Override the configured provider with 'fake' or 'openai'."),
    ] = None,
) -> None:
    """Compare the required retrieval configurations."""
    benchmark_config = config or Path("configs/benchmark.yaml")
    try:
        execution = execute_benchmark(
            benchmark_config,
            provider=_validated_provider(provider),
        )
    except RagEvalError as exc:
        _fail(str(exc))

    typer.echo(
        f"Completed {len(execution.results)} configuration(s) over "
        f"{len(execution.dataset.questions)} question(s)."
    )
    typer.echo(f"Corpus checksum: {execution.corpus_checksum}")
    typer.echo(f"Summary CSV: {execution.outputs.summary_csv}")
    typer.echo(f"Markdown report: {execution.outputs.markdown_report}")
    for result in execution.results:
        metrics = result.aggregate_metrics
        typer.echo(
            f"- {result.config.name}: "
            f"Hit Rate@k={_metric_text(metrics.get('hit_rate_at_k'))}, "
            f"MRR={_metric_text(metrics.get('mrr'))}, "
            f"Citation validity={_metric_text(metrics.get('citation_validity'))}"
        )


@app.command()
def report(
    results_dir: Annotated[
        Path | None,
        typer.Option(
            "--results-dir",
            file_okay=False,
            help="Directory containing experiment result files.",
        ),
    ] = None,
    config: ConfigOption = None,
) -> None:
    """Generate a Markdown report from benchmark result files."""
    settings = _command_settings(config)
    source_dir = results_dir or settings.results_dir
    result_paths = sorted(source_dir.glob("*.json"))
    if not result_paths:
        _fail(f"No experiment JSON files were found in {source_dir}.")
    try:
        results = [load_experiment_result(path) for path in result_paths]
        output_path = write_markdown_report(results, source_dir / "report.md")
    except RagEvalError as exc:
        _fail(str(exc))
    typer.echo(f"Markdown report: {output_path}")


@app.command()
def doctor(config: ConfigOption = None) -> None:
    """Report local runtime, configuration, directory, index, and API-key status."""
    try:
        settings = load_settings(config)
    except ConfigurationError as exc:
        typer.echo(f"[error] configuration: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    configure_logging(settings.log_level)
    python_supported = sys.version_info >= (3, 11)
    typer.echo(
        f"[{'ok' if python_supported else 'error'}] "
        f"Python {platform.python_version()} (requires 3.11+)"
    )
    for label, path in (
        ("data directory", settings.data_dir),
        ("results directory", settings.results_dir),
    ):
        typer.echo(f"[{'ok' if path.is_dir() else 'missing'}] {label}: {path}")
    typer.echo(
        f"[{'ok' if settings.index_dir.is_dir() else 'missing'}] "
        f"index directory: {settings.index_dir}"
    )
    key_status = (
        "present" if settings.openai_api_key is not None else "not set (fake provider works)"
    )
    typer.echo(f"[info] OPENAI_API_KEY: {key_status}")
    typer.echo(f"[info] configured provider: {settings.provider}")
    for package_name in ("rag-agent-eval-toolkit", "numpy", "openai", "typer"):
        try:
            package_version = version(package_name)
        except PackageNotFoundError:
            package_version = "not installed"
        typer.echo(f"[info] {package_name}: {package_version}")
    if not python_supported:
        raise typer.Exit(code=1)


def _evaluate(
    *,
    config: Path | None,
    experiment_name: str | None,
    provider: str | None,
) -> None:
    benchmark_config = config or Path("configs/benchmark.yaml")
    try:
        result = execute_evaluation(
            benchmark_config,
            experiment_name=experiment_name,
            provider=_validated_provider(provider),
        )
    except RagEvalError as exc:
        _fail(str(exc))
    typer.echo(f"Configuration: {result.config.name}")
    typer.echo(
        "Questions: "
        f"{int(result.aggregate_metrics.get('successful_question_count', 0.0))} successful, "
        f"{int(result.aggregate_metrics.get('failed_question_count', 0.0))} failed"
    )
    for label, key in (
        ("Hit Rate@k", "hit_rate_at_k"),
        ("MRR", "mrr"),
        ("Source Recall@k", "source_recall_at_k"),
        ("Required-fact coverage", "required_fact_coverage"),
        ("Citation validity", "citation_validity"),
        ("Citation precision", "citation_precision"),
        ("Unanswerable accuracy", "unanswerable_accuracy"),
    ):
        typer.echo(f"{label}: {_metric_text(result.aggregate_metrics.get(key))}")
    for warning in result.warnings:
        typer.echo(f"Warning: {warning}", err=True)


def _validated_provider(value: str | None) -> ProviderName | None:
    if value is None:
        return None
    normalised = value.strip().lower()
    if normalised not in {"fake", "openai"}:
        _fail("--provider must be 'fake' or 'openai'.")
    return cast(ProviderName, normalised)


def _metric_text(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _command_settings(config_path: Path | None) -> Settings:
    try:
        settings = load_settings(config_path)
    except ConfigurationError as exc:
        _fail(f"configuration: {exc}")
    configure_logging(settings.log_level)
    return settings


def _fail(message: str) -> NoReturn:
    typer.echo(f"[error] {message}", err=True)
    raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
