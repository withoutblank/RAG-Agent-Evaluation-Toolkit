"""Offline integration from corpus loading through evaluation report writing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rag_agent_eval_toolkit.chunking import chunk_documents
from rag_agent_eval_toolkit.embeddings import FakeEmbeddingProvider
from rag_agent_eval_toolkit.evaluation import load_evaluation_dataset
from rag_agent_eval_toolkit.experiments import run_experiment
from rag_agent_eval_toolkit.generation import FakeGenerationProvider
from rag_agent_eval_toolkit.index import NumpyVectorStore
from rag_agent_eval_toolkit.ingestion import NORMALISATION_METHOD, load_documents
from rag_agent_eval_toolkit.models import ExperimentConfig
from rag_agent_eval_toolkit.rag import DirectRagPipeline
from rag_agent_eval_toolkit.reporting import write_benchmark_outputs
from rag_agent_eval_toolkit.retrieval import Retriever


def test_offline_pipeline_evaluates_and_writes_report(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "guide.md").write_text(
        "# Charging\n\nInspect the connector before charging.",
        encoding="utf-8",
        newline="\n",
    )
    dataset_path = tmp_path / "questions.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "ev-offline",
                "question": "What should be inspected before charging?",
                "expected_answer": "Inspect the connector.",
                "expected_sources": ["guide.md"],
                "required_facts": ["inspect the connector"],
                "answerable": True,
                "tags": ["charging"],
                "difficulty": "easy",
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    documents = load_documents(corpus_dir)
    chunks = chunk_documents(documents, chunk_size=80, overlap=10)
    corpus_checksum = hashlib.sha256(
        "".join(document.content_sha256 for document in documents).encode()
    ).hexdigest()
    embedding_provider = FakeEmbeddingProvider(dimensions=128, seed=42)
    store = NumpyVectorStore.build(
        chunks,
        embedding_provider,
        corpus_checksum=corpus_checksum,
        normalisation_method=NORMALISATION_METHOD,
        created_at_utc="2026-01-01T00:00:00Z",
    )
    retriever = Retriever(store, embedding_provider, default_top_k=1)
    citation_label = f"[guide.md#{chunks[0].chunk_id}]"
    generation_provider = FakeGenerationProvider(
        response=f"Inspect the connector before charging. {citation_label}",
        model_name="fake-cited-v1",
    )
    pipeline = DirectRagPipeline(
        retriever,
        generation_provider,
        default_top_k=1,
        temperature=0,
        prompt_version="v1",
    )
    dataset = load_evaluation_dataset(dataset_path)
    config = ExperimentConfig(
        name="offline-k1",
        chunk_size=80,
        overlap=10,
        top_k=1,
        embedding_model=embedding_provider.model_name,
        generation_model=generation_provider.model_name,
        seed=42,
    )

    def execute(question: str, experiment: ExperimentConfig) -> object:
        return pipeline.ask(question, top_k=experiment.top_k)

    result = run_experiment(
        config,
        dataset,
        execute,
        corpus_checksum=corpus_checksum,
        available_sources={"guide.md"},
        git_commit="e" * 40,
        timestamp_factory=lambda: "2026-01-01T00:00:00Z",
        run_id_factory=lambda: "offline-run",
    )
    outputs = write_benchmark_outputs([result], tmp_path / "results")

    assert result.aggregate_metrics["hit_rate_at_k"] == 1.0
    assert result.aggregate_metrics["required_fact_coverage"] == 1.0
    assert result.aggregate_metrics["citation_validity"] == 1.0
    assert outputs.experiment_json[0].is_file()
    assert outputs.summary_csv.is_file()
    assert "offline-k1" in outputs.markdown_report.read_text(encoding="utf-8")
