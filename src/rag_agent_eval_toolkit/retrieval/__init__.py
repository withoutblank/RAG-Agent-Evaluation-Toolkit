"""Ranked retrieval over a compatible embedding index."""

from rag_agent_eval_toolkit.retrieval.retriever import (
    Retriever,
    citation_label_for,
)

__all__ = ["Retriever", "citation_label_for"]
