"""Injectable deterministic and hosted answer-generation providers."""

from rag_agent_eval_toolkit.generation.base import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    GenerationProvider,
    GenerationResult,
)
from rag_agent_eval_toolkit.generation.fake_provider import (
    DeterministicFakeGenerationProvider,
    FakeGenerationProvider,
)
from rag_agent_eval_toolkit.generation.openai_provider import (
    OpenAIGenerationClient,
    OpenAIGenerationProvider,
)

__all__ = [
    "INSUFFICIENT_EVIDENCE_ANSWER",
    "DeterministicFakeGenerationProvider",
    "FakeGenerationProvider",
    "GenerationProvider",
    "GenerationResult",
    "OpenAIGenerationClient",
    "OpenAIGenerationProvider",
]
