"""Domain-specific exceptions for the toolkit."""


class RagEvalError(Exception):
    """Base exception for expected toolkit failures."""


class ConfigurationError(RagEvalError):
    """Raised when application configuration is invalid or unreadable."""


class UnsupportedFileTypeError(RagEvalError):
    """Raised when document ingestion receives an unsupported file type."""


class DocumentLoadError(RagEvalError):
    """Raised when a supported document cannot be loaded."""


class EmptyDocumentError(DocumentLoadError):
    """Raised when a loaded document contains no usable text."""


class InvalidChunkConfigError(RagEvalError):
    """Raised when a chunking configuration is invalid."""


class EmbeddingProviderError(RagEvalError):
    """Raised when an embedding provider cannot complete a request."""


class IndexCompatibilityError(RagEvalError):
    """Raised when persisted index data is missing or incompatible."""


class RetrievalError(RagEvalError):
    """Raised when evidence retrieval cannot be completed."""


class GenerationProviderError(RagEvalError):
    """Raised when a generation provider cannot complete a request."""


class CitationValidationError(RagEvalError):
    """Raised when an answer contains an invalid citation."""


class EvaluationDatasetError(RagEvalError):
    """Raised when an evaluation dataset is invalid or unreadable."""


class ExperimentError(RagEvalError):
    """Raised when an experiment cannot be executed or recorded."""


__all__ = [
    "CitationValidationError",
    "ConfigurationError",
    "DocumentLoadError",
    "EmbeddingProviderError",
    "EmptyDocumentError",
    "EvaluationDatasetError",
    "ExperimentError",
    "GenerationProviderError",
    "IndexCompatibilityError",
    "InvalidChunkConfigError",
    "RagEvalError",
    "RetrievalError",
    "UnsupportedFileTypeError",
]
