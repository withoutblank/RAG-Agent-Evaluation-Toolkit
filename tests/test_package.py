"""Package import smoke tests."""

from rag_agent_eval_toolkit import __version__


def test_package_exposes_version() -> None:
    """The source-layout package imports and exposes its release version."""
    assert __version__ == "0.1.0"
