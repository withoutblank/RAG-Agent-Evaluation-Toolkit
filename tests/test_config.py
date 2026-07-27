"""Configuration precedence and validation tests."""

from pathlib import Path

import pytest

from rag_agent_eval_toolkit.config import Settings, load_settings
from rag_agent_eval_toolkit.exceptions import ConfigurationError


@pytest.fixture(autouse=True)
def clear_configuration_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep configuration tests independent of a developer's live environment."""
    for variable in (
        "OPENAI_API_KEY",
        "RAG_CHUNK_SIZE",
        "RAG_DATA_DIR",
        "RAG_EMBEDDING_MODEL",
        "RAG_GENERATION_MODEL",
        "RAG_INDEX_DIR",
        "RAG_LOG_LEVEL",
        "RAG_OVERLAP",
        "RAG_PROVIDER",
        "RAG_RESULTS_DIR",
        "RAG_SEED",
        "RAG_TEMPERATURE",
        "RAG_TOP_K",
    ):
        monkeypatch.delenv(variable, raising=False)


def test_settings_have_offline_safe_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Defaults select the fake provider and deterministic benchmark values."""
    monkeypatch.chdir(tmp_path)

    settings = load_settings()

    assert settings.provider == "fake"
    assert settings.chunk_size == 600
    assert settings.overlap == 100
    assert settings.top_k == 3
    assert settings.seed == 42
    assert settings.openai_api_key is None


def test_settings_precedence_is_cli_environment_yaml_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Higher-priority CLI and environment sources replace YAML and defaults."""
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "demo.yaml"
    config_path.write_text(
        "chunk_size: 300\noverlap: 50\ntop_k: 5\nlog_level: warning\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("RAG_CHUNK_SIZE", "450")

    settings = load_settings(config_path, {"top_k": 7})

    assert settings.chunk_size == 450
    assert settings.overlap == 50
    assert settings.top_k == 7
    assert settings.log_level == "WARNING"


def test_settings_reject_invalid_chunk_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Invalid overlap is converted into an actionable domain error."""
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ConfigurationError, match="overlap"):
        load_settings(overrides={"chunk_size": 100, "overlap": 100})


def test_yaml_cannot_supply_api_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Sensitive values are not accepted from a repository configuration file."""
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "unsafe.yaml"
    config_path.write_text("openai_api_key: not-a-real-key\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="must come from the environment"):
        load_settings(config_path)


def test_settings_type_is_public() -> None:
    """The public settings class remains directly usable by application modules."""
    assert Settings(provider="fake", _env_file=None).provider == "fake"
