"""Typed application configuration with explicit source precedence."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self, cast

import yaml
from pydantic import Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag_agent_eval_toolkit.exceptions import ConfigurationError

_SENSITIVE_FIELDS = frozenset({"openai_api_key"})


class Settings(BaseSettings):
    """Validated settings loaded from safe defaults, environment variables, and ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        env_prefix="RAG_",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    provider: Literal["fake", "openai"] = "fake"
    data_dir: Path = Path("sample_data/fictional_ev_support")
    index_dir: Path = Path(".rag_eval/index")
    results_dir: Path = Path("results")
    embedding_model: str = "text-embedding-3-small"
    generation_model: str = "gpt-4.1-mini"
    log_level: str = "INFO"
    chunk_size: int = Field(default=600, gt=0)
    overlap: int = Field(default=100, ge=0)
    top_k: int = Field(default=3, gt=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    prompt_version: str = Field(default="v1", min_length=1)
    seed: int = 42
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="OPENAI_API_KEY",
        repr=False,
    )

    @field_validator("embedding_model", "generation_model", "prompt_version")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        """Reject empty model and prompt-version identifiers."""
        normalised = value.strip()
        if not normalised:
            raise ValueError("value must not be empty")
        return normalised

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        """Normalise and validate the configured logging level."""
        normalised = value.strip().upper()
        if normalised not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
            raise ValueError("log_level must be one of CRITICAL, ERROR, WARNING, INFO, or DEBUG")
        return normalised

    @model_validator(mode="after")
    def validate_chunk_configuration(self) -> Self:
        """Ensure overlap remains smaller than the configured chunk size."""
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        return self


def load_settings(
    config_path: Path | None = None,
    overrides: Mapping[str, object] | None = None,
) -> Settings:
    """Load settings using CLI overrides, environment, YAML, then defaults.

    ``overrides`` represents explicitly supplied CLI values. Entries whose value is
    ``None`` are ignored so an omitted CLI option does not mask a lower-priority source.
    API keys are intentionally accepted only through environment or ``.env`` sources.
    """

    try:
        environment_settings = Settings()
        environment_fields = environment_settings.model_fields_set
        resolved: dict[str, Any] = environment_settings.model_dump()

        file_values = _load_yaml_config(config_path) if config_path is not None else {}
        _validate_keys(file_values, source="configuration file")
        _reject_sensitive_values(file_values, source="configuration file")
        for key, value in file_values.items():
            if key not in environment_fields:
                resolved[key] = value

        cli_values = {key: value for key, value in (overrides or {}).items() if value is not None}
        _validate_keys(cli_values, source="CLI overrides")
        _reject_sensitive_values(cli_values, source="CLI overrides")
        resolved.update(cli_values)
        return Settings.model_validate(resolved)
    except ConfigurationError:
        raise
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid application configuration: {exc}") from exc


def _load_yaml_config(config_path: Path) -> dict[str, object]:
    resolved_path = config_path.expanduser()
    if resolved_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ConfigurationError(
            f"Configuration file must use a .yaml or .yml extension: {config_path}"
        )
    try:
        content = resolved_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ConfigurationError(f"Unable to read configuration file {config_path}: {exc}") from exc

    try:
        loaded = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"Invalid YAML in configuration file {config_path}: {exc}"
        ) from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict) or not all(isinstance(key, str) for key in loaded):
        raise ConfigurationError("Configuration file must contain a mapping with string keys")
    return cast(dict[str, object], loaded)


def _validate_keys(values: Mapping[str, object], *, source: str) -> None:
    unknown = sorted(set(values).difference(Settings.model_fields))
    if unknown:
        joined = ", ".join(unknown)
        raise ConfigurationError(f"Unknown setting(s) in {source}: {joined}")


def _reject_sensitive_values(values: Mapping[str, object], *, source: str) -> None:
    supplied = sorted(_SENSITIVE_FIELDS.intersection(values))
    if supplied:
        joined = ", ".join(supplied)
        raise ConfigurationError(
            f"Sensitive setting(s) must come from the environment, not {source}: {joined}"
        )


__all__ = ["Settings", "load_settings"]
