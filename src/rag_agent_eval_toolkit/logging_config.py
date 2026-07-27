"""Structured, non-secret application logging helpers."""

import json
import logging
from datetime import UTC, datetime
from typing import Any

from rag_agent_eval_toolkit.exceptions import ConfigurationError

_LOGGER_NAME = "rag_agent_eval_toolkit"
_SENSITIVE_NAME_FRAGMENTS = ("api_key", "authorization", "password", "secret", "token")
_STANDARD_LOG_RECORD_FIELDS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)


class JsonFormatter(logging.Formatter):
    """Format log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        """Return a JSON representation of a log record."""
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _STANDARD_LOG_RECORD_FIELDS or key.startswith("_"):
                continue
            payload[key] = (
                "[REDACTED]"
                if any(fragment in key.lower() for fragment in _SENSITIVE_NAME_FRAGMENTS)
                else value
            )
        if record.exc_info is not None and record.exc_info[0] is not None:
            payload["exception_type"] = record.exc_info[0].__name__
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO", *, json_output: bool = True) -> logging.Logger:
    """Configure and return the toolkit logger without changing global handlers."""
    normalised_level = level.strip().upper()
    numeric_level = logging.getLevelNamesMapping().get(normalised_level)
    if numeric_level is None:
        raise ConfigurationError(f"Unsupported log level: {level}")

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(numeric_level)
    logger.propagate = False
    handler = logging.StreamHandler()
    if json_output:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    logger.handlers.clear()
    logger.addHandler(handler)
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the toolkit logger or one of its named children."""
    if name is None:
        return logging.getLogger(_LOGGER_NAME)
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")


__all__ = ["JsonFormatter", "configure_logging", "get_logger"]
