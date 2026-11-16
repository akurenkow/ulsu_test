"""Структурированное (JSON) логирование, общее для ETL и ML пайплайнов."""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import structlog

_configured = False


def configure_logging(level: Optional[str] = None, *, json_output: bool = True) -> None:
    """Настроить structlog один раз на процесс (идемпотентно)."""
    global _configured
    if _configured:
        return

    level_name = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    py_level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=py_level)

    renderer = (
        structlog.processors.JSONRenderer(ensure_ascii=False)
        if json_output
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(py_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """Вернуть структурированный логгер (настраивает логирование при первом вызове)."""
    configure_logging()
    return structlog.get_logger(name)
