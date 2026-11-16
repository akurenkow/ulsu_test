"""Фикстуры и маркеры для автотестов (этап 5).

Юнит-тесты (`test_dq.py`, `test_marts.py`, `test_ml_features.py`,
`test_ml_split.py`) БД не требуют и работают локально.

Интеграционные тесты (маркер `db`: `test_idempotency.py`, `test_sql_smoke.py`)
требуют поднятый PostgreSQL с источником и хранилищем (`docker compose up -d`)
и **автоматически пропускаются**, если БД недоступна (например, локальный прогон
без Docker). Параметры подключения читаются из переменных окружения —
см. `.env.example` (`DB_HOST`, `DB_PORT`, `POSTGRES_DB`, `DWH_DB`, `DWH_USER`,
`DWH_PASSWORD`).
"""
from __future__ import annotations

import pytest
from sqlalchemy import Engine, text

from common.config import get_settings
from etl.db import dwh_engine, source_engine


def _pg_reachable(engine: Engine) -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def settings():
    return get_settings()


@pytest.fixture(scope="session")
def pg_source(settings):
    engine = source_engine(settings)
    if not _pg_reachable(engine):
        pytest.skip(
            "Источник PostgreSQL недоступен — поднимите инфраструктуру: "
            "`docker compose up -d`."
        )
    yield engine
    engine.dispose()


@pytest.fixture(scope="session")
def pg_dwh(settings):
    engine = dwh_engine(settings)
    if not _pg_reachable(engine):
        pytest.skip(
            "Хранилище PostgreSQL недоступно — поднимите инфраструктуру: "
            "`docker compose up -d`."
        )
    yield engine
    engine.dispose()
