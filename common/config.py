"""Конфигурация приложения — только из переменных окружения (файл `.env`)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv

load_dotenv()


def resolve_dwh_sql_dir() -> Path:
    """Каталог DDL хранилища (``sql/dwh``) для локального и Docker/Airflow запуска.

    Порядок разрешения:
    1. переменная окружения ``DWH_SQL_DIR`` (явное переопределение);
    2. ``<repo>/sql/dwh`` (локально и в ``Dockerfile.etl`` образе);
    3. ``/opt/sql/dwh`` (монтирование в контейнерах Airflow).
    """
    env = os.getenv("DWH_SQL_DIR")
    if env:
        return Path(env)
    repo_root = Path(__file__).resolve().parent.parent
    for candidate in (repo_root / "sql" / "dwh", Path("/opt/sql/dwh")):
        if (candidate / "00_core_schema.sql").exists():
            return candidate
    return repo_root / "sql" / "dwh"


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    source_db: str
    dwh_db: str
    db_host: str
    db_port: int
    db_user: str
    db_password: str
    as_of: str
    random_seed: float
    ml_random_seed: int
    ml_point_loss: str
    ml_depth: int
    ml_l2_leaf_reg: float
    ml_learning_rate: float
    log_level: str

    def dsn(self, database: str) -> str:
        return (
            f"postgresql+psycopg://{quote_plus(self.db_user)}:{quote_plus(self.db_password)}"
            f"@{self.db_host}:{self.db_port}/{database}"
        )

    @property
    def source_dsn(self) -> str:
        return self.dsn(self.source_db)

    @property
    def dwh_dsn(self) -> str:
        return self.dsn(self.dwh_db)


def get_settings() -> Settings:
    """Прочитать настройки из переменных окружения (с dev-дефолтами)."""
    return Settings(
        source_db=_env("POSTGRES_DB", "deals_bigdata"),
        dwh_db=_env("DWH_DB", "dwh"),
        db_host=_env("DB_HOST", "localhost"),
        db_port=int(_env("DB_PORT", "5432")),
        db_user=_env("DWH_USER", "dwh_user"),
        db_password=_env("DWH_PASSWORD", "dwh_password"),
        as_of=_env("AS_OF", "2026-10-26"),
        random_seed=float(_env("RANDOM_SEED", "0.20261026")),
        ml_random_seed=int(_env("ML_RANDOM_SEED", "42")),
        ml_point_loss=_env("ML_POINT_LOSS", "RMSE"),
        ml_depth=int(_env("ML_DEPTH", "6")),
        ml_l2_leaf_reg=float(_env("ML_L2_LEAF_REG", "3.0")),
        ml_learning_rate=float(_env("ML_LEARNING_RATE", "0.03")),
        log_level=_env("LOG_LEVEL", "INFO"),
    )
