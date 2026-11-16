"""Пересчёт аналитических витрин слоя ``mart`` (этап 3).

Полный цикл ``python -m etl.main --rebuild-marts --as-of 2026-10-26``:

1. применяет DDL core/mart идемпотентно (``00_core_schema.sql`` + ``01_mart_schema.sql``);
2. обновляет справочники core (города/кузова — через ``load_references``) и копирует
   source-таблицы ``core.price_body_types`` / ``core.price_dates``, чтобы весь расчёт
   витрины выполнялся в одной БД ``dwh`` чистым SQL (без FDW/кросс-БД JOIN);
3. атомарно перезаписывает ``mart.mart_route_card`` (``DELETE + INSERT`` в одной
   транзакции) эталонным SQL ``sql/dwh/02_mart_route_card.sql`` и
   ``mart.fact_vs_corridor`` эталонным SQL ``sql/dwh/03_fact_vs_corridor.sql``.

Повторный пересчёт за ту же дату снимка строго идемпотентен: справочники
заменяются целиком (``DELETE + INSERT``), витрина перезаписывается атомарно,
``PRIMARY KEY`` по зерну не даёт появиться дублям.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sqlalchemy import Engine, text

from common.config import Settings, get_settings, resolve_dwh_sql_dir
from common.logging import get_logger
from etl.db import apply_ddl, df_to_tuples, dwh_engine, replace_rows, source_engine
from etl.references import load_references

log = get_logger("etl.marts")

# Порядок колонок mart.mart_route_card — совпадает с SELECT из
# sql/dwh/02_mart_route_card.sql и с DDL ClickHouse (для тривиальной переливки).
MART_ROUTE_CARD_COLUMNS = (
    "month", "loading_city_gcid", "unloading_city_gcid",
    "loading_city_name", "unloading_city_name",
    "body_group", "weight_category", "volume_category",
    "p25_trip", "median_trip", "p75_trip", "avg_trip",
    "p25_km", "median_km", "p75_km",
    "trips_count", "carriers_count", "cv_rate",
    "mom_pct", "yoy_pct", "reliability_level",
)

# Порядок колонок mart.fact_vs_corridor — совпадает с SELECT из
# sql/dwh/03_fact_vs_corridor.sql.
FACT_VS_CORRIDOR_COLUMNS = (
    "price_id", "loading_date", "loading_city_gcid", "unloading_city_gcid",
    "loading_city_name", "unloading_city_name", "body_group", "weight_category",
    "distance", "actual_price_rub", "actual_price_km",
    "p25_trip", "median_trip", "p75_trip", "p25_km", "median_km", "p75_km",
    "trips_count", "carriers_count", "reliability_level",
    "trip_vs_median_pct", "km_vs_median_pct", "corridor_band",
)

PRICE_BODY_TYPES_COLUMNS = ("price_id", "body_type_id")
PRICE_DATES_COLUMNS = ("price_id", "price_date")

# В эталонном SQL дата снимка записана литералом; при --as-of он подставляется.
_AS_OF_LITERAL = "2026-10-26"


@dataclass
class MartStats:
    route_card_rows: int = 0
    fact_vs_corridor_rows: int = 0
    price_body_types_rows: int = 0
    price_dates_rows: int = 0


def parse_as_of(value: Optional[str]) -> str:
    """Вернуть дату снимка в каноническом виде ``YYYY-MM-DD`` (валидация ввода)."""
    if not value:
        return get_settings().as_of
    return dt.date.fromisoformat(value).isoformat()


def substitute_as_of(sql: str, as_of: str) -> str:
    """Подставить дату снимка вместо литерала в эталонном SQL (глобальная замена)."""
    return sql.replace(_AS_OF_LITERAL, as_of)


def _apply_ddl(dwh: Engine) -> None:
    ddl_dir = resolve_dwh_sql_dir()
    for name in ("00_core_schema.sql", "01_mart_schema.sql"):
        path = ddl_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Не найден DDL: {path}")
        apply_ddl(dwh, path.read_text(encoding="utf-8"))
    log.info("etl.marts.ddl.applied")


def load_price_body_types(src: Engine, dwh: Engine) -> int:
    """Скопировать public.price_body_types → core.price_body_types (атомарно)."""
    with src.connect() as conn:
        df = pd.read_sql(
            text(
                "SELECT price_id, body_type_id "
                "FROM public.price_body_types ORDER BY price_id, body_type_id"
            ),
            conn,
        )
    rows = df_to_tuples(df, PRICE_BODY_TYPES_COLUMNS)
    replace_rows(dwh, "core.price_body_types", PRICE_BODY_TYPES_COLUMNS, rows)
    return len(rows)


def load_price_dates(src: Engine, dwh: Engine) -> int:
    """Скопировать public.price_dates → core.price_dates (атомарно)."""
    with src.connect() as conn:
        df = pd.read_sql(
            text(
                "SELECT price_id, price_date "
                "FROM public.price_dates ORDER BY price_id, price_date"
            ),
            conn,
        )
    rows = df_to_tuples(df, PRICE_DATES_COLUMNS, date_columns=("price_date",))
    replace_rows(dwh, "core.price_dates", PRICE_DATES_COLUMNS, rows)
    return len(rows)


def _route_card_insert_sql(as_of: str) -> str:
    ddl_dir = resolve_dwh_sql_dir()
    path = ddl_dir / "02_mart_route_card.sql"
    if not path.exists():
        raise FileNotFoundError(f"Не найден эталонный SQL: {path}")
    select_sql = substitute_as_of(path.read_text(encoding="utf-8"), as_of)
    cols = ", ".join(MART_ROUTE_CARD_COLUMNS)
    return f"INSERT INTO mart.mart_route_card ({cols})\n{select_sql}"


def rebuild_route_card(dwh: Engine, as_of: str) -> int:
    """Атомарно перезаписать ``mart.mart_route_card`` и вернуть число строк."""
    sql = _route_card_insert_sql(as_of)
    with dwh.begin() as conn:
        conn.execute(text("DELETE FROM mart.mart_route_card"))
        conn.execute(text(sql))
    with dwh.connect() as conn:
        return int(conn.execute(text("SELECT count(*) FROM mart.mart_route_card")).scalar())


def _fact_vs_corridor_insert_sql(as_of: str) -> str:
    ddl_dir = resolve_dwh_sql_dir()
    path = ddl_dir / "03_fact_vs_corridor.sql"
    if not path.exists():
        raise FileNotFoundError(f"Не найден эталонный SQL: {path}")
    select_sql = substitute_as_of(path.read_text(encoding="utf-8"), as_of)
    cols = ", ".join(FACT_VS_CORRIDOR_COLUMNS)
    return f"INSERT INTO mart.fact_vs_corridor ({cols})\n{select_sql}"


def rebuild_fact_vs_corridor(dwh: Engine, as_of: str) -> int:
    """Атомарно перезаписать ``mart.fact_vs_corridor`` и вернуть число строк."""
    sql = _fact_vs_corridor_insert_sql(as_of)
    with dwh.begin() as conn:
        conn.execute(text("DELETE FROM mart.fact_vs_corridor"))
        conn.execute(text(sql))
    with dwh.connect() as conn:
        return int(conn.execute(text("SELECT count(*) FROM mart.fact_vs_corridor")).scalar())


def rebuild_marts(as_of: Optional[str] = None, settings: Optional[Settings] = None) -> MartStats:
    """Полный идемпотентный пересчёт витрин mart (точка входа ``--rebuild-marts``)."""
    settings = settings or get_settings()
    as_of = parse_as_of(as_of)
    src = source_engine(settings)
    dwh = dwh_engine(settings)

    _apply_ddl(dwh)

    log.info("etl.marts.references.start", as_of=as_of)
    load_references(src, dwh)
    pbt_rows = load_price_body_types(src, dwh)
    pd_rows = load_price_dates(src, dwh)
    log.info(
        "etl.marts.references.loaded",
        price_body_types=pbt_rows,
        price_dates=pd_rows,
    )

    log.info("etl.marts.route_card.start", as_of=as_of)
    route_rows = rebuild_route_card(dwh, as_of)
    log.info("etl.marts.route_card.done", as_of=as_of, route_card_rows=route_rows)

    log.info("etl.marts.fact_vs_corridor.start", as_of=as_of)
    fact_rows = rebuild_fact_vs_corridor(dwh, as_of)
    log.info("etl.marts.fact_vs_corridor.done", as_of=as_of, fact_vs_corridor_rows=fact_rows)

    return MartStats(
        route_card_rows=route_rows,
        fact_vs_corridor_rows=fact_rows,
        price_body_types_rows=pbt_rows,
        price_dates_rows=pd_rows,
    )
