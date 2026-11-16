"""Оркестрация ETL: извлечение -> DQ/очистка -> конвертация -> обогащение -> загрузка.

Пайплайн читает операционный источник батчами (``chunksize``), валидирует каждую
порцию, конвертирует валюту в RUB по точному курсу ``nearest_rate_date``,
заполняет категории груза и идемпотентно пишет результат в ``core``. Отбракованные
строки не теряются: попадают в карантин ``core.dq_quarantine_prices`` с кодом
причины и агрегируются в логе.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional

import pandas as pd
from sqlalchemy import Engine, text

from common.config import Settings, get_settings, resolve_dwh_sql_dir
from common.logging import get_logger
from etl import dq
from etl.db import apply_ddl, dwh_engine, source_engine
from etl.loader import load_quarantine, upsert_prices
from etl.references import load_references

log = get_logger("etl.pipeline")

SOURCE_COLUMNS = (
    "id", "created_at", "updated_at", "subscriber_id", "organization_id",
    "loading_date", "loading_city_gcid", "unloading_city_gcid", "distance",
    "total_weight", "total_volume", "body_type", "price", "currency_code",
    "complex", "price_type", "item_id", "is_fact", "nearest_rate_date",
    "meta_item_id", "end_loading_date", "source", "outlier_mark",
    "weight_category", "volume_category",
)

_SELECT_SQL = (
    "SELECT " + ", ".join(SOURCE_COLUMNS) + " "
    "FROM public.prices "
    "WHERE loading_date >= :start_date AND loading_date <= :end_date "
    "ORDER BY id"
)


@dataclass
class EtlStats:
    rows_read: int = 0
    rows_loaded: int = 0
    rows_quarantined: int = 0
    reason_counts: Dict[str, int] = field(
        default_factory=lambda: {code: 0 for code in dq.REASON_CODES}
    )


def _read_chunks(engine: Engine, start_date: str, end_date: str, batch_size: int) -> Iterator[pd.DataFrame]:
    return pd.read_sql(
        text(_SELECT_SQL),
        engine,
        params={"start_date": start_date, "end_date": end_date},
        chunksize=batch_size,
    )


def _apply_core_ddl(dwh: Engine) -> None:
    ddl_path = resolve_dwh_sql_dir() / "00_core_schema.sql"
    if not ddl_path.exists():
        raise FileNotFoundError(f"Не найден DDL core-слоя: {ddl_path}")
    apply_ddl(dwh, ddl_path.read_text(encoding="utf-8"))
    log.info("etl.core_ddl.applied", path=str(ddl_path))


def run(
    start_date: str,
    end_date: str,
    *,
    batch_size: int = 10000,
    settings: Optional[Settings] = None,
) -> EtlStats:
    """Выполнить ETL за окно ``[start_date, end_date]`` (включительно)."""
    settings = settings or get_settings()
    src = source_engine(settings)
    dwh = dwh_engine(settings)
    batch_id = f"{start_date}_{end_date}"
    stats = EtlStats()
    quarantine_parts: list[pd.DataFrame] = []

    _apply_core_ddl(dwh)

    log.info(
        "etl.references.start",
        start_date=start_date,
        end_date=end_date,
        batch_id=batch_id,
    )
    city_gcids, rates_df = load_references(src, dwh)
    rate_map = dq.build_rate_map(rates_df)
    log.info(
        "etl.references.loaded",
        cities=len(city_gcids),
        exchange_rates=len(rates_df),
    )

    log.info(
        "etl.extract.start",
        start_date=start_date,
        end_date=end_date,
        batch_size=batch_size,
    )
    for chunk in _read_chunks(src, start_date, end_date, batch_size):
        stats.rows_read += len(chunk)
        valid, quarantine = dq.evaluate_dq(chunk, rate_map, city_gcids)

        if not valid.empty:
            stats.rows_loaded += upsert_prices(dwh, valid, batch_id)

        if not quarantine.empty:
            stats.rows_quarantined += len(quarantine)
            quarantine_parts.append(quarantine)

    # Атомарно перезаписать карантин окна (DELETE + INSERT в одной транзакции).
    if quarantine_parts:
        quarantine_df = pd.concat(quarantine_parts, ignore_index=True)
    else:
        quarantine_df = pd.DataFrame(columns=list(dq.QUARANTINE_COLUMNS))
    load_quarantine(
        dwh,
        quarantine_df,
        batch_id=batch_id,
        start_date=start_date,
        end_date=end_date,
        detected_at=dt.datetime.now(dt.timezone.utc),
    )

    # Агрегируем статистику отсева по кодам (для лога и проверок).
    for code in dq.REASON_CODES:
        if not quarantine_df.empty:
            stats.reason_counts[code] = int((quarantine_df["reason_code"] == code).sum())
        else:
            stats.reason_counts[code] = 0

    log.info(
        "etl.done",
        start_date=start_date,
        end_date=end_date,
        rows_read=stats.rows_read,
        rows_loaded=stats.rows_loaded,
        rows_quarantined=stats.rows_quarantined,
        reason_counts=stats.reason_counts,
    )
    return stats
