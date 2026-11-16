"""Идемпотентная загрузка ``core.prices`` и ``core.dq_quarantine_prices``.

* ``core.prices`` — ``UPSERT`` по естественному ключу ``price_id``
  (``INSERT ... ON CONFLICT DO UPDATE``). Повторный прогон за тот же период
  не задваивает строки и не меняет суммы/перцентили.
* карантин — атомарная перезапись окна ``loading_date``:
  ``DELETE`` строк окна + ``INSERT`` (``ON CONFLICT DO NOTHING`` для защиты от
  дублей причин при перекрывающихся окнах).
"""
from __future__ import annotations

import datetime as dt
from typing import List, Tuple

import pandas as pd
from sqlalchemy import Engine, text

from etl import dq
from etl.db import df_to_tuples, executemany

# core.prices: CORE_COLUMNS + etl_batch_id (etl_loaded_at заполняет СУБД).
PRICE_INSERT_COLUMNS: Tuple[str, ...] = dq.CORE_COLUMNS + ("etl_batch_id",)

PRICE_DATE_COLUMNS: Tuple[str, ...] = ("loading_date", "nearest_rate_date", "end_loading_date")
PRICE_TS_COLUMNS: Tuple[str, ...] = ("created_at", "updated_at")

QUARANTINE_INSERT_COLUMNS: Tuple[str, ...] = dq.QUARANTINE_COLUMNS + ("detected_at", "batch_id")
QUARANTINE_DATE_COLUMNS: Tuple[str, ...] = ("nearest_rate_date", "loading_date")

_PRICE_UPSERT_SQL = """
INSERT INTO core.prices (
    price_id, loading_date, created_at, updated_at, subscriber_id, organization_id,
    loading_city_gcid, unloading_city_gcid, distance, total_weight, total_volume,
    body_type, price, currency_code, nearest_rate_date, rate_value, price_rub,
    complex, price_type, is_fact, outlier_mark, weight_category, volume_category,
    source, item_id, meta_item_id, end_loading_date, etl_batch_id
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
          %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (price_id) DO UPDATE SET
    loading_date        = EXCLUDED.loading_date,
    created_at          = EXCLUDED.created_at,
    updated_at          = EXCLUDED.updated_at,
    subscriber_id       = EXCLUDED.subscriber_id,
    organization_id     = EXCLUDED.organization_id,
    loading_city_gcid   = EXCLUDED.loading_city_gcid,
    unloading_city_gcid = EXCLUDED.unloading_city_gcid,
    distance            = EXCLUDED.distance,
    total_weight        = EXCLUDED.total_weight,
    total_volume        = EXCLUDED.total_volume,
    body_type           = EXCLUDED.body_type,
    price               = EXCLUDED.price,
    currency_code       = EXCLUDED.currency_code,
    nearest_rate_date   = EXCLUDED.nearest_rate_date,
    rate_value          = EXCLUDED.rate_value,
    price_rub           = EXCLUDED.price_rub,
    complex             = EXCLUDED.complex,
    price_type          = EXCLUDED.price_type,
    is_fact             = EXCLUDED.is_fact,
    outlier_mark        = EXCLUDED.outlier_mark,
    weight_category     = EXCLUDED.weight_category,
    volume_category     = EXCLUDED.volume_category,
    source              = EXCLUDED.source,
    item_id             = EXCLUDED.item_id,
    meta_item_id        = EXCLUDED.meta_item_id,
    end_loading_date    = EXCLUDED.end_loading_date,
    etl_batch_id        = EXCLUDED.etl_batch_id,
    etl_loaded_at       = now()
"""

_QUARANTINE_INSERT_SQL = """
INSERT INTO core.dq_quarantine_prices (
    price_id, reason_code, raw_price, currency_code, nearest_rate_date, loading_date,
    loading_city_gcid, unloading_city_gcid, distance, total_weight, total_volume,
    body_type, price_type, detected_at, batch_id
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (price_id, reason_code) DO NOTHING
"""


def _core_rows(valid_df: pd.DataFrame, batch_id: str) -> List[Tuple]:
    df = valid_df.rename(columns={"id": "price_id"}).copy()
    df["etl_batch_id"] = batch_id
    df = df[list(PRICE_INSERT_COLUMNS)]
    return df_to_tuples(
        df,
        PRICE_INSERT_COLUMNS,
        date_columns=PRICE_DATE_COLUMNS,
        ts_columns=PRICE_TS_COLUMNS,
    )


def upsert_prices(dwh_engine: Engine, valid_df: pd.DataFrame, batch_id: str) -> int:
    """Записать валидные строки в ``core.prices`` (UPSERT по price_id)."""
    rows = _core_rows(valid_df, batch_id)
    executemany(dwh_engine, _PRICE_UPSERT_SQL, rows)
    return len(rows)


def load_quarantine(
    dwh_engine: Engine,
    quarantine_df: pd.DataFrame,
    *,
    batch_id: str,
    start_date: str,
    end_date: str,
    detected_at: dt.datetime,
) -> int:
    """Атомарно перезаписать карантин для окна ``loading_date`` и вернуть число строк."""
    rows: List[Tuple] = []
    if not quarantine_df.empty:
        q = quarantine_df.copy()
        q["detected_at"] = detected_at
        q["batch_id"] = batch_id
        q = q[list(QUARANTINE_INSERT_COLUMNS)]
        rows = df_to_tuples(q, QUARANTINE_INSERT_COLUMNS, date_columns=QUARANTINE_DATE_COLUMNS)

    with dwh_engine.begin() as conn:
        conn.execute(
            text(
                "DELETE FROM core.dq_quarantine_prices "
                "WHERE loading_date >= :start_date AND loading_date <= :end_date"
            ),
            {"start_date": start_date, "end_date": end_date},
        )
        if rows:
            with conn.connection.cursor() as cur:
                cur.executemany(_QUARANTINE_INSERT_SQL, rows)
    return len(rows)
