"""Низкоуровневые операции с БД: движки, применение DDL, батч-вставки.

Весь доступ к СУБД — через этот модуль. Записи выполняются батчами
(``executemany`` / векторизованная вставка), построчные ``INSERT`` в цикле
запрещены требованием задания.
"""
from __future__ import annotations

import datetime as dt
import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import Connection

from common.config import Settings, get_settings


def make_engine(dsn: str) -> Engine:
    return create_engine(dsn, future=True)


def source_engine(settings: Optional[Settings] = None) -> Engine:
    return make_engine((settings or get_settings()).source_dsn)


def dwh_engine(settings: Optional[Settings] = None) -> Engine:
    return make_engine((settings or get_settings()).dwh_dsn)


def apply_ddl(engine: Engine, ddl: str) -> None:
    """Идемпотентно применить DDL (``CREATE ... IF NOT EXISTS``)."""
    with engine.begin() as conn:
        conn.execute(text(ddl))


def _cell(v, *, as_date: bool, as_ts: bool):
    """Привести одну ячейку DataFrame к нативному Python-значению для psycopg."""
    if v is None or v is pd.NaT or v is pd.NA:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, pd.Timestamp):
        v = v.to_pydatetime()
    if isinstance(v, dt.datetime):
        return v.date() if as_date else v
    if isinstance(v, dt.date):
        return dt.datetime(v.year, v.month, v.day) if as_ts else v
    return v


def df_to_tuples(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    date_columns: Sequence[str] = (),
    ts_columns: Sequence[str] = (),
) -> List[Tuple]:
    """Преобразовать DataFrame в список кортежей нативных Python-значений.

    Используется для батч-вставки через ``executemany``. Нативные типы обязательны,
    т.к. psycopg3 не адаптирует numpy-скаляры автоматически.
    """
    date_cols = set(date_columns)
    ts_cols = set(ts_columns)
    ordered = list(columns)
    rows: List[Tuple] = []
    for rec in df[ordered].itertuples(index=False, name=None):
        rows.append(
            tuple(
                _cell(v, as_date=col in date_cols, as_ts=col in ts_cols)
                for col, v in zip(ordered, rec)
            )
        )
    return rows


def executemany(engine: Engine, sql: str, rows: Iterable[Tuple]) -> None:
    """Векторизованная вставка батчем (psycopg3 executemany -> pipeline)."""
    data = list(rows)
    if not data:
        return
    with engine.begin() as conn:
        with conn.connection.cursor() as cur:
            cur.executemany(sql, data)


def execute(conn: Connection, sql: str, params: Optional[dict] = None) -> None:
    conn.execute(text(sql), params or {})


def replace_rows(
    engine: Engine,
    table: str,
    columns: Sequence[str],
    rows: List[Tuple],
) -> None:
    """Атомарно заменить содержимое таблицы (``DELETE`` + ``executemany`` в одной tx).

    ``table``/``columns`` — константы из кода (не пользовательский ввод).
    """
    cols = list(columns)
    placeholders = ", ".join(["%s"] * len(cols))
    insert_sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
    with engine.begin() as conn:
        conn.execute(text(f"DELETE FROM {table}"))
        if rows:
            with conn.connection.cursor() as cur:
                cur.executemany(insert_sql, rows)
