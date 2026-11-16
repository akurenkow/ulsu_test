"""Загрузка справочников источника в слой ``core`` (идемпотентно).

Справочники малы и детерминированы, поэтому используется атомарная замена
(``DELETE`` + батч-вставка в одной транзакции). Результат не зависит от числа
повторных запусков.

* ``core.cities`` — дедуплицированный справочник: одна строка на ``gc_id``
  (минимальный ``cities.id``), что устраняет размножение сделок на этапе 3.
* ``core.body_types`` — копия справочника кузовов 1С с маппингом ``std_id``.
* ``core.exchange_rates`` — копия курсов ЦБ для точной конвертации валют.
* ``core.body_groups`` — статический справочник (seeded в DDL).
"""
from __future__ import annotations

from typing import List, Set, Tuple

import pandas as pd
from sqlalchemy import Engine, text
from sqlalchemy.engine import Connection

from etl.db import df_to_tuples, replace_rows

CITIES_COLUMNS: Tuple[str, ...] = (
    "id", "full_name_local", "full_name_gc", "gc_id", "name",
    "longitude", "latitude", "created_at", "updated_at",
)
BODY_TYPES_COLUMNS: Tuple[str, ...] = ("id", "name", "id_from_1c", "std_id")
EXCHANGE_RATES_COLUMNS: Tuple[str, ...] = ("id", "num_code", "value", "setting_date")


def read_cities(conn: Connection) -> pd.DataFrame:
    df = pd.read_sql(
        text(
            "SELECT id, full_name_local, full_name_gc, gc_id, name, "
            "       longitude, latitude, created_at, updated_at "
            "FROM public.cities"
        ),
        conn,
    )
    # Дедупликация: минимальный id на gc_id (правило задания/этапа 3).
    return df.sort_values(["gc_id", "id"]).drop_duplicates("gc_id", keep="first").reset_index(drop=True)


def read_body_types(conn: Connection) -> pd.DataFrame:
    df = pd.read_sql(
        text('SELECT id, name, "id_from_1C" AS id_from_1c, std_id FROM public.body_types'),
        conn,
    )
    return df.sort_values("id").reset_index(drop=True)


def read_exchange_rates(conn: Connection) -> pd.DataFrame:
    df = pd.read_sql(
        text(
            "SELECT id, num_code, value, setting_date "
            "FROM public.exchange_rates ORDER BY num_code, setting_date"
        ),
        conn,
    )
    return df.reset_index(drop=True)


def load_cities(source_engine: Engine, dwh_engine: Engine) -> Set[int]:
    with source_engine.connect() as conn:
        cities = read_cities(conn)
    rows = df_to_tuples(
        cities, CITIES_COLUMNS,
        date_columns=(), ts_columns=("created_at", "updated_at"),
    )
    replace_rows(dwh_engine, "core.cities", CITIES_COLUMNS, rows)
    return set(int(g) for g in cities["gc_id"])


def load_body_types(source_engine: Engine, dwh_engine: Engine) -> None:
    with source_engine.connect() as conn:
        body_types = read_body_types(conn)
    rows = df_to_tuples(body_types, BODY_TYPES_COLUMNS)
    replace_rows(dwh_engine, "core.body_types", BODY_TYPES_COLUMNS, rows)


def load_exchange_rates(source_engine: Engine, dwh_engine: Engine) -> pd.DataFrame:
    with source_engine.connect() as conn:
        rates = read_exchange_rates(conn)
    rows = df_to_tuples(rates, EXCHANGE_RATES_COLUMNS, date_columns=("setting_date",))
    replace_rows(dwh_engine, "core.exchange_rates", EXCHANGE_RATES_COLUMNS, rows)
    return rates


def load_references(source_engine: Engine, dwh_engine: Engine) -> Tuple[Set[int], pd.DataFrame]:
    """Загрузить справочники и вернуть ``(city_gcids, exchange_rates)``."""
    city_gcids = load_cities(source_engine, dwh_engine)
    load_body_types(source_engine, dwh_engine)
    rates = load_exchange_rates(source_engine, dwh_engine)
    return city_gcids, rates
