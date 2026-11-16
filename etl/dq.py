"""Правила валидации, конвертации валют и обогащения категорий (Data Quality).

Модуль не выполняет I/O: чистые функции над ``pandas.DataFrame`` / скалярными
значениями, чтобы правила было просто покрывать юнит-тестами (этап 5).

Семантика причин карантина соответствует ``specs/step_02_etl_data_quality.md``:

+---------------------------+----------------------+--------------------------------------+
| Правило                   | reason_code          | Условие                              |
+===========================+======================+======================================+
| невалидная цена           | INVALID_PRICE        | price <= 0 (или NULL)                |
| RUB вне source-диапазона | PRICE_BELOW_SOURCE_RANGE | currency=643 и 0 < price <= 1000 |
| невалидная дистанция      | INVALID_DISTANCE     | distance <= 0                        |
| отрицательный вес         | INVALID_WEIGHT       | total_weight < 0                     |
| отрицательный объём       | INVALID_VOLUME       | total_volume < 0                     |
| город вне справочника     | MISSING_CITY         | loading/unloading_city_gcid нет в cities |
| нет точного курса         | MISSING_RATE         | валюта != 643 и курс по nearest_rate_date не найден |
+---------------------------+----------------------+--------------------------------------+

``PRICE_BELOW_SOURCE_RANGE`` применяется только к RUB-ставкам ``0 < price <= 1000``:
нулевые/отрицательные цены уже покрыты ``INVALID_PRICE`` (строки не учитываются
дважды одной и той же причиной).
"""
from __future__ import annotations

import datetime as dt
from decimal import ROUND_HALF_UP, Decimal
from typing import Dict, Iterable, Mapping, Set, Tuple

import numpy as np
import pandas as pd

# Коды причин карантина.
INVALID_PRICE = "INVALID_PRICE"
PRICE_BELOW_SOURCE_RANGE = "PRICE_BELOW_SOURCE_RANGE"
INVALID_DISTANCE = "INVALID_DISTANCE"
INVALID_WEIGHT = "INVALID_WEIGHT"
INVALID_VOLUME = "INVALID_VOLUME"
MISSING_CITY = "MISSING_CITY"
MISSING_RATE = "MISSING_RATE"

REASON_CODES: Tuple[str, ...] = (
    INVALID_PRICE,
    PRICE_BELOW_SOURCE_RANGE,
    INVALID_DISTANCE,
    INVALID_WEIGHT,
    INVALID_VOLUME,
    MISSING_CITY,
    MISSING_RATE,
)

RUB = "643"

# Колонки карантина (в порядке DDL core.dq_quarantine_prices, без id/detected_at/batch_id).
QUARANTINE_COLUMNS: Tuple[str, ...] = (
    "price_id",
    "reason_code",
    "raw_price",
    "currency_code",
    "nearest_rate_date",
    "loading_date",
    "loading_city_gcid",
    "unloading_city_gcid",
    "distance",
    "total_weight",
    "total_volume",
    "body_type",
    "price_type",
)

# Колонки core.prices (в порядке DDL, без etl_loaded_at — заполняется в loader).
CORE_COLUMNS: Tuple[str, ...] = (
    "price_id",
    "loading_date",
    "created_at",
    "updated_at",
    "subscriber_id",
    "organization_id",
    "loading_city_gcid",
    "unloading_city_gcid",
    "distance",
    "total_weight",
    "total_volume",
    "body_type",
    "price",
    "currency_code",
    "nearest_rate_date",
    "rate_value",
    "price_rub",
    "complex",
    "price_type",
    "is_fact",
    "outlier_mark",
    "weight_category",
    "volume_category",
    "source",
    "item_id",
    "meta_item_id",
    "end_loading_date",
)


def round_half_up(value: float, ndigits: int = 1) -> float:
    """Округление «половина вверх» (как PostgreSQL ``round(numeric, n)``).

    Python ``round`` использует банковское округление и чувствителен к
    двоичному представлению float, поэтому здесь — Decimal от ``str(value)``.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    quantum = Decimal(1).scaleb(-ndigits)
    return float(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def classify_weight_category(total_weight: float) -> str:
    """Весовая категория по ``total_weight`` (округление до 0.1 т)."""
    w = round_half_up(float(total_weight), 1)
    if w >= 30.0:
        return "G20T"
    if w >= 10.0:
        return "FTL"
    if w >= 5.0:
        return "10T"
    if w >= 2.5:
        return "5T"
    if w >= 0.5:
        return "L5T"
    return "OTH"


def classify_volume_category(total_volume: float) -> str:
    """Объёмная категория: FTL при 0 < V <= 120, иначе OTH."""
    v = float(total_volume)
    return "FTL" if 0.0 < v <= 120.0 else "OTH"


def _canon_date(v) -> object:
    """Привести значение даты к каноническому ``datetime.date`` (или None)."""
    if v is None or v is pd.NaT or v is pd.NA:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, pd.Timestamp):
        return v.date()
    if isinstance(v, dt.datetime):
        return v.date()
    if isinstance(v, dt.date):
        return v
    return v


def build_rate_map(exchange_rates: pd.DataFrame) -> Dict[Tuple[str, object], float]:
    """Построить словарь точных курсов: ``(num_code, setting_date) -> value``."""
    rate_map: Dict[Tuple[str, object], float] = {}
    for num_code, setting_date, value in zip(
        exchange_rates["num_code"],
        exchange_rates["setting_date"],
        exchange_rates["value"],
    ):
        rate_map[(str(num_code), _canon_date(setting_date))] = float(value)
    return rate_map


def _lookup_rate(df: pd.DataFrame, rate_map: Mapping[Tuple[str, object], float]) -> pd.Series:
    """Векторный lookup курса по ``(currency_code, nearest_rate_date)``.

    Для RUB курс принудительно 1.0 (даже при ``nearest_rate_date IS NULL``).
    """
    keys = [
        (str(cc), _canon_date(nd))
        for cc, nd in zip(df["currency_code"], df["nearest_rate_date"])
    ]
    rate = pd.Series([rate_map.get(k, np.nan) for k in keys], index=df.index, dtype="float64")
    rate.loc[df["currency_code"] == RUB] = 1.0
    return rate


def _fill_missing_category(df: pd.DataFrame, column: str, source_column: str, classifier) -> None:
    """Заполнить пропуски категории, не перезаписывая уже заполненные значения."""
    missing = df[column].isna()
    if missing.any():
        df.loc[missing, column] = df.loc[missing, source_column].map(classifier)


def evaluate_dq(
    df: pd.DataFrame,
    rate_map: Mapping[Tuple[str, object], float],
    city_gcids: Iterable[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Проверить строки, сконвертировать валюту и обогатить категории.

    Параметры
    ----------
    df : DataFrame со столбцами операционного источника ``prices`` (``id`` — ключ).
    rate_map : точные курсы ``(num_code, setting_date) -> value``.
    city_gcids : множество ``gc_id`` справочника городов.

    Возвращает
    ----------
    (valid, quarantine) : кортеж DataFrame.
        * valid — валидные строки с ``rate_value``, ``price_rub`` и категориями;
        * quarantine — по одной строке на ``(price_id, reason_code)``.
    """
    df = df.copy()
    city_set: Set[int] = set(city_gcids)

    df["rate_value"] = _lookup_rate(df, rate_map)

    price_nonpositive = df["price"].isna() | (df["price"] <= 0)
    price_below_range = (
        (~price_nonpositive)
        & (df["price"] > 0)
        & (df["price"] <= 1000)
        & (df["currency_code"] == RUB)
    )
    invalid_distance = df["distance"].isna() | (df["distance"] <= 0)
    invalid_weight = df["total_weight"].isna() | (df["total_weight"] < 0)
    invalid_volume = df["total_volume"].isna() | (df["total_volume"] < 0)
    missing_city = (
        ~df["loading_city_gcid"].isin(city_set)
        | ~df["unloading_city_gcid"].isin(city_set)
    )
    missing_rate = (df["currency_code"] != RUB) & df["rate_value"].isna()

    reason_masks = {
        INVALID_PRICE: price_nonpositive,
        PRICE_BELOW_SOURCE_RANGE: price_below_range,
        INVALID_DISTANCE: invalid_distance,
        INVALID_WEIGHT: invalid_weight,
        INVALID_VOLUME: invalid_volume,
        MISSING_CITY: missing_city,
        MISSING_RATE: missing_rate,
    }

    quarantine_parts: list[pd.DataFrame] = []
    for reason_code, mask in reason_masks.items():
        if not bool(mask.any()):
            continue
        part = pd.DataFrame(
            {
                "price_id": df.loc[mask, "id"],
                "reason_code": reason_code,
                "raw_price": df.loc[mask, "price"],
                "currency_code": df.loc[mask, "currency_code"],
                "nearest_rate_date": df.loc[mask, "nearest_rate_date"],
                "loading_date": df.loc[mask, "loading_date"],
                "loading_city_gcid": df.loc[mask, "loading_city_gcid"],
                "unloading_city_gcid": df.loc[mask, "unloading_city_gcid"],
                "distance": df.loc[mask, "distance"],
                "total_weight": df.loc[mask, "total_weight"],
                "total_volume": df.loc[mask, "total_volume"],
                "body_type": df.loc[mask, "body_type"],
                "price_type": df.loc[mask, "price_type"],
            }
        )[list(QUARANTINE_COLUMNS)]
        quarantine_parts.append(part)

    quarantine = (
        pd.concat(quarantine_parts, ignore_index=True)
        if quarantine_parts
        else pd.DataFrame(columns=list(QUARANTINE_COLUMNS))
    )

    any_invalid = np.zeros(len(df), dtype=bool)
    for mask in reason_masks.values():
        any_invalid |= mask.to_numpy(dtype=bool)

    valid = df.loc[~any_invalid].copy()

    valid["price_rub"] = valid["price"] * valid["rate_value"]
    _fill_missing_category(valid, "weight_category", "total_weight", classify_weight_category)
    _fill_missing_category(valid, "volume_category", "total_volume", classify_volume_category)

    return valid, quarantine
