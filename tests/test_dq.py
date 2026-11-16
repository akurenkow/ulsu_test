"""Юнит-тесты правил Data Quality (этап 2): категории, валюта, карантин."""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from etl import dq

COLS = [
    "id", "price", "currency_code", "nearest_rate_date", "loading_city_gcid",
    "unloading_city_gcid", "distance", "total_weight", "total_volume",
    "body_type", "price_type", "weight_category", "volume_category", "loading_date",
]


def frame(rows):
    return pd.DataFrame(rows, columns=COLS)


@pytest.mark.parametrize(
    ("weight", "expected"),
    [
        (30.0, "G20T"),
        (29.9, "FTL"),
        (10.0, "FTL"),
        (9.9, "10T"),
        (5.0, "10T"),
        (4.9, "5T"),
        (2.5, "5T"),
        (2.4, "L5T"),
        (0.5, "L5T"),
        (0.4, "OTH"),
        (0.0, "OTH"),
        (4.95, "10T"),   # округление до 0.1 т: 4.95 -> 5.0
        (29.95, "G20T"),  # округление до 0.1 т: 29.95 -> 30.0
    ],
)
def test_classify_weight_category(weight, expected):
    assert dq.classify_weight_category(weight) == expected


@pytest.mark.parametrize(
    ("volume", "expected"),
    [(0.0, "OTH"), (0.1, "FTL"), (120.0, "FTL"), (120.1, "OTH"), (-1.0, "OTH")],
)
def test_classify_volume_category(volume, expected):
    assert dq.classify_volume_category(volume) == expected


def test_round_half_up_matches_sql():
    assert dq.round_half_up(4.95, 1) == 5.0
    assert dq.round_half_up(0.45, 1) == 0.5
    assert dq.round_half_up(10.0, 1) == 10.0


def _valid_row(**overrides):
    row = dict(
        id=1,
        price=84500.0,
        currency_code="643",
        nearest_rate_date=None,
        loading_city_gcid=1001,
        unloading_city_gcid=1003,
        distance=897.0,
        total_weight=20.0,
        total_volume=80.0,
        body_type=56,
        price_type=3,
        weight_category=None,
        volume_category=None,
        loading_date=dt.date(2026, 9, 15),
    )
    row.update(overrides)
    return row


def test_valid_rub_row_is_converted_and_enriched():
    rate_map = {}
    valid, quarantine = dq.evaluate_dq(frame([_valid_row()]), rate_map, {1001, 1003})
    assert len(valid) == 1
    assert len(quarantine) == 0
    assert valid.iloc[0]["price_rub"] == 84500.0
    assert valid.iloc[0]["rate_value"] == 1.0
    assert valid.iloc[0]["weight_category"] == "FTL"
    assert valid.iloc[0]["volume_category"] == "FTL"


def test_foreign_currency_with_exact_rate():
    rate_map = {("840", dt.date(2026, 9, 12)): 90.0}
    row = _valid_row(
        currency_code="840",
        price=1000.0,
        nearest_rate_date=dt.date(2026, 9, 12),
    )
    valid, quarantine = dq.evaluate_dq(frame([row]), rate_map, {1001, 1003})
    assert len(valid) == 1
    assert valid.iloc[0]["price_rub"] == 90000.0


def test_foreign_currency_missing_rate_quarantined():
    row = _valid_row(currency_code="840", price=1000.0, nearest_rate_date=None)
    valid, quarantine = dq.evaluate_dq(frame([row]), {}, {1001, 1003})
    assert len(valid) == 0
    assert list(quarantine["reason_code"]) == [dq.MISSING_RATE]


def test_foreign_currency_no_fallback_to_loading_date():
    # Курс есть на loading_date, но не на nearest_rate_date -> fallback по
    # loading_date НЕ применяется: строка уходит в карантин MISSING_RATE.
    row = _valid_row(
        currency_code="840",
        price=1000.0,
        nearest_rate_date=dt.date(2026, 9, 10),   # курса на эту дату нет
        loading_date=dt.date(2026, 9, 15),        # курс есть только на эту дату
    )
    rate_map = {("840", dt.date(2026, 9, 15)): 90.0}
    valid, quarantine = dq.evaluate_dq(frame([row]), rate_map, {1001, 1003})
    assert len(valid) == 0
    assert list(quarantine["reason_code"]) == [dq.MISSING_RATE]


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"price": 0.0}, dq.INVALID_PRICE),
        ({"price": -5.0}, dq.INVALID_PRICE),
        ({"price": 500.0, "currency_code": "643"}, dq.PRICE_BELOW_SOURCE_RANGE),
        ({"distance": 0.0}, dq.INVALID_DISTANCE),
        ({"total_weight": -1.0}, dq.INVALID_WEIGHT),
        ({"total_volume": -1.0}, dq.INVALID_VOLUME),
        ({"loading_city_gcid": 9999}, dq.MISSING_CITY),
        ({"unloading_city_gcid": 9999}, dq.MISSING_CITY),
    ],
)
def test_dq_reasons(overrides, reason):
    valid, quarantine = dq.evaluate_dq(frame([_valid_row(**overrides)]), {}, {1001, 1003})
    assert len(valid) == 0
    assert reason in set(quarantine["reason_code"])


def test_multiple_reasons_one_row_per_reason():
    row = _valid_row(price=0.0, distance=0.0, loading_city_gcid=9999)
    valid, quarantine = dq.evaluate_dq(frame([row]), {}, {1001, 1003})
    assert len(valid) == 0
    assert set(quarantine["reason_code"]) == {
        dq.INVALID_PRICE,
        dq.INVALID_DISTANCE,
        dq.MISSING_CITY,
    }
    assert len(quarantine) == 3
