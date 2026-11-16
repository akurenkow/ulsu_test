"""Юнит-тесты вспомогательной логики пересчёта витрин (этап 3).

Полные SQL smoke-проверки витрины (идемпотентность, дедупликация городов,
унификация кузовов) покрываются на этапе 5 и требуют поднятой БД — см.
specs/step_05_tests_and_verification.md.
"""
from __future__ import annotations

import pytest

from etl import marts


def test_parse_as_of_canonical():
    assert marts.parse_as_of("2026-10-26") == "2026-10-26"


def test_parse_as_of_rejects_invalid_date():
    with pytest.raises(ValueError):
        marts.parse_as_of("not-a-date")


def test_substitute_as_of_replaces_literal():
    sql = "SELECT * FROM t WHERE d <= DATE '2026-10-26' AND x < '2026-10-26'"
    out = marts.substitute_as_of(sql, "2025-01-01")
    assert "2026-10-26" not in out
    assert "DATE '2025-01-01'" in out


def test_route_card_columns_match_spec_order():
    # 21 колонка; первые три — зерно месяца и городов; метрики доверия в конце.
    cols = marts.MART_ROUTE_CARD_COLUMNS
    assert len(cols) == 21
    assert cols[0] == "month"
    assert cols[1:3] == ("loading_city_gcid", "unloading_city_gcid")
    assert cols[5:8] == ("body_group", "weight_category", "volume_category")
    assert cols[15:17] == ("trips_count", "carriers_count")
    assert cols[-1] == "reliability_level"


def test_route_card_insert_sql_wraps_reference_select():
    sql = marts._route_card_insert_sql("2026-10-26")
    assert sql.startswith("INSERT INTO mart.mart_route_card (")
    assert "WITH" in sql
    assert "DATE '2026-10-26'" in sql

    # подстановка другой даты снимка меняет и фильтр будущих месяцев, и свежесть
    sql2 = marts._route_card_insert_sql("2025-01-01")
    assert "2026-10-26" not in sql2
    assert "DATE '2025-01-01'" in sql2


def test_fact_vs_corridor_columns_and_sql():
    cols = marts.FACT_VS_CORRIDOR_COLUMNS
    assert len(cols) == 23
    assert cols[0] == "price_id"
    assert "actual_price_rub" in cols
    assert cols[-1] == "corridor_band"

    sql = marts._fact_vs_corridor_insert_sql("2026-10-26")
    assert sql.startswith("INSERT INTO mart.fact_vs_corridor (")
    assert "DATE '2026-10-26'" in sql
    assert "mart.mart_route_card" in sql
