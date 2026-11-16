"""Юнит-тесты SQL-контракта API демо-дашборда (этап 6).

Проверяют чистую логику ``api.queries`` (без БД и без fastapi): маппинг срезов,
сборку SQL и состав справочников.
"""
from __future__ import annotations

from api import queries


def test_dim_params_none_means_all():
    assert queries.dim_params(None, None, None, None, None) == {
        "lc": None, "uc": None, "body": None, "ton": None, "leg": None,
    }


def test_dim_params_passthrough():
    p = queries.dim_params(1001, 1002, "Тент", "20", "s")
    assert p == {"lc": 1001, "uc": 1002, "body": "Тент", "ton": "20", "leg": "s"}


def test_series_sql_has_weekly_grain_and_km_percentiles():
    sql = queries.series_sql()
    assert "date_trunc('week', p.loading_date)::date" in sql
    assert "price_rub / distance" in sql
    assert "percentile_cont(0.50)" in sql
    assert "GROUP BY week" in sql


def test_sql_uses_mart_body_group_resolution():
    # Группа кузова должна резолвиться как в эталонном mart_route_card.
    sql = queries.series_sql()
    assert "core.price_body_types" in sql
    assert "core.body_types" in sql
    assert "std_id" in sql


def test_dim_filters_cast_null_params():
    # NULL-параметры приводятся CAST'ом (иначе psycopg/Postgres не выведет тип).
    sql = queries.series_sql()
    assert "CAST(:lc AS INTEGER)" in sql
    assert "CAST(:body AS VARCHAR)" in sql


def test_direction_sql_limits_and_names_cities():
    sql = queries.directions_sql(15)
    assert "LIMIT 15" in sql
    assert "core.cities" in sql
    assert "coalesce(" in sql


def test_ref_vocab_matches_page():
    assert queries.BODIES == ["Тент", "Рефрижератор", "Изотерм", "Борт", "Прочие", "Не указан"]
    assert [t["key"] for t in queries.TONS] == ["5", "10", "20"]
    assert [l["key"] for l in queries.LEGS] == ["s", "m", "l"]


def test_forecast_sql_reads_quantile_predictions():
    sql = queries.forecast_sql()
    assert "mart.ml_price_predictions" in sql
    assert "predicted_price_rub_lo" in sql
    assert "predicted_price_rub_hi" in sql
    assert "percentile_cont(0.50)" in sql
    assert ":horizon" in sql
    assert ":model_name" in sql
    assert "GROUP BY week" in sql
