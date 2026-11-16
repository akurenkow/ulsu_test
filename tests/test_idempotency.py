"""Интеграционные тесты идемпотентности (этап 5, задача 5.2).

Требуют поднятый PostgreSQL с источником и хранилищем (`docker compose up -d`);
при недоступной БД автоматически пропускаются (см. `tests/conftest.py`).

Проверяют, что повторный прогон за один и тот же период/снимок не задваивает
строки и не меняет контрольные суммы и перцентили:

* ETL — `core.prices` + `core.dq_quarantine_prices`;
* пересчёт витрин — `mart.mart_route_card`;
* пакетный скоринг ML — `mart.ml_price_predictions` (тот же UPSERT-механизм
  `ml.pipeline._upsert_predictions`, который использует `ml.pipeline.run`).
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest
from sqlalchemy import text

from etl.marts import rebuild_marts
from etl.pipeline import run as run_etl
from ml.pipeline import PREDICTION_COLUMNS, _upsert_predictions

pytestmark = pytest.mark.db

AS_OF = "2026-10-26"
# Окно ETL с фактами и заложенными DQ-аномалиями (быстрое, детерминированное).
ETL_START, ETL_END = "2026-09-01", "2026-09-30"


def _core_snapshot(conn, start: str, end: str) -> dict:
    row = conn.execute(
        text(
            "SELECT count(*)                                        AS n, "
            "       count(DISTINCT price_id)                        AS n_distinct, "
            "       coalesce(round(sum(price_rub)::numeric, 2), 0)  AS sum_rub, "
            "       coalesce(round(percentile_cont(0.5) WITHIN GROUP "
            "           (ORDER BY price_rub)::numeric, 2), 0)       AS median "
            "FROM core.prices "
            "WHERE loading_date >= :s AND loading_date <= :e"
        ),
        {"s": start, "e": end},
    ).one()
    return dict(row._mapping)


def _quarantine_snapshot(conn, start: str, end: str) -> dict:
    rows = conn.execute(
        text(
            "SELECT reason_code, count(*) AS n "
            "FROM core.dq_quarantine_prices "
            "WHERE loading_date >= :s AND loading_date <= :e "
            "GROUP BY reason_code ORDER BY reason_code"
        ),
        {"s": start, "e": end},
    ).all()
    return {r.reason_code: r.n for r in rows}


def test_etl_idempotent(pg_dwh):
    run_etl(ETL_START, ETL_END)
    with pg_dwh.connect() as conn:
        core1 = _core_snapshot(conn, ETL_START, ETL_END)
        q1 = _quarantine_snapshot(conn, ETL_START, ETL_END)

    run_etl(ETL_START, ETL_END)  # повторный прогон за тот же период
    with pg_dwh.connect() as conn:
        core2 = _core_snapshot(conn, ETL_START, ETL_END)
        q2 = _quarantine_snapshot(conn, ETL_START, ETL_END)

    # число строк, суммы и медиана не меняются; дублей price_id нет.
    assert core1["n"] == core2["n"] > 0
    assert core1["n"] == core1["n_distinct"] == core2["n_distinct"]
    assert core1["sum_rub"] == core2["sum_rub"]
    assert core1["median"] == core2["median"]
    # карантин окна перезаписывается атомарно — статистика причин стабильна.
    assert q1 == q2


def test_mart_rebuild_idempotent(pg_dwh):
    def snapshot(conn) -> dict:
        row = conn.execute(
            text(
                "SELECT count(*)                                        AS n, "
                "  count(*) FILTER (WHERE NOT (p25_trip <= median_trip "
                "        AND median_trip <= p75_trip))                   AS trip_viol, "
                "  count(*) FILTER (WHERE NOT (p25_km <= median_km "
                "        AND median_km <= p75_km))                       AS km_viol, "
                "  coalesce(round(sum(median_trip)::numeric, 2), 0)      AS sum_median, "
                "  coalesce(round(sum(trips_count)::numeric, 0), 0)      AS sum_trips, "
                "  count(DISTINCT (month, loading_city_gcid, unloading_city_gcid, "
                "        body_group, weight_category))                   AS n_grain "
                "FROM mart.mart_route_card"
            )
        ).one()
        return dict(row._mapping)

    rebuild_marts(AS_OF)
    with pg_dwh.connect() as conn:
        s1 = snapshot(conn)

    rebuild_marts(AS_OF)  # повторный пересчёт за тот же снимок
    with pg_dwh.connect() as conn:
        s2 = snapshot(conn)

    assert s1["n"] == s2["n"] > 0
    # PRIMARY KEY по зерну + атомарная перезапись -> дублей зерна нет.
    assert s1["n"] == s1["n_grain"] == s2["n_grain"]
    assert s1["sum_median"] == s2["sum_median"]
    assert s1["sum_trips"] == s2["sum_trips"]
    assert s1["trip_viol"] == s1["km_viol"] == 0


def test_ml_upsert_idempotent(pg_dwh):
    """Повторный скоринг не дублирует строки в ``mart.ml_price_predictions``.

    Используем сам UPSERT-механизм скоринга с синтетическими строками в
    зарезервированном диапазоне отрицательных ``price_id``, чтобы не задеть
    реальные прогнозы; по завершении строки удаляются.
    """
    ids = [-9_000_001, -9_000_002, -9_000_003]
    frame = pd.DataFrame(
        {
            "price_id": ids,
            "loading_date": [
                dt.date(2026, 10, 20),
                dt.date(2026, 10, 21),
                dt.date(2026, 11, 10),  # будущая заявка -> факт/ошибки NULL
            ],
            "loading_city_gcid": [1001, 1001, 1001],
            "unloading_city_gcid": [1003, 1003, 1003],
            "body_group": ["Тент", "Тент", "Тент"],
            "actual_price_rub": [84500.0, 90000.0, None],
            "predicted_price_rub": [84000.0, 91000.0, 152000.0],
            "predicted_price_rub_lo": [83000.0, 90000.0, 148000.0],
            "predicted_price_rub_hi": [85000.0, 92000.0, 156000.0],
            "abs_error_rub": [500.0, 1000.0, None],
            "ape_pct": [0.59, 1.11, None],
            "model_name": ["test-upsert"] * 3,
            "scored_at": [dt.datetime.now(dt.timezone.utc)] * 3,
        }
    )[list(PREDICTION_COLUMNS)]

    try:
        _upsert_predictions(pg_dwh, frame)
        with pg_dwh.connect() as conn:
            n1 = conn.execute(
                text(
                    "SELECT count(*) FROM mart.ml_price_predictions "
                    "WHERE model_name = 'test-upsert'"
                )
            ).scalar()

        # Повторный скоринг с обновлёнными значениями -> UPSERT, а не вставка.
        frame2 = frame.copy()
        frame2["predicted_price_rub"] = frame2["predicted_price_rub"] + 1.0
        _upsert_predictions(pg_dwh, frame2)

        with pg_dwh.connect() as conn:
            n2 = conn.execute(
                text(
                    "SELECT count(*) FROM mart.ml_price_predictions "
                    "WHERE model_name = 'test-upsert'"
                )
            ).scalar()
            updated = conn.execute(
                text(
                    "SELECT predicted_price_rub FROM mart.ml_price_predictions "
                    "WHERE model_name = 'test-upsert' AND price_id = :pid"
                ),
                {"pid": ids[0]},
            ).scalar()
    finally:
        with pg_dwh.begin() as conn:
            conn.execute(
                text(
                    "DELETE FROM mart.ml_price_predictions "
                    "WHERE model_name = 'test-upsert'"
                )
            )

    assert n1 == 3
    assert n2 == 3  # без дублей после повторного скоринга
    assert float(updated) == float(frame2["predicted_price_rub"].iloc[0])
