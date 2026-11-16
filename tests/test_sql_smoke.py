"""SQL smoke-проверки витрин и справочников (этап 5, задача 5.3).

Требуют поднятый PostgreSQL с источником и хранилищем (`docker compose up -d`);
при недоступной БД автоматически пропускаются (см. `tests/conftest.py`).

Проверяют:
* консистентность витрины `mart.mart_route_card`;
* отсутствие размножения сделок при дублях `gc_id` справочника городов;
* унификацию групп кузова (одна цена — одна строка на группу);
* NULL-факт/ошибки для будущих ML-прогнозов.
"""
from __future__ import annotations

import pytest
from sqlalchemy import text

pytestmark = pytest.mark.db


def test_mart_route_card_consistency(pg_dwh):
    row = pg_dwh.connect().execute(
        text(
            "SELECT count(*)                                                          AS total, "
            "  count(*) FILTER (WHERE trips_count < 0)                                AS neg_trips, "
            "  count(*) FILTER (WHERE NOT (p25_trip <= median_trip "
            "        AND median_trip <= p75_trip))                                    AS trip_viol, "
            "  count(*) FILTER (WHERE NOT (p25_km <= median_km "
            "        AND median_km <= p75_km))                                        AS km_viol, "
            "  count(*) FILTER (WHERE volume_category IS NOT NULL "
            "        AND volume_category NOT IN ('FTL', 'OTH'))                       AS bad_volume "
            "FROM mart.mart_route_card"
        )
    ).one()

    assert row.total > 0
    assert row.neg_trips == 0
    assert row.trip_viol == 0
    assert row.km_viol == 0
    assert row.bad_volume == 0


def test_cities_dedup_no_fanout(pg_source, pg_dwh):
    # 1) в операционном источнике у Москвы (1001) и Казани (1003) есть дубли gc_id.
    dup = pg_source.connect().execute(
        text(
            "SELECT gc_id, count(*) AS n FROM public.cities "
            "WHERE gc_id IN (1001, 1003) GROUP BY gc_id ORDER BY gc_id"
        )
    ).all()
    dup_map = {r.gc_id: r.n for r in dup}
    assert dup_map.get(1001, 0) > 1
    assert dup_map.get(1003, 0) > 1

    # 2) в core.cities gc_id уникален — дедупликация выполнена.
    with pg_dwh.connect() as conn:
        core = conn.execute(
            text("SELECT count(*) AS total, count(DISTINCT gc_id) AS distinct_gc FROM core.cities")
        ).one()
        one_each = conn.execute(
            text(
                "SELECT gc_id, count(*) AS n FROM core.cities "
                "WHERE gc_id IN (1001, 1003) GROUP BY gc_id"
            )
        ).all()
    assert core.total == core.distinct_gc
    assert {r.n for r in one_each} == {1}

    # 3) JOIN к дедуплицированному core.cities не размножает сделки.
    with pg_dwh.connect() as conn:
        prices = conn.execute(text("SELECT count(*) FROM core.prices")).scalar()
        joined = conn.execute(
            text(
                "SELECT count(*) FROM core.prices p "
                "JOIN core.cities c ON c.gc_id = p.loading_city_gcid"
            )
        ).scalar()
    assert joined == prices


def test_body_group_no_duplicate_price_within_group(pg_dwh):
    """Одна цена даёт одну строку на группу кузова (UNION + DISTINCT)."""
    row = pg_dwh.connect().execute(
        text(
            "WITH body_group AS ("
            "    SELECT DISTINCT p.price_id,"
            "        CASE bt.std_id WHEN 56 THEN 'Тент' WHEN 4 THEN 'Рефрижератор'"
            "                       WHEN 2 THEN 'Изотерм' WHEN 15 THEN 'Борт' ELSE 'Прочие' END"
            "        AS body_group"
            "    FROM core.prices p"
            "    JOIN core.price_body_types pbt ON pbt.price_id = p.price_id"
            "    JOIN core.body_types bt ON bt.id = pbt.body_type_id"
            "    UNION"
            "    SELECT p.price_id,"
            "        CASE WHEN p.body_type IS NULL THEN 'Не указан'"
            "             WHEN p.body_type = 56 THEN 'Тент' WHEN p.body_type = 4 THEN 'Рефрижератор'"
            "             WHEN p.body_type = 2 THEN 'Изотерм' WHEN p.body_type = 15 THEN 'Борт'"
            "             ELSE 'Прочие' END"
            "    FROM core.prices p"
            "    WHERE NOT EXISTS (SELECT 1 FROM core.price_body_types pbt"
            "                      WHERE pbt.price_id = p.price_id)"
            ")"
            "SELECT count(*) AS total, "
            "       count(DISTINCT (price_id, body_group)) AS distinct_pairs "
            "FROM body_group"
        )
    ).one()

    assert row.total > 0
    assert row.total == row.distinct_pairs  # нет дублей (price_id, body_group)


def test_ml_future_predictions_have_null_fact(pg_dwh):
    row = pg_dwh.connect().execute(
        text(
            "SELECT count(*)                                             AS future_total, "
            "  count(*) FILTER (WHERE actual_price_rub IS NOT NULL)      AS bad_fact, "
            "  count(*) FILTER (WHERE abs_error_rub  IS NOT NULL)        AS bad_abs, "
            "  count(*) FILTER (WHERE ape_pct        IS NOT NULL)        AS bad_ape "
            "FROM mart.ml_price_predictions "
            "WHERE loading_date > :as_of"
        ),
        {"as_of": "2026-10-26"},
    ).one()

    assert row.future_total > 0
    # Для будущих заявок факт и ошибки обязаны быть NULL.
    assert row.bad_fact == row.bad_abs == row.bad_ape == 0


def test_ml_predictions_have_ordered_interval(pg_dwh):
    """Квантильный прогноз хранит честный интервал lo <= med <= hi."""
    row = pg_dwh.connect().execute(
        text(
            "SELECT count(*)                                                        AS total, "
            "  count(*) FILTER (WHERE predicted_price_rub_lo IS NULL)               AS bad_lo, "
            "  count(*) FILTER (WHERE predicted_price_rub_hi IS NULL)               AS bad_hi, "
            "  count(*) FILTER (WHERE NOT (predicted_price_rub_lo <= predicted_price_rub "
            "        AND predicted_price_rub <= predicted_price_rub_hi))            AS bad_order "
            "FROM mart.ml_price_predictions"
        )
    ).one()

    assert row.total > 0
    assert row.bad_lo == 0
    assert row.bad_hi == 0
    assert row.bad_order == 0


def test_route_card_facts_view_matches_mart(pg_dwh):
    """view mart.v_route_card_facts — та же выборка, что и mart.mart_route_card.

    Пульная медиана Superset'а считается по неагрегированным фактам, поэтому
    универсум view обязан совпадать с витриной: число ячеек = число строк марта,
    а число фактов = сумма trips_count по ячейкам.
    """
    with pg_dwh.connect() as conn:
        mart = conn.execute(
            text(
                "SELECT count(*) AS rows, sum(trips_count) AS trips "
                "FROM mart.mart_route_card"
            )
        ).one()
        view = conn.execute(
            text(
                "SELECT count(*)                                                       AS fact_rows, "
                "  count(DISTINCT (month, loading_city_gcid, unloading_city_gcid, "
                "                    body_group, weight_category))                    AS cells "
                "FROM mart.v_route_card_facts"
            )
        ).one()

    assert mart.rows > 0
    assert view.cells == mart.rows
    assert view.fact_rows == mart.trips


def test_route_card_facts_pooled_corridor_ordered(pg_dwh):
    """Пульные квартили по группе кузова монотонны: P25 <= P50 <= P75."""
    row = pg_dwh.connect().execute(
        text(
            "SELECT count(*) FILTER (WHERE NOT (p25 <= p50 AND p50 <= p75)) AS viol "
            "FROM ("
            "  SELECT percentile_cont(0.25) WITHIN GROUP (ORDER BY price_rub) AS p25, "
            "         percentile_cont(0.50) WITHIN GROUP (ORDER BY price_rub) AS p50, "
            "         percentile_cont(0.75) WITHIN GROUP (ORDER BY price_rub) AS p75 "
            "  FROM mart.v_route_card_facts "
            "  GROUP BY body_group"
            ") t"
        )
    ).one()

    assert row.viol == 0
