"""SQL-запросы API демо-дашборда «Рынок ставок».

Все агрегаты считаются поверх ``core.prices`` (очищенные сделки в RUB) в БД
``dwh``. Зерно — календарная неделя (``date_trunc('week', loading_date)``).

Маппинг срезов демо-страницы на данные DWH:

* ``body``  — группа кузова (как в ``mart_route_card``): приоритет
  ``core.price_body_types`` → ``core.body_types.std_id``, fallback —
  ``core.prices.body_type`` (56→Тент, 4→Рефрижератор, 2→Изотерм, 15→Борт,
  NULL→«Не указан», прочее→«Прочие»). Цены с несколькими кузовами учитываются
  в каждой из своих групп.
* ``ton``   — весовая категория ``weight_category``: 5T/L5T/OTH→«5» (до 5 т),
  10T→«10» (5–10 т), FTL/G20T→«20» (20 т и более).
* ``leg``   — производное от ``distance``: <300→s, 300–1000→m, >1000→l.
* ``cargo`` — в данных отсутствует, поэтому срез не отдаётся.

Фильтры выборки совпадают с эталонным расчётом ``mart_route_card``:
``price_type IN (1,2,3,4,5,9)`` (факт), ``complex = false``, ``price_rub > 0``,
``distance > 0``, ``outlier_mark = false``.
"""
from __future__ import annotations

import os

# --- справочники-константы (vocab страницы) ---------------------------------
BODIES = ["Тент", "Рефрижератор", "Изотерм", "Борт", "Прочие", "Не указан"]
TONS = [
    {"key": "5", "label": "до 5 т"},
    {"key": "10", "label": "5–10 т"},
    {"key": "20", "label": "20 т и более"},
]
LEGS = [
    {"key": "s", "label": "до 300 км"},
    {"key": "m", "label": "300–1000 км"},
    {"key": "l", "label": "1000+ км"},
]

# Дата снимка (as_of): граница между историческими (завершёнными) и плановыми
# погрузками. Совпадает с генератором синтетических данных sql/synthetic.sql.
AS_OF = os.getenv("AS_OF", "2026-10-26")

# Прогноз тарифа: имя модели и горизонт «2–4 недели» (28 дней от as_of).
FORECAST_MODEL_NAME = "catboost"
FORECAST_HORIZON_DAYS = 28

# Базовые фильтры выборки (алиас таблицы — ``p``).
_BASE_FILTERS = """
      p.price_type IN (1, 2, 3, 4, 5, 9)
  AND p.complex = false
  AND p.price_rub > 0
  AND p.distance > 0
  AND p.outlier_mark = false
  AND p.loading_date IS NOT NULL
"""

# Фильтры срезов (применяются к CTE ``facts``). Параметры приводятся явным
# CAST'ом, чтобы PostgreSQL мог вывести тип NULL-параметра (иначе — ошибка
# «could not determine data type of parameter»).
_DIM_FILTERS = """
      (CAST(:lc AS INTEGER) IS NULL OR loading_city_gcid = CAST(:lc AS INTEGER))
  AND (CAST(:uc AS INTEGER) IS NULL OR unloading_city_gcid = CAST(:uc AS INTEGER))
  AND (CAST(:body AS VARCHAR) IS NULL OR body_group = CAST(:body AS VARCHAR))
  AND (CAST(:ton AS VARCHAR) IS NULL OR ton = CAST(:ton AS VARCHAR))
  AND (CAST(:leg AS VARCHAR) IS NULL OR leg = CAST(:leg AS VARCHAR))
"""

# Группа кузова — та же логика, что в эталонном SQL mart_route_card:
# приоритет price_body_types -> body_types.std_id, fallback prices.body_type.
_BODY_GROUP = """
    SELECT DISTINCT
        p.price_id,
        CASE bt.std_id
            WHEN 56 THEN 'Тент'
            WHEN 4  THEN 'Рефрижератор'
            WHEN 2  THEN 'Изотерм'
            WHEN 15 THEN 'Борт'
            ELSE 'Прочие'
        END AS body_group
    FROM core.prices p
    JOIN core.price_body_types pbt ON pbt.price_id = p.price_id
    JOIN core.body_types bt ON bt.id = pbt.body_type_id
    UNION
    SELECT
        p.price_id,
        CASE
            WHEN p.body_type IS NULL THEN 'Не указан'
            WHEN p.body_type = 56 THEN 'Тент'
            WHEN p.body_type = 4  THEN 'Рефрижератор'
            WHEN p.body_type = 2  THEN 'Изотерм'
            WHEN p.body_type = 15 THEN 'Борт'
            ELSE 'Прочие'
        END AS body_group
    FROM core.prices p
    WHERE NOT EXISTS (
        SELECT 1 FROM core.price_body_types pbt WHERE pbt.price_id = p.price_id
    )
"""

_FACTS = f"""
    SELECT
        date_trunc('week', p.loading_date)::date AS week,
        p.loading_city_gcid,
        p.unloading_city_gcid,
        bg.body_group,
        CASE p.weight_category
            WHEN 'G20T' THEN '20'
            WHEN 'FTL'  THEN '20'
            WHEN '10T'  THEN '10'
            WHEN '5T'   THEN '5'
            WHEN 'L5T'  THEN '5'
            ELSE '5'
        END AS ton,
        CASE
            WHEN p.distance < 300 THEN 's'
            WHEN p.distance <= 1000 THEN 'm'
            ELSE 'l'
        END AS leg,
        p.distance,
        p.price_rub,
        (p.loading_date > DATE '{AS_OF}') AS planned
    FROM core.prices p
    JOIN body_group bg ON bg.price_id = p.price_id
    WHERE {_BASE_FILTERS}
"""

_PREFIX = f"WITH body_group AS (\n{_BODY_GROUP}\n), facts AS (\n{_FACTS}\n)"


def dim_params(lc, uc, body, ton, leg) -> dict:
    """Параметры срезов для запросов (``None`` = «все»)."""
    return {"lc": lc, "uc": uc, "body": body, "ton": ton, "leg": leg}


def max_week_sql() -> str:
    """Последняя неделя с данными (для окон «4 недели»)."""
    return (
        f"SELECT max(date_trunc('week', loading_date)::date) AS max_week "
        f"FROM core.prices p WHERE {_BASE_FILTERS}"
    )


def series_sql() -> str:
    """Еженедельный ряд ставок (lo/med/hi — ₽/км, n — число сделок)."""
    return (
        _PREFIX
        + f"""
SELECT week,
       (percentile_cont(0.25) WITHIN GROUP (ORDER BY price_rub / distance))::float8 AS lo,
       (percentile_cont(0.50) WITHIN GROUP (ORDER BY price_rub / distance))::float8 AS med,
       (percentile_cont(0.75) WITHIN GROUP (ORDER BY price_rub / distance))::float8 AS hi,
       count(*) AS n,
       count(*) FILTER (WHERE planned) AS n_planned
FROM facts
WHERE {_DIM_FILTERS}
GROUP BY week
ORDER BY week
"""
    )


def stats_sql() -> str:
    """Медианная ставка за рейс (₽) и характерное плечо (км) по срезу."""
    return (
        _PREFIX
        + f"""
SELECT (percentile_cont(0.50) WITHIN GROUP (ORDER BY price_rub))::float8 AS median_trip,
       (percentile_cont(0.50) WITHIN GROUP (ORDER BY distance))::float8 AS km,
       count(*) AS total_n
FROM facts
WHERE {_DIM_FILTERS}
"""
    )


def forecast_sql() -> str:
    """Еженедельный прогноз ставки ₽/км (med — точечная MAPE-модель; lo/hi — q25/q75).

    Читает ``mart.ml_price_predictions`` (плановые заявки ``loading_date`` в
    интервале ``(as_of, horizon]``), джойнит ``core.prices`` для ``distance`` и
    весовой категории. Агрегат — медиана прогнозных ₽/км по неделям.
    """
    return f"""
WITH f AS (
    SELECT
        date_trunc('week', mp.loading_date)::date AS week,
        mp.loading_city_gcid,
        mp.unloading_city_gcid,
        mp.body_group,
        CASE p.weight_category
            WHEN 'G20T' THEN '20'
            WHEN 'FTL'  THEN '20'
            WHEN '10T'  THEN '10'
            WHEN '5T'   THEN '5'
            WHEN 'L5T'  THEN '5'
            ELSE '5'
        END AS ton,
        CASE
            WHEN p.distance < 300 THEN 's'
            WHEN p.distance <= 1000 THEN 'm'
            ELSE 'l'
        END AS leg,
        p.distance,
        mp.predicted_price_rub_lo,
        mp.predicted_price_rub,
        mp.predicted_price_rub_hi
    FROM mart.ml_price_predictions mp
    JOIN core.prices p ON p.price_id = mp.price_id
    WHERE mp.model_name = :model_name
      AND mp.loading_date > :as_of
      AND mp.loading_date <= :horizon
)
SELECT week,
       (percentile_cont(0.50) WITHIN GROUP (ORDER BY predicted_price_rub_lo / distance))::float8 AS lo,
       (percentile_cont(0.50) WITHIN GROUP (ORDER BY predicted_price_rub / distance))::float8 AS med,
       (percentile_cont(0.50) WITHIN GROUP (ORDER BY predicted_price_rub_hi / distance))::float8 AS hi,
       count(*) AS n
FROM f
WHERE {_DIM_FILTERS}
GROUP BY week
ORDER BY week
"""


def directions_sql(limit: int) -> str:
    """Топ направлений по числу сделок (для справочника срезов)."""
    return (
        _PREFIX
        + f"""
SELECT f.loading_city_gcid,
       f.unloading_city_gcid,
       (percentile_cont(0.50) WITHIN GROUP (ORDER BY f.distance))::float8 AS km,
       count(*) AS trips,
       coalesce(cl.name, f.loading_city_gcid::text) AS from_name,
       coalesce(cu.name, f.unloading_city_gcid::text) AS to_name
FROM facts f
LEFT JOIN core.cities cl ON cl.gc_id = f.loading_city_gcid
LEFT JOIN core.cities cu ON cu.gc_id = f.unloading_city_gcid
GROUP BY f.loading_city_gcid, f.unloading_city_gcid, cl.name, cu.name
ORDER BY trips DESC
LIMIT {int(limit)}
"""
    )


def heat_sql() -> str:
    """Heatmap «кузов × плечо»: медиана последних 4 недель и Δ к прошлым 4."""
    return (
        _PREFIX
        + f""",
cur AS (
    SELECT body_group, leg,
           (percentile_cont(0.50) WITHIN GROUP (ORDER BY price_rub / distance))::float8 AS v,
           count(*) AS n
    FROM facts
    WHERE {_DIM_FILTERS} AND week >= :cur_start
    GROUP BY body_group, leg
),
prv AS (
    SELECT body_group, leg,
           (percentile_cont(0.50) WITHIN GROUP (ORDER BY price_rub / distance))::float8 AS v
    FROM facts
    WHERE {_DIM_FILTERS} AND week >= :prev_start AND week < :cur_start
    GROUP BY body_group, leg
)
SELECT coalesce(c.body_group, p.body_group) AS body_group,
       coalesce(c.leg, p.leg) AS leg,
       c.v AS v, p.v AS v_prev, c.n AS n
FROM cur c
FULL JOIN prv p ON p.body_group = c.body_group AND p.leg = c.leg
ORDER BY body_group, leg
"""
    )


def bars_sql() -> str:
    """Бары по кузовам: медиана последних 4 недель и Δ к прошлым 4."""
    return (
        _PREFIX
        + f""",
cur AS (
    SELECT body_group,
           (percentile_cont(0.50) WITHIN GROUP (ORDER BY price_rub / distance))::float8 AS v,
           count(*) AS n
    FROM facts
    WHERE {_DIM_FILTERS} AND week >= :cur_start
    GROUP BY body_group
),
prv AS (
    SELECT body_group,
           (percentile_cont(0.50) WITHIN GROUP (ORDER BY price_rub / distance))::float8 AS v
    FROM facts
    WHERE {_DIM_FILTERS} AND week >= :prev_start AND week < :cur_start
    GROUP BY body_group
)
SELECT coalesce(c.body_group, p.body_group) AS body_group,
       c.v AS v, p.v AS v_prev, c.n AS n
FROM cur c
FULL JOIN prv p ON p.body_group = c.body_group
ORDER BY body_group
"""
    )


def top_sql(limit: int) -> str:
    """Топ направлений по сделкам за 4 недели + текущая ставка и Δ."""
    return (
        _PREFIX
        + f""",
dirs AS (
    SELECT loading_city_gcid, unloading_city_gcid,
           (percentile_cont(0.50) WITHIN GROUP (ORDER BY distance))::float8 AS km,
           (percentile_cont(0.50) WITHIN GROUP (ORDER BY price_rub / distance) FILTER (WHERE week >= :cur_start))::float8 AS v_cur,
           (percentile_cont(0.50) WITHIN GROUP (ORDER BY price_rub / distance) FILTER (WHERE week >= :prev_start AND week < :cur_start))::float8 AS v_prev,
           count(*) FILTER (WHERE week >= :cur_start) AS n_cur
    FROM facts
    WHERE {_DIM_FILTERS}
    GROUP BY loading_city_gcid, unloading_city_gcid
    ORDER BY n_cur DESC
    LIMIT {int(limit)}
)
SELECT d.loading_city_gcid, d.unloading_city_gcid, d.km, d.v_cur, d.v_prev, d.n_cur,
       coalesce(cl.name, d.loading_city_gcid::text) AS from_name,
       coalesce(cu.name, d.unloading_city_gcid::text) AS to_name
FROM dirs d
LEFT JOIN core.cities cl ON cl.gc_id = d.loading_city_gcid
LEFT JOIN core.cities cu ON cu.gc_id = d.unloading_city_gcid
ORDER BY d.n_cur DESC
"""
    )


def spark_sql(pairs) -> str:
    """Недельные медианы (спарклайн) для заданных направлений за 13 недель."""
    in_clause = ", ".join(f"({int(lc)}, {int(uc)})" for lc, uc in pairs)
    return (
        _PREFIX
        + f"""
SELECT loading_city_gcid, unloading_city_gcid, week,
       (percentile_cont(0.50) WITHIN GROUP (ORDER BY price_rub / distance))::float8 AS med
FROM facts
WHERE week >= :spark_start
  AND (loading_city_gcid, unloading_city_gcid) IN ({in_clause})
GROUP BY loading_city_gcid, unloading_city_gcid, week
ORDER BY loading_city_gcid, unloading_city_gcid, week
"""
    )
