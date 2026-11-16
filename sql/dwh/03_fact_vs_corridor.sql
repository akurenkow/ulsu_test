-- ============================================================================
-- Эталонный SQL расчёта витрины mart.fact_vs_corridor (факт × рыночный коридор).
-- PostgreSQL 17, БД `dwh`.
--
-- Как прогнать вручную (просмотр результата без записи в таблицу):
--   psql -U dwh_user -d dwh -f sql/dwh/03_fact_vs_corridor.sql
--
-- Пайплайн (`etl/marts.py`) использует этот же SELECT как источник для
-- `INSERT INTO mart.fact_vs_corridor` (атомарная перезапись). Дата снимка задана
-- литералом '2026-10-26' (значение AS_OF); при пересчёте за другую дату
-- `etl/marts.py` подставляет её вместо литерала.
--
-- Логика: каждая закрытая сделка (тот же универсум, что и mart_route_card)
-- сопоставляется с коридором P25–P50–P75 своей ячейки (месяц погрузки × маршрут
-- × группа кузова × весовая категория) из mart.mart_route_card. Отклонение
-- факта от коридора считается в процентах к медиане и классифицируется по
-- зонам: ниже P25 / внутри коридора / выше P75 (по руб./км).
-- ============================================================================

WITH
-- ---------- 1. Унификация групп кузова (как в mart_route_card) ----------
body_group AS (
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
),

-- ---------- 2. Закрытые сделки (факты) с их рыночной ячейкой ----------
facts AS (
    SELECT
        p.price_id,
        p.loading_date,
        date_trunc('month', p.loading_date)::date AS month,
        p.loading_city_gcid,
        p.unloading_city_gcid,
        bg.body_group,
        p.weight_category,
        p.distance,
        p.price_rub,
        round((p.price_rub / NULLIF(p.distance, 0)), 4) AS price_km
    FROM core.prices p
    JOIN body_group bg ON bg.price_id = p.price_id
    WHERE p.price_type IN (1, 2, 3, 4, 5, 9)        -- is_fact = true
      AND p.complex = false                          -- только прямые маршруты
      AND p.price > 0
      AND p.distance > 0
      AND p.price_rub IS NOT NULL
      AND p.price_rub > 0
      AND p.outlier_mark = false
      AND p.loading_date <= DATE '2026-10-26'        -- без плановых погрузок будущего
)

-- ---------- 3. Факт × коридор ----------
SELECT
    f.price_id,
    f.loading_date,
    f.loading_city_gcid,
    f.unloading_city_gcid,
    cl.name AS loading_city_name,
    cu.name AS unloading_city_name,
    f.body_group,
    f.weight_category,
    f.distance,
    f.price_rub                                                             AS actual_price_rub,
    f.price_km                                                              AS actual_price_km,
    c.p25_trip,
    c.median_trip,
    c.p75_trip,
    c.p25_km,
    c.median_km,
    c.p75_km,
    c.trips_count,
    c.carriers_count,
    c.reliability_level,
    CASE WHEN c.median_trip = 0 THEN NULL
         ELSE (f.price_rub - c.median_trip) / c.median_trip * 100 END      AS trip_vs_median_pct,
    CASE WHEN c.median_km = 0 THEN NULL
         ELSE (f.price_km - c.median_km) / c.median_km * 100 END           AS km_vs_median_pct,
    CASE WHEN f.price_km < c.p25_km THEN 'BELOW_P25'
         WHEN f.price_km > c.p75_km THEN 'ABOVE_P75'
         ELSE 'IN_CORRIDOR' END                                             AS corridor_band
FROM facts f
JOIN mart.mart_route_card c
    ON c.month = f.month
   AND c.loading_city_gcid = f.loading_city_gcid
   AND c.unloading_city_gcid = f.unloading_city_gcid
   AND c.body_group = f.body_group
   AND c.weight_category = f.weight_category
JOIN core.cities cl ON cl.gc_id = f.loading_city_gcid
JOIN core.cities cu ON cu.gc_id = f.unloading_city_gcid
ORDER BY f.price_id;
