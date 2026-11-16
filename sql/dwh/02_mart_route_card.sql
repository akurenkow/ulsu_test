-- ============================================================================
-- Эталонный SQL расчёта витрины mart.mart_route_card (карточка маршрута и
-- коридор рыночных ставок). PostgreSQL 17, БД `dwh`.
--
-- Как прогнать вручную (просмотр результата без записи в таблицу):
--   psql -U dwh_user -d dwh -f sql/dwh/02_mart_route_card.sql
--
-- Пайплайн (`etl/marts.py`) использует этот же SELECT как источник для
-- `INSERT INTO mart.mart_route_card` (атомарная перезапись). Дата снимка задана
-- литералом '2026-10-26' (значение AS_OF из окружения); при пересчёте за другую
-- дату `etl/marts.py` подставляет её вместо литерала.
--
-- Логика (соответствует specs/step_03_dwh_mart_layer.md и assignment.md, Вопрос 2):
-- * зерно = месяц × маршрут × группа кузова × весовая категория;
-- * месяц строится из price_dates.price_date (долгосрочная заявка попадает в
--   несколько месяцев), будущие месяцы (price_date > as_of) исключены;
-- * города — из дедуплицированного core.cities (одна строка на gc_id, минимальный id);
-- * группа кузова: price_body_types (по std_id родителя) → fallback prices.body_type
--   (уже std_id) → «Не указан»; дублирование цены внутри группы схлопывается;
-- * фильтры: is_fact (price_type IN 1,2,3,4,5,9), complex = false, price > 0,
--   distance > 0, price_rub > 0, outlier_mark = false.
-- ============================================================================

WITH
-- ---------- 1. Унификация групп кузова ----------
body_group AS (
    -- (a) приоритет: price_body_types -> body_types.std_id -> бизнес-группа.
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
    -- (b) fallback: нет строк в price_body_types -> prices.body_type (уже std_id).
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

-- ---------- 2. Развёртка цен по месяцам (одна строка на price_id × месяц) ----------
price_months AS (
    SELECT DISTINCT
        price_id,
        date_trunc('month', price_date)::date AS month
    FROM core.price_dates
    WHERE price_date <= DATE '2026-10-26'          -- без плановых месяцев будущего
),

-- ---------- 3. Факты: цена × месяц × группа кузова, с фильтрами выборки ----------
facts AS (
    SELECT
        p.price_id,
        p.loading_city_gcid,
        p.unloading_city_gcid,
        bg.body_group,
        p.weight_category,
        p.volume_category,
        p.price_rub,
        p.distance,
        p.organization_id,
        p.price_type,
        pm.month
    FROM core.prices p
    JOIN body_group bg ON bg.price_id = p.price_id
    JOIN price_months pm ON pm.price_id = p.price_id
    WHERE p.price_type IN (1, 2, 3, 4, 5, 9)        -- is_fact = true
      AND p.complex = false                          -- только прямые маршруты
      AND p.price > 0
      AND p.distance > 0
      AND p.price_rub IS NOT NULL
      AND p.price_rub > 0
      AND p.outlier_mark = false
),

-- ---------- 4. Агрегация по ячейкам зерна ----------
cells AS (
    SELECT
        month,
        loading_city_gcid,
        unloading_city_gcid,
        body_group,
        weight_category,
        percentile_cont(0.25) WITHIN GROUP (ORDER BY price_rub)                       AS p25_trip,
        percentile_cont(0.50) WITHIN GROUP (ORDER BY price_rub)                       AS median_trip,
        percentile_cont(0.75) WITHIN GROUP (ORDER BY price_rub)                       AS p75_trip,
        avg(price_rub)                                                                AS avg_trip,
        percentile_cont(0.25) WITHIN GROUP (ORDER BY price_rub / NULLIF(distance, 0)) AS p25_km,
        percentile_cont(0.50) WITHIN GROUP (ORDER BY price_rub / NULLIF(distance, 0)) AS median_km,
        percentile_cont(0.75) WITHIN GROUP (ORDER BY price_rub / NULLIF(distance, 0)) AS p75_km,
        count(DISTINCT price_id)                                                      AS trips_count,
        count(DISTINCT organization_id) FILTER (WHERE price_type IN (3, 9))           AS carriers_count,
        CASE WHEN count(DISTINCT price_id) < 2 THEN NULL
             ELSE stddev_samp(price_rub) / NULLIF(avg(price_rub), 0) END              AS cv_rate,
        mode() WITHIN GROUP (ORDER BY volume_category)                                AS volume_category
    FROM facts
    GROUP BY month, loading_city_gcid, unloading_city_gcid, body_group, weight_category
)

-- ---------- 5. Карточка маршрута: имена городов + MoM/YoY + Data Trust ----------
SELECT
    c.month,
    c.loading_city_gcid,
    c.unloading_city_gcid,
    cl.name AS loading_city_name,
    cu.name AS unloading_city_name,
    c.body_group,
    c.weight_category,
    c.volume_category,
    c.p25_trip,
    c.median_trip,
    c.p75_trip,
    c.avg_trip,
    c.p25_km,
    c.median_km,
    c.p75_km,
    c.trips_count,
    c.carriers_count,
    c.cv_rate,
    CASE WHEN pm.median_trip IS NULL OR pm.median_trip = 0 THEN NULL
         ELSE (c.median_trip - pm.median_trip) / pm.median_trip * 100 END AS mom_pct,
    CASE WHEN py.median_trip IS NULL OR py.median_trip = 0 THEN NULL
         ELSE (c.median_trip - py.median_trip) / py.median_trip * 100 END AS yoy_pct,
    CASE WHEN c.trips_count >= 30 AND c.carriers_count >= 5
              AND (DATE '2026-10-26' - c.month) <= 30 THEN 'HIGH'
         WHEN c.trips_count >= 10 THEN 'MEDIUM'
         ELSE 'LOW_DATA' END AS reliability_level
FROM cells c
JOIN core.cities cl ON cl.gc_id = c.loading_city_gcid
JOIN core.cities cu ON cu.gc_id = c.unloading_city_gcid
LEFT JOIN cells pm
    ON pm.month = (c.month - interval '1 month')::date
   AND pm.loading_city_gcid = c.loading_city_gcid
   AND pm.unloading_city_gcid = c.unloading_city_gcid
   AND pm.body_group = c.body_group
   AND pm.weight_category = c.weight_category
LEFT JOIN cells py
    ON py.month = (c.month - interval '1 year')::date
   AND py.loading_city_gcid = c.loading_city_gcid
   AND py.unloading_city_gcid = c.unloading_city_gcid
   AND py.body_group = c.body_group
   AND py.weight_category = c.weight_category
ORDER BY c.month, c.loading_city_gcid, c.unloading_city_gcid, c.body_group, c.weight_category;
