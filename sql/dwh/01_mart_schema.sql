-- ============================================================================
-- Mart Layer (слой `mart`) и reference-таблицы этапа 3 — DDL.
-- PostgreSQL 17. Выполняется идемпотентно (`CREATE ... IF NOT EXISTS`).
--
-- Назначение:
-- * `core.price_body_types` / `core.price_dates` — копии таблиц операционного
--   источника (`public.price_body_types`, `public.price_dates`), которые
--   переносятся в хранилище, чтобы весь расчёт витрины выполнялся в одной БД
--   `dwh` чистым SQL (PostgreSQL не умеет JOIN через базы без FDW).
-- * `mart.mart_route_card` — витрина «Карточка маршрута и коридор рыночных
--   ставок» (эталонный расчёт — `02_mart_route_card.sql`).
-- * `mart.ml_price_predictions` — витрина прогнозов тарифа «Факт vs Прогноз»
--   (этап 4; наполняется `ml/pipeline.py`).
--
-- Стратегия обновления `mart_route_card`: атомарная перезапись (DELETE + INSERT
-- в одной транзакции) в `etl/marts.py`. PRIMARY KEY по зерну витрины — страховка
-- от дублей при любом повторном прогоне.
--
-- Порядок колонок `mart_route_card` совпадает с SELECT из
-- `02_mart_route_card.sql` и с DDL ClickHouse `sql/clickhouse/00_schema.sql`,
-- чтобы переливка PostgreSQL → ClickHouse была тривиальной.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- core.price_body_types — копия public.price_body_types (price_id × body_type_id).
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS core.price_body_types (
    price_id     bigint  NOT NULL,
    body_type_id integer NOT NULL,
    CONSTRAINT core_price_body_types_pkey PRIMARY KEY (price_id, body_type_id)
);

CREATE INDEX IF NOT EXISTS core_price_body_types_body_idx
    ON core.price_body_types (body_type_id);

-- ----------------------------------------------------------------------------
-- core.price_dates — копия public.price_dates (помесячная развёртка долгосрочных
-- заявок от loading_date до end_loading_date). Источник зерна месяца витрины.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS core.price_dates (
    price_id   bigint NOT NULL,
    price_date date   NOT NULL,
    CONSTRAINT core_price_dates_pkey PRIMARY KEY (price_id, price_date)
);

CREATE INDEX IF NOT EXISTS core_price_dates_date_idx
    ON core.price_dates (price_date);

-- ----------------------------------------------------------------------------
-- mart.mart_route_card — витрина «Карточка маршрута и коридор рыночных ставок».
-- Зерно: месяц × город отправления × город назначения × группа кузова ×
--        весовая категория.
-- `volume_category` — атрибут (модальное значение ячейки), НЕ входит в grain.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS mart.mart_route_card (
    month               date             NOT NULL,   -- date_trunc('month', price_date)
    loading_city_gcid   integer          NOT NULL,
    unloading_city_gcid integer          NOT NULL,
    loading_city_name   varchar(255)     NOT NULL,
    unloading_city_name varchar(255)     NOT NULL,
    body_group          varchar(32)      NOT NULL,   -- Тент|Рефрижератор|Изотерм|Борт|Прочие|Не указан
    weight_category     varchar(4)       NOT NULL,   -- G20T|FTL|10T|5T|L5T|OTH
    volume_category     varchar(4),                   -- FTL|OTH (атрибут, не в grain)
    p25_trip            numeric(20,2)     NOT NULL,   -- руб./рейс
    median_trip         numeric(20,2)     NOT NULL,   -- руб./рейс
    p75_trip            numeric(20,2)     NOT NULL,   -- руб./рейс
    avg_trip            numeric(20,2)     NOT NULL,   -- руб./рейс
    p25_km              numeric(20,4)     NOT NULL,   -- руб./км
    median_km           numeric(20,4)     NOT NULL,   -- руб./км
    p75_km              numeric(20,4)     NOT NULL,   -- руб./км
    trips_count         bigint           NOT NULL,   -- COUNT(DISTINCT price_id)
    carriers_count      bigint           NOT NULL,   -- COUNT(DISTINCT organization_id) при price_type IN (3,9)
    cv_rate             numeric(20,6),                -- stddev_samp(price_rub)/avg(price_rub); NULL при <2 наблюдений
    mom_pct             numeric(20,4),                -- изменение P50 к календарному прошлому месяцу, %
    yoy_pct             numeric(20,4),                -- изменение P50 к тому же месяцу прошлого года, %
    reliability_level   varchar(16)      NOT NULL,   -- HIGH | MEDIUM | LOW_DATA
    CONSTRAINT mart_route_card_pkey PRIMARY KEY
        (month, loading_city_gcid, unloading_city_gcid, body_group, weight_category)
);

-- Индексы под фильтры BI:
-- * карточка одного маршрута (помесячная история направления/кузова/веса);
-- * сводный анализ по периоду.
CREATE INDEX IF NOT EXISTS mart_route_card_route_idx
    ON mart.mart_route_card (loading_city_gcid, unloading_city_gcid, body_group, weight_category, month);

CREATE INDEX IF NOT EXISTS mart_route_card_month_idx
    ON mart.mart_route_card (month);

-- Миграция float -> decimal (точные деньги/коридоры без артефактов float8).
ALTER TABLE mart.mart_route_card ALTER COLUMN p25_trip    TYPE numeric(20,2) USING p25_trip::numeric(20,2);
ALTER TABLE mart.mart_route_card ALTER COLUMN median_trip TYPE numeric(20,2) USING median_trip::numeric(20,2);
ALTER TABLE mart.mart_route_card ALTER COLUMN p75_trip    TYPE numeric(20,2) USING p75_trip::numeric(20,2);
ALTER TABLE mart.mart_route_card ALTER COLUMN avg_trip    TYPE numeric(20,2) USING avg_trip::numeric(20,2);
ALTER TABLE mart.mart_route_card ALTER COLUMN p25_km      TYPE numeric(20,4) USING p25_km::numeric(20,4);
ALTER TABLE mart.mart_route_card ALTER COLUMN median_km   TYPE numeric(20,4) USING median_km::numeric(20,4);
ALTER TABLE mart.mart_route_card ALTER COLUMN p75_km      TYPE numeric(20,4) USING p75_km::numeric(20,4);
ALTER TABLE mart.mart_route_card ALTER COLUMN cv_rate     TYPE numeric(20,6) USING cv_rate::numeric(20,6);
ALTER TABLE mart.mart_route_card ALTER COLUMN mom_pct     TYPE numeric(20,4) USING mom_pct::numeric(20,4);
ALTER TABLE mart.mart_route_card ALTER COLUMN yoy_pct     TYPE numeric(20,4) USING yoy_pct::numeric(20,4);

-- ----------------------------------------------------------------------------
-- mart.ml_price_predictions — витрина прогнозов тарифа (этап 4): «Факт vs Прогноз».
-- Ключ идемпотентности — (price_id, model_name): повторный скоринг UPSERT'ом
-- не дублирует строки. Для будущих заявок actual_price_rub/abs_error_rub/ape_pct = NULL.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS mart.ml_price_predictions (
    price_id            bigint           NOT NULL,   -- ссылка на core.prices.price_id
    loading_date        date             NOT NULL,   -- дата погрузки
    loading_city_gcid   integer          NOT NULL,   -- маршрут (отправление)
    unloading_city_gcid integer          NOT NULL,   -- маршрут (назначение)
    body_group          varchar(32)      NOT NULL,   -- стандартизированная группа кузова
    actual_price_rub    numeric(20,2),            -- факт (NULL для будущих заявок)
    predicted_price_rub numeric(20,2) NOT NULL,   -- точечный прогноз тарифа (MAPE-модель по умолчанию), руб. без НДС
    predicted_price_rub_lo numeric(20,2) NOT NULL, -- нижняя граница интервала (q25)
    predicted_price_rub_hi numeric(20,2) NOT NULL, -- верхняя граница интервала (q75)
    abs_error_rub       numeric(20,2),            -- |факт - прогноз|, руб. (NULL для будущих)
    ape_pct             numeric(20,4),            -- |(факт - прогноз)/факт| * 100 (NULL для будущих)
    model_name          varchar(64)      NOT NULL,   -- имя модели (например, 'catboost')
    scored_at           timestamptz      NOT NULL,   -- время скоринга
    CONSTRAINT ml_price_predictions_pkey PRIMARY KEY (price_id, model_name)
);

-- Идемпотентное расширение уже созданных витрин (квантильный интервал lo/hi).
ALTER TABLE mart.ml_price_predictions ADD COLUMN IF NOT EXISTS predicted_price_rub_lo numeric(20,2);
ALTER TABLE mart.ml_price_predictions ADD COLUMN IF NOT EXISTS predicted_price_rub_hi numeric(20,2);

-- Миграция float -> decimal (точные прогнозы/ошибки без артефактов float8).
ALTER TABLE mart.ml_price_predictions ALTER COLUMN actual_price_rub        TYPE numeric(20,2) USING actual_price_rub::numeric(20,2);
ALTER TABLE mart.ml_price_predictions ALTER COLUMN predicted_price_rub     TYPE numeric(20,2) USING predicted_price_rub::numeric(20,2);
ALTER TABLE mart.ml_price_predictions ALTER COLUMN predicted_price_rub_lo  TYPE numeric(20,2) USING predicted_price_rub_lo::numeric(20,2);
ALTER TABLE mart.ml_price_predictions ALTER COLUMN predicted_price_rub_hi  TYPE numeric(20,2) USING predicted_price_rub_hi::numeric(20,2);
ALTER TABLE mart.ml_price_predictions ALTER COLUMN abs_error_rub           TYPE numeric(20,2) USING abs_error_rub::numeric(20,2);
ALTER TABLE mart.ml_price_predictions ALTER COLUMN ape_pct                 TYPE numeric(20,4) USING ape_pct::numeric(20,4);

CREATE INDEX IF NOT EXISTS ml_price_predictions_loading_date_idx
    ON mart.ml_price_predictions (loading_date);

CREATE INDEX IF NOT EXISTS ml_price_predictions_route_idx
    ON mart.ml_price_predictions (loading_city_gcid, unloading_city_gcid, body_group);

-- ----------------------------------------------------------------------------
-- mart.fact_vs_corridor — «Факт × рыночный коридор» (этап 3): каждая закрытая
-- сделка сопоставляется с коридором P25–P50–P75 своей рыночной ячейки
-- (месяц погрузки × маршрут × кузов × вес). Источник коридора — mart_route_card.
-- Зерно — price_id (одна строка на закрытую сделку).
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS mart.fact_vs_corridor (
    price_id             bigint           NOT NULL,      -- = core.prices.price_id
    loading_date         date             NOT NULL,      -- дата погрузки
    loading_city_gcid    integer          NOT NULL,
    unloading_city_gcid  integer          NOT NULL,
    loading_city_name    varchar(255)     NOT NULL,
    unloading_city_name  varchar(255)     NOT NULL,
    body_group           varchar(32)      NOT NULL,      -- цена с несколькими кузовами — по строке на группу
    weight_category      varchar(4)       NOT NULL,
    distance             numeric(12,2)    NOT NULL,      -- км
    actual_price_rub     numeric(20,2)    NOT NULL,      -- факт, руб./рейс
    actual_price_km      numeric(20,4)    NOT NULL,      -- факт, руб./км
    p25_trip             numeric(20,2)    NOT NULL,      -- коридор, руб./рейс
    median_trip          numeric(20,2)    NOT NULL,
    p75_trip             numeric(20,2)    NOT NULL,
    p25_km               numeric(20,4)    NOT NULL,      -- коридор, руб./км
    median_km            numeric(20,4)    NOT NULL,
    p75_km               numeric(20,4)    NOT NULL,
    trips_count          bigint           NOT NULL,      -- Data Trust ячейки
    carriers_count       bigint           NOT NULL,
    reliability_level    varchar(16)      NOT NULL,
    trip_vs_median_pct   numeric(20,4),                  -- (факт - P50)/P50, % за рейс
    km_vs_median_pct     numeric(20,4),                  -- (факт - P50)/P50, % за км
    corridor_band        varchar(16)      NOT NULL,      -- BELOW_P25 | IN_CORRIDOR | ABOVE_P75
    CONSTRAINT fact_vs_corridor_pkey PRIMARY KEY (price_id, body_group)
);

CREATE INDEX IF NOT EXISTS fact_vs_corridor_loading_date_idx
    ON mart.fact_vs_corridor (loading_date);

CREATE INDEX IF NOT EXISTS fact_vs_corridor_route_idx
    ON mart.fact_vs_corridor (loading_city_gcid, unloading_city_gcid, body_group, weight_category);

-- ----------------------------------------------------------------------------
-- mart.v_route_card_facts — неагрегированные факты витрины mart_route_card
-- (одна строка на price_id × месяц × группу кузова). Используется Superset'ом
-- для честной сводной (пульной) медианы/квартилей: `AVG(median_trip)` по
-- предпосчитанным ячейкам — это «среднее по медианам», а не рыночная медиана.
-- Пульная медиана считается `PERCENTILE_CONT` по исходным `price_rub`/`price_km`,
-- поэтому каждая сделка учитывается ровно один раз (взвешивание по числу рейсов
-- происходит автоматически).
--
-- Универсум, группы кузова и месяц идентичны `02_mart_route_card.sql`; дата
-- снимка зафиксирована тем же литералом, что и в эталонных SQL (etl/marts.py
-- подставляет AS_OF при пересчёте витрин).
-- ----------------------------------------------------------------------------
CREATE OR REPLACE VIEW mart.v_route_card_facts AS
WITH body_group AS (
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
price_months AS (
    SELECT DISTINCT
        price_id,
        date_trunc('month', price_date)::date AS month
    FROM core.price_dates
    WHERE price_date <= DATE '2026-10-26'
)
SELECT
    pm.month,
    p.loading_city_gcid,
    p.unloading_city_gcid,
    cl.name AS loading_city_name,
    cu.name AS unloading_city_name,
    bg.body_group,
    p.weight_category,
    p.price_rub,
    (p.price_rub / NULLIF(p.distance, 0)) AS price_km
FROM core.prices p
JOIN body_group bg ON bg.price_id = p.price_id
JOIN price_months pm ON pm.price_id = p.price_id
JOIN core.cities cl ON cl.gc_id = p.loading_city_gcid
JOIN core.cities cu ON cu.gc_id = p.unloading_city_gcid
WHERE p.price_type IN (1, 2, 3, 4, 5, 9)
  AND p.complex = false
  AND p.price > 0
  AND p.distance > 0
  AND p.price_rub IS NOT NULL
  AND p.price_rub > 0
  AND p.outlier_mark = false;
