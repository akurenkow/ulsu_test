-- ClickHouse analytical layer (бонус ★).
-- Вариант A: полный перенос аналитических витрин PostgreSQL в ClickHouse.
-- PostgreSQL остаётся обязательным источником истины (ETL/Mart/ML); ClickHouse —
-- ускоритель BI-запросов. Результаты (число строк, суммы, перцентили) совпадают
-- с PostgreSQL.
--
-- Идемпотентность перезаливки: движок ReplacingMergeTree дедуплицирует по ключу
-- сортировки (grain витрины / ключ прогноза). После загрузки выполняется
-- OPTIMIZE ... FINAL (или чтение с FINAL), см. clickhouse/loader.py.

CREATE DATABASE IF NOT EXISTS dwh;

-- ---------------------------------------------------------------------------
-- mart_route_card — карточка маршрута и рыночный коридор ставок.
-- Ключ дедупликации = grain витрины:
--   месяц × город отправления × город назначения × группа кузова × весовая категория.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dwh.mart_route_card
(
    month               Date,                   -- месяц из price_dates.price_date (первое число)
    loading_city_gcid   UInt32,                 -- город отправления
    unloading_city_gcid UInt32,                 -- город назначения
    loading_city_name   String,                 -- наименование (атрибут)
    unloading_city_name String,                 -- наименование (атрибут)
    body_group          LowCardinality(String), -- группа кузова (grain)
    weight_category     LowCardinality(String), -- весовая категория (grain)
    volume_category     LowCardinality(String), -- объёмная категория (атрибут, не в grain)
    p25_trip            Float64,                -- руб./рейс
    median_trip         Float64,                -- руб./рейс
    p75_trip            Float64,                -- руб./рейс
    avg_trip            Float64,                -- руб./рейс
    p25_km              Float64,                -- руб./км
    median_km           Float64,                -- руб./км
    p75_km              Float64,                -- руб./км
    trips_count         UInt32,                 -- число уникальных сделок
    carriers_count      UInt32,                 -- число уникальных перевозчиков
    cv_rate             Nullable(Float64),      -- stddev_samp/avg (NULL при <2 наблюдений)
    mom_pct             Nullable(Float64),      -- изменение P50 к прошлому месяцу, %
    yoy_pct             Nullable(Float64),      -- изменение P50 к прошлому году, %
    reliability_level   LowCardinality(String)  -- HIGH | MEDIUM | LOW_DATA
)
ENGINE = ReplacingMergeTree
ORDER BY (month, loading_city_gcid, unloading_city_gcid, body_group, weight_category);

-- ---------------------------------------------------------------------------
-- ml_price_predictions — факт vs прогноз тарифа.
-- Ключ дедупликации = (price_id, model_name).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dwh.ml_price_predictions
(
    price_id            UInt64,                 -- ссылка на сделку
    model_name          LowCardinality(String), -- имя модели
    loading_date        Date,                   -- дата погрузки
    loading_city_gcid   UInt32,
    unloading_city_gcid UInt32,
    body_group          LowCardinality(String),
    actual_price_rub    Nullable(Float64),      -- NULL для будущих заявок
    predicted_price_rub Float64,               -- точечный прогноз (MAPE-модель по умолчанию)
    predicted_price_rub_lo Float64,             -- нижняя граница интервала (q25)
    predicted_price_rub_hi Float64,             -- верхняя граница интервала (q75)
    abs_error_rub       Nullable(Float64),      -- NULL для будущих заявок
    ape_pct             Nullable(Float64),      -- NULL для будущих заявок
    scored_at           DateTime                 -- время скоринга
)
ENGINE = ReplacingMergeTree
ORDER BY (price_id, model_name);

-- Идемпотентное расширение уже созданных таблиц (квантильный интервал lo/hi).
ALTER TABLE dwh.ml_price_predictions ADD COLUMN IF NOT EXISTS predicted_price_rub_lo Float64;
ALTER TABLE dwh.ml_price_predictions ADD COLUMN IF NOT EXISTS predicted_price_rub_hi Float64;
