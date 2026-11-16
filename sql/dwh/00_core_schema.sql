-- ============================================================================
-- Core DWH (слой `core`) хранилища `dwh` — DDL этапа 2 (ETL & Data Quality).
-- PostgreSQL 17. Выполняется идемпотентно (`CREATE ... IF NOT EXISTS`).
--
-- Источник (не модифицируется): БД `deals_bigdata`, схема `public`
--   (sql/schema.sql + sql/synthetic.sql).
-- Здесь строится очищенный базовый слой: нормализованная цена `price_rub`,
-- заполненные категории груза и карантин отбракованных строк.
--
-- Ключевые решения:
-- * `core.prices.price_id` = `deals_bigdata.public.prices.id` (естественный ключ).
--   Идемпотентность загрузки обеспечивает `INSERT ... ON CONFLICT (price_id) DO UPDATE`.
-- * `core.cities` — дедуплицированный справочник: одна строка на `gc_id`
--   (минимальный `cities.id`). Используется на этапе 3 (март) и этапе 4 (ML).
-- * `core.exchange_rates` — копия курсов ЦБ; уникальность (num_code, setting_date)
--   для точной конвертации валют по `nearest_rate_date`.
-- * `core.body_groups` — канонические бизнес-группы кузова по `std_id`.
--   Fallback-группы «Прочие» (остальные std_id) и «Не указан» (NULL) не являются
--   строками справочника — это логика маппинга на этапе 3.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- core.prices — очищенные сделки с ценой в RUB и заполненными категориями.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS core.prices (
    price_id            bigint           PRIMARY KEY,      -- = public.prices.id
    loading_date        date             NOT NULL,
    created_at          timestamp(0),
    updated_at          timestamp(0),
    subscriber_id       integer,
    organization_id     integer,
    loading_city_gcid   integer          NOT NULL,
    unloading_city_gcid integer          NOT NULL,
    distance            numeric(12,2)     NOT NULL,
    total_weight        numeric(12,2)     NOT NULL,
    total_volume        numeric(12,2)     NOT NULL,
    body_type           integer,
    price               numeric(20,2)     NOT NULL,         -- ставка в исходной валюте
    currency_code       varchar(3)       NOT NULL,
    nearest_rate_date   date,
    rate_value          numeric(20,6)     NOT NULL,         -- применённый курс (643 -> 1.0)
    price_rub           numeric(20,2)     NOT NULL,         -- ставка в RUB без НДС
    complex             boolean          NOT NULL,
    price_type          smallint         NOT NULL,
    is_fact             boolean          NOT NULL,
    outlier_mark        boolean          NOT NULL,
    weight_category     varchar(4)       NOT NULL,         -- заполнена (G20T..OTH)
    volume_category     varchar(4)       NOT NULL,         -- заполнена (FTL/OTH)
    source              varchar(255),
    item_id             integer,
    meta_item_id        bigint,
    end_loading_date    date,
    etl_batch_id        text             NOT NULL,         -- окно загрузки (provenance)
    etl_loaded_at       timestamptz      NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS core_prices_loading_date_idx ON core.prices (loading_date);
CREATE INDEX IF NOT EXISTS core_prices_route_idx
    ON core.prices (loading_city_gcid, unloading_city_gcid);

-- Миграция float -> decimal: точные деньги/меры без артефактов float8.
ALTER TABLE core.prices ALTER COLUMN distance     TYPE numeric(12,2) USING distance::numeric(12,2);
ALTER TABLE core.prices ALTER COLUMN total_weight TYPE numeric(12,2) USING total_weight::numeric(12,2);
ALTER TABLE core.prices ALTER COLUMN total_volume TYPE numeric(12,2) USING total_volume::numeric(12,2);
ALTER TABLE core.prices ALTER COLUMN price        TYPE numeric(20,2) USING price::numeric(20,2);
ALTER TABLE core.prices ALTER COLUMN rate_value   TYPE numeric(20,6) USING rate_value::numeric(20,6);
ALTER TABLE core.prices ALTER COLUMN price_rub    TYPE numeric(20,2) USING price_rub::numeric(20,2);

-- ----------------------------------------------------------------------------
-- core.cities — дедуплицированный справочник населённых пунктов (1 строка на gc_id).
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS core.cities (
    id              bigint        PRIMARY KEY,             -- минимальный public.cities.id
    full_name_local varchar(255)  NOT NULL,
    full_name_gc    varchar(255)  NOT NULL,
    gc_id           integer       NOT NULL,
    name            varchar(255)  NOT NULL,
    longitude       double precision,
    latitude        double precision,
    created_at      timestamp(0),
    updated_at      timestamp(0),
    CONSTRAINT core_cities_gc_id_key UNIQUE (gc_id)
);

-- ----------------------------------------------------------------------------
-- core.body_types — копия справочника кузовов 1С с маппингом на std_id.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS core.body_types (
    id          bigint        PRIMARY KEY,
    name        varchar(255)  NOT NULL,
    id_from_1c  integer       NOT NULL,
    std_id      integer
);

-- ----------------------------------------------------------------------------
-- core.body_groups — канонические бизнес-группы кузова (grain витрин этапа 3).
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS core.body_groups (
    std_id      integer       PRIMARY KEY,                 -- 56/4/2/15
    group_name  varchar(32)   NOT NULL
);

INSERT INTO core.body_groups (std_id, group_name) VALUES
    (56, 'Тент'),
    (4,  'Рефрижератор'),
    (2,  'Изотерм'),
    (15, 'Борт')
ON CONFLICT (std_id) DO UPDATE SET group_name = EXCLUDED.group_name;

-- ----------------------------------------------------------------------------
-- core.exchange_rates — копия курсов ЦБ РФ (для конвертации 840/978 -> RUB).
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS core.exchange_rates (
    id           bigint        PRIMARY KEY,
    num_code     varchar(3)    NOT NULL,
    value        double precision NOT NULL,
    setting_date date          NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS core_exchange_rates_code_date_idx
    ON core.exchange_rates (num_code, setting_date);

-- ----------------------------------------------------------------------------
-- core.dq_quarantine_prices — карантин отбракованных строк (по одной строке на
-- (price_id, reason_code), чтобы статистика отсева считалась по каждой причине).
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS core.dq_quarantine_prices (
    id                  bigserial       PRIMARY KEY,
    price_id            bigint          NOT NULL,
    reason_code         varchar(64)     NOT NULL,
    raw_price           double precision,
    currency_code       varchar(3),
    nearest_rate_date   date,
    loading_date        date,
    loading_city_gcid   integer,
    unloading_city_gcid integer,
    distance            double precision,
    total_weight        double precision,
    total_volume        double precision,
    body_type           integer,
    price_type          smallint,
    detected_at         timestamptz     NOT NULL DEFAULT now(),
    batch_id            text            NOT NULL,
    CONSTRAINT dq_quarantine_price_reason_key UNIQUE (price_id, reason_code)
);

CREATE INDEX IF NOT EXISTS dq_quarantine_loading_date_idx
    ON core.dq_quarantine_prices (loading_date);
CREATE INDEX IF NOT EXISTS dq_quarantine_reason_code_idx
    ON core.dq_quarantine_prices (reason_code);
