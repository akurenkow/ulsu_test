# Apache Superset (бонус ★)

Публикация аналитических витрин (`dwh.mart.mart_route_card`, `dwh.mart.ml_price_predictions`)
в Apache Superset. Демо-дашборд поставляется готовым экспортом и импортируется
автоматически при первом запуске — ручная настройка чартов не требуется.

## Запуск

```bash
docker compose --profile superset up -d
```

UI: http://localhost:8088 (порт — `SUPERSET_HOST_PORT`, по умолчанию `8088`).
Логин/пароль — `SUPERSET_ADMIN_USER` / `SUPERSET_ADMIN_PASSWORD` (по умолчанию `admin`/`admin`).

После старта в UI доступен дашборд **«Тарифы и коридор ставок (демо)»** с девятью
чартами и нативными фильтрами (направление, кузов, весовая категория):

- карточка маршрута (коридор `p25/p50/p75` за рейс и за км, динамика MoM/YoY, Data Trust);
- медиана ставки по месяцам — пульная медиана (`PERCENTILE_CONT`) по фактам
  `mart.v_route_card_facts`, а не `AVG(median_trip)`;
- коридор P25–P50–P75 по группе кузова — пульные квартили по фактам;
- коридор ставки за рейс по маршруту (помесячно, drill-down через фильтры);
- «Факт vs Прогноз» по `mart.ml_price_predictions` (включая доверительный интервал
  `predicted_price_rub_lo`/`predicted_price_rub_hi`);
- точность модели — KPI «MAPE на отложенном тесте» (одно число `AVG(ape_pct)`);
- «Рейсы существенно выше прогноза» (факт > прогноз и APE > 20%);
- «Факт vs коридор P25–P75» по `mart.fact_vs_corridor` (отклонение факта от коридора);
- «Рейсы по зонам коридора» (ниже P25 / в коридоре / выше P75).

Сводные чарты («Медиана по месяцам», «Коридор по кузовам») читают view
`mart.v_route_card_facts` (см. `sql/dwh/01_mart_schema.sql`): это неагрегированные
факты той же выборки, что и `mart_route_card`, поэтому пульная медиана/квартили
считаются по исходным сделкам, а не усреднением предпосчитанных медиан ячеек.

## Структура

```text
superset/
├── superset-init.sh        # идемпотентный bootstrap (db upgrade, admin, init, импорт, run)
├── README.md               # этот файл
└── dashboards/
    ├── demo.zip            # экспорт дашборда/чартов/датасетов/БД (импортируется автоматически)
    └── README.md           # состав экспорта и инструкция пересборки
```

## Как это работает (bootstrap)

`superset-init.sh` при каждом старте идемпотентно выполняет:

1. `superset db upgrade`, `create-admin`, `superset init`;
2. `python import_dashboards.py` — импортирует `dashboards/demo.zip`: создаёт/обновляет
   БД `dwh`, датасеты, чарты и дашборд (сущности совпадают по uuid, поэтому повторный
   импорт не дублирует). Пароль БД в экспорте замаскирован (`XXXXXXXXXX`), поэтому
   скрипт подставляет реальный пароль из `SUPERSET_DWH_URI`;
3. `set-database-uri --database_name dwh --uri $SUPERSET_DWH_URI` — подстраховка,
   приводит URI к значению из окружения (если `DWH_PASSWORD` переопределён).

URI формируется из переменных окружения (без паролей в коде):

```text
postgresql+psycopg2://<DWH_USER>:<DWH_PASSWORD>@postgres:5432/<DWH_DB>
```

Показатели дашборда сверяются с SQL-проверками витрин (`sql/dwh/`, этап 3).
