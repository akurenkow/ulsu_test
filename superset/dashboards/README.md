# Экспорт демо-дашборда Superset

Сюда кладётся экспорт демо-дашборда (`demo.zip`), который автоматически
импортируется при старте сервиса `superset` (см. `superset/superset-init.sh`).

## Состав `demo.zip`

Экспорт содержит всё необходимое для воспроизведения дашборда на чистой машине:

- **database** `dwh` (`postgresql+psycopg2://dwh_user:***@postgres:5432/dwh`);
- **datasets** `dwh.mart.mart_route_card`, `dwh.mart.ml_price_predictions`,
  `dwh.mart.fact_vs_corridor` и `dwh.mart.v_route_card_facts` (view с неагрегированными
  фактами — для честной сводной/пульной медианы);
- **charts (9 шт.):**
  1. «Карточка маршрута: коридор и Data Trust» — таблица (raw): месяц, города,
     группа кузова, весовая категория, коридор `p25/p50/p75` за рейс и за км,
     динамика `mom_pct`/`yoy_pct`, `trips_count`, `carriers_count`, `cv_rate`,
     `reliability_level`;
  2. «Медиана ставки (руб./рейс) по месяцам» — временной график пульной медианы
     `PERCENTILE_CONT(0.5) ... (ORDER BY price_rub)` по `v_route_card_facts`
     (вместо `AVG(median_trip)` — среднего по медианам);
  3. «Коридор ставок P25–P50–P75 по группе кузова» — пульные квартили
     `PERCENTILE_CONT(0.25/0.5/0.75)` по `price_rub` и `price_km` в разрезе
     `body_group`;
  4. «Факт vs Прогноз тарифа» — таблица по `mart.ml_price_predictions`, включая
     доверительный интервал прогноза `predicted_price_rub_lo`/`predicted_price_rub_hi`;
  5. «Коридор ставки за рейс по маршруту (помесячно)» — линейный график
     `p25/p50/p75` с drill-down по нативным фильтрам;
  6. «Точность модели: MAPE (тест)» — KPI (big number) `AVG(ape_pct)` по
     `mart.ml_price_predictions` (MAPE на отложенной тестовой выборке);
  7. «Рейсы существенно выше прогноза» — таблица с фильтром
     `actual_price_rub > predicted_price_rub AND ape_pct > 20`;
  8. «Факт vs коридор P25–P75» — таблица по `mart.fact_vs_corridor`: факт сделки,
     коридор ячейки, отклонение к медиане и зона (`BELOW_P25` / `IN_CORRIDOR` /
     `ABOVE_P75`);
  9. «Рейсы по зонам коридора» — агрегат `count(*)` по `corridor_band`.
- **native filters (4 шт.):** Отправление, Назначение, Кузов, Весовая категория
  (по `mart_route_card`, `fact_vs_corridor` и `v_route_card_facts`).
- **dashboard** «Тарифы и коридор ставок (демо)».

## Как пересобрать экспорт (после изменения чартов)

1. Изменить датасеты/чарты/дашборд в UI Superset.
2. Экспортировать все дашборды в контейнере:

   ```bash
   docker compose --profile superset exec superset \
     superset export-dashboards --username admin -f /tmp/demo.zip
   docker compose --profile superset cp superset:/tmp/demo.zip ./dashboards/demo.zip
   ```

   (в Superset 4.0 `export-dashboards` экспортирует все дашборды инстанса и
   требует `--username`; перед повторным импортом вручную лишние дашборды
   удаляются из экспорта).

3. Пароль БД в экспорте замаскирован (`XXXXXXXXXX`). При старте он восстанавливается
   из окружения: `superset-init.sh` импортирует `demo.zip` через
   `superset/import_dashboards.py` (подставляет пароль из `SUPERSET_DWH_URI`),
   затем вызывает `set-database-uri` для приведения URI к значению из окружения.

Импорт на чистой машине происходит автоматически при `docker compose --profile superset up -d`.
