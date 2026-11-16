#!/usr/bin/env bash
# Идемпотентная инициализация операционного источника deals_bigdata.
# Применяет эталонные sql/schema.sql + sql/synthetic.sql (смонтированы в /opt/sql).
set -e
PSQL_USER="${POSTGRES_USER:-postgres}"
PSQL_DB="${POSTGRES_DB:-deals_bigdata}"
DWH_USER="${DWH_USER:-dwh_user}"
N_PRICES="${N_PRICES:-300000}"

# Защита от повторной инициализации: пропускаем, если таблица prices уже наполнена.
table_exists=$(psql -tA -U "${PSQL_USER}" -d "${PSQL_DB}" \
  -c "SELECT to_regclass('public.prices') IS NOT NULL;" 2>/dev/null || echo f)

if [ "${table_exists}" = "t" ]; then
  rows=$(psql -tA -U "${PSQL_USER}" -d "${PSQL_DB}" \
    -c "SELECT count(*) FROM prices;" 2>/dev/null || echo 0)
  if [ "${rows:-0}" -gt 0 ]; then
    echo "[init-source] источник уже инициализирован (prices=${rows}) — пропускаю."
    exit 0
  fi
fi

echo "[init-source] применяю sql/schema.sql"
psql -v ON_ERROR_STOP=1 -U "${PSQL_USER}" -d "${PSQL_DB}" -f /opt/sql/schema.sql

echo "[init-source] применяю sql/synthetic.sql (n_prices=${N_PRICES})"
psql -v ON_ERROR_STOP=1 -U "${PSQL_USER}" -d "${PSQL_DB}" \
  -v n_prices="${N_PRICES}" -f /opt/sql/synthetic.sql

echo "[init-source] выдаю чтение источника роли '${DWH_USER}'"
psql -v ON_ERROR_STOP=1 -U "${PSQL_USER}" -d "${PSQL_DB}" -v dwh_user="${DWH_USER}" <<'SQL'
GRANT USAGE ON SCHEMA public TO :"dwh_user";
GRANT SELECT ON ALL TABLES IN SCHEMA public TO :"dwh_user";
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO :"dwh_user";
SQL

rows=$(psql -tA -U "${PSQL_USER}" -d "${PSQL_DB}" -c 'SELECT count(*) FROM prices;')
echo "[init-source] готово: prices=${rows} (n_prices=${N_PRICES})"
