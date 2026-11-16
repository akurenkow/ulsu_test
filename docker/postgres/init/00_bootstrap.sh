#!/usr/bin/env bash
# Идемпотентный bootstrap хранилища: БД dwh, схемы raw/core/mart и прикладная роль.
# Выполняется при первой инициализации postgres-контейнера (docker-entrypoint-initdb.d).
set -e

PSQL_USER="${POSTGRES_USER:-postgres}"
PSQL_DB="${POSTGRES_DB:-deals_bigdata}"
DWH_DB="${DWH_DB:-dwh}"
DWH_USER="${DWH_USER:-dwh_user}"
DWH_PASSWORD="${DWH_PASSWORD:-dwh_password}"

echo "[bootstrap] создаю БД хранилища '${DWH_DB}' (если отсутствует)"
psql -v ON_ERROR_STOP=1 -U "${PSQL_USER}" -d "${PSQL_DB}" <<SQL
SELECT 'CREATE DATABASE "${DWH_DB}"'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '${DWH_DB}')
\gexec
SQL

echo "[bootstrap] создаю прикладную роль '${DWH_USER}'"
# NB: без dollar-quoting ($$), иначе psql не подставляет :'var'.
psql -v ON_ERROR_STOP=1 -U "${PSQL_USER}" -d "${PSQL_DB}" \
  -v dwh_user="${DWH_USER}" -v dwh_pw="${DWH_PASSWORD}" <<'SQL'
SELECT format('CREATE ROLE %I LOGIN PASSWORD %L', :'dwh_user', :'dwh_pw')
WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname = :'dwh_user')\gexec
SQL

echo "[bootstrap] выдаю CONNECT на БД '${DWH_DB}'"
psql -v ON_ERROR_STOP=1 -U "${PSQL_USER}" -d "${PSQL_DB}" \
  -v dwh_db="${DWH_DB}" -v dwh_user="${DWH_USER}" <<'SQL'
GRANT CONNECT ON DATABASE :"dwh_db" TO :"dwh_user";
SQL

echo "[bootstrap] создаю схемы raw/core/mart в '${DWH_DB}'"
psql -v ON_ERROR_STOP=1 -U "${PSQL_USER}" -d "${DWH_DB}" -v dwh_user="${DWH_USER}" <<'SQL'
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS core;
CREATE SCHEMA IF NOT EXISTS mart;
GRANT USAGE, CREATE ON SCHEMA raw, core, mart TO :"dwh_user";
ALTER DEFAULT PRIVILEGES IN SCHEMA raw, core, mart GRANT ALL ON TABLES TO :"dwh_user";
ALTER DEFAULT PRIVILEGES IN SCHEMA raw, core, mart GRANT ALL ON SEQUENCES TO :"dwh_user";
SQL

echo "[bootstrap] готово: БД=${DWH_DB}, схемы=raw,core,mart, роль=${DWH_USER}"
