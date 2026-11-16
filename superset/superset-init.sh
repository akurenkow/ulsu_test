#!/usr/bin/env bash
# Идемпотентная инициализация Superset и запуск веб-сервера.
# Используется как command сервиса `superset` (см. docker-compose.yml, профиль `superset`).
set -euo pipefail

: "${SUPERSET_ADMIN_USER:=admin}"
: "${SUPERSET_ADMIN_PASSWORD:=admin}"
: "${SUPERSET_ADMIN_FIRSTNAME:=Admin}"
: "${SUPERSET_ADMIN_LASTNAME:=User}"
: "${SUPERSET_ADMIN_EMAIL:=admin@example.com}"

# Метаданные Superset — SQLite в named-volume /app/superset_home (по умолчанию образа).
superset db upgrade

# Повторный вызов при существующем пользователе завершится ошибкой — это ок.
superset fab create-admin \
  --username "${SUPERSET_ADMIN_USER}" \
  --firstname "${SUPERSET_ADMIN_FIRSTNAME}" \
  --lastname "${SUPERSET_ADMIN_LASTNAME}" \
  --email "${SUPERSET_ADMIN_EMAIL}" \
  --password "${SUPERSET_ADMIN_PASSWORD}" || true

superset init

# Демо-дашборд (superset/dashboards/demo.zip) импортируется на каждом старте
# (идемпотентно: сущности совпадают по uuid, повторный импорт не дублирует).
# В экспорте пароль БД замаскирован (`XXXXXXXXXX`), поэтому импорт делает
# `import_dashboards.py`: он подставляет реальный пароль из `SUPERSET_DWH_URI`
# и создаёт БД `dwh`, датасеты, чарты и дашборд.
if [ -f /opt/superset/dashboards/demo.zip ]; then
  python /opt/superset/import_dashboards.py || true
fi

# Подстраховка: URI DWH приводится к значению из окружения (если пользователь
# переопределил DWH_PASSWORD; при совпадении ничего не меняет).
superset set-database-uri --database_name dwh --uri "${SUPERSET_DWH_URI}" || true

exec superset run -h 0.0.0.0 -p 8088 --with-threads --reload --debugger
