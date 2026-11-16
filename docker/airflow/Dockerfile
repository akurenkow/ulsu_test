# Apache Airflow image extended with the project's Python dependencies.
# Бонус ★: оркестрация обязательного пайплайна (docker compose --profile airflow).
#
# Стадии DAG запускают `python -m etl.main` / `python -m ml.main` прямо в контейнере,
# поэтому здесь устанавливаются те же зависимости, что и в Dockerfile.etl/ml.

FROM apache/airflow:2.10.2-python3.11

USER root
# psql нужен задаче init_source, которая переиспользует docker/postgres/init/*.sh.
RUN apt-get update \
    && apt-get install -y --no-install-recommends postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Зависимости проекта (pandas/SQLAlchemy/psycopg + scikit-learn/CatBoost).
COPY requirements.txt /tmp/ulog/requirements.txt
COPY requirements-ml.txt /tmp/ulog/requirements-ml.txt

USER airflow
# Устанавливаем против официального constraint-файла Airflow 2.10.2,
# чтобы не сломать зафиксированные транзитивные зависимости Airflow.
RUN pip install --no-cache-dir --user \
      -r /tmp/ulog/requirements-ml.txt \
      --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.2/constraints-3.11.txt"

# Код проекта монтируется в рантайме (docker-compose.yml, секция x-airflow-common):
#   ./common -> /opt/airflow/common
#   ./etl    -> /opt/airflow/etl
#   ./ml     -> /opt/airflow/ml
#   ./sql    -> /opt/sql
#   ./docker/postgres/init -> /opt/ulog-init
WORKDIR /opt/airflow
