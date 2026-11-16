"""DAG ``etl_dwh_ml`` — оркестрация обязательного пайплайна (бонус ★: Apache Airflow).

Полный цикл: ``init_source → etl → rebuild_marts → ml_train_score``.

Принципы:
* каждая стадия вызывает ту же команду, что и ручной запуск обязательной части
  (``python -m etl.main`` / ``python -m ml.main``), поэтому результаты совпадают;
* окно дат берётся из ``logical_date`` (``{{ ds }}``) либо переопределяется через
  ``dag_run.conf`` при ручном триггере — это даёт backfill за произвольный период;
* идемпотентность обеспечивается самими стадиями (UPSERT / DELETE+INSERT в
  обязательной части), повторный прогон за тот же ``ds`` не дублирует строки.

Примеры запуска::

    # обычный запуск (одни сутки logical_date)
    airflow dags trigger etl_dwh_ml

    # backfill за весь доступный период
    airflow dags trigger etl_dwh_ml \
      --conf '{"start_date": "2026-01-01", "end_date": "2026-10-26"}'
"""

from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

DAG_ID = "etl_dwh_ml"

# Дефолты Airflow Variables/``dag_run.conf`` (значения в один символ-источник —
# в .env.example / docker-compose).
DEFAULT_AS_OF = "2026-10-26"

default_args = {
    "owner": "data-engineering",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": 60,  # секунды
}

with DAG(
    dag_id=DAG_ID,
    description="init source → ETL/DQ → mart → ML train+score (★ Airflow)",
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["etl", "dwh", "ml", "bonus", "airflow"],
) as dag:

    # 1. Идемпотентная инициализация источника и схем dwh.
    #    Переиспользует логику docker/postgres/init (00_bootstrap.sh + 10_init_source.sh).
    init_source = BashOperator(
        task_id="init_source",
        bash_command=(
            "set -euo pipefail\n"
            "export PGHOST=\"${POSTGRES_HOST:-postgres}\"\n"
            "export PGUSER=\"${POSTGRES_USER:-postgres}\"\n"
            "export PGPASSWORD=\"${POSTGRES_PASSWORD:-postgres}\"\n"
            "bash /opt/ulog-init/00_bootstrap.sh\n"
            "bash /opt/ulog-init/10_init_source.sh\n"
        ),
    )

    # 2. ETL & DQ за окно дат (loading_date). По умолчанию — один логический день.
    etl = BashOperator(
        task_id="etl",
        bash_command=(
            'python -m etl.main '
            '--start-date "{{ dag_run.conf.get("start_date", ds) }}" '
            '--end-date "{{ dag_run.conf.get("end_date", ds) }}"'
        ),
    )

    # 3. Пересчёт витрин mart с параметром as_of.
    rebuild_marts = BashOperator(
        task_id="rebuild_marts",
        bash_command=(
            'python -m etl.main --rebuild-marts '
            '--as-of "{{ dag_run.conf.get("as_of", '
            'var.value.get("AS_OF", "' + DEFAULT_AS_OF + '")) }}"'
        ),
    )

    # 4. Обучение модели и batch-скоринг (прогнозы -> mart.ml_price_predictions).
    ml_train_score = BashOperator(
        task_id="ml_train_score",
        bash_command=(
            'python -m ml.main '
            '--as-of "{{ dag_run.conf.get("as_of", '
            'var.value.get("AS_OF", "' + DEFAULT_AS_OF + '")) }}"'
        ),
    )

    init_source >> etl >> rebuild_marts >> ml_train_score
