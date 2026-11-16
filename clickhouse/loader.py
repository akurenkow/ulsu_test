"""Загрузка витрин PostgreSQL → ClickHouse (бонус ★, этап 3).

Переливает ``dwh.mart.mart_route_card`` (PostgreSQL) в ``dwh.mart_route_card``
(ClickHouse, ``ReplacingMergeTree``) батчами через ``clickhouse-connect`` — без
построчных вставок — и после загрузки выполняет ``OPTIMIZE TABLE ... FINAL`` для
дедупликации по ключу сортировки (grain витрины). В конце сверяет число строк и
контрольные суммы с PostgreSQL.

Запуск:
    # локально (ClickHouse на localhost:9000)
    python -m clickhouse.loader

    # внутри docker-compose (профили etl + clickhouse)
    docker compose --profile clickhouse up -d
    docker compose --profile etl run --rm etl python -m clickhouse.loader

Плановые прогнозы ``mart.ml_price_predictions`` добавляются на этапе 4.
"""
from __future__ import annotations

import argparse
import os
from typing import Optional, Sequence

import pandas as pd
from sqlalchemy import text

from common.config import get_settings
from common.logging import configure_logging, get_logger
from etl.db import df_to_tuples, dwh_engine
from etl.marts import MART_ROUTE_CARD_COLUMNS

log = get_logger("clickhouse")

CH_ROUTE_CARD_TABLE = "dwh.mart_route_card"
CH_ML_PREDICTION_TABLE = "dwh.ml_price_predictions"

# Порядок колонок mart.ml_price_predictions в DDL ClickHouse (dwh.ml_price_predictions).
ML_PREDICTION_COLUMNS = (
    "price_id", "model_name", "loading_date", "loading_city_gcid",
    "unloading_city_gcid", "body_group", "actual_price_rub",
    "predicted_price_rub", "predicted_price_rub_lo", "predicted_price_rub_hi",
    "abs_error_rub", "ape_pct", "scored_at",
)


def _ch_client():
    """Лениво импортировать clickhouse-connect (зависимость requirements-clickhouse.txt)."""
    try:
        import clickhouse_connect  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "clickhouse-connect не установлен; установите requirements-clickhouse.txt."
        ) from exc
    return clickhouse_connect


def _pg_route_card_frame():
    """Прочитать mart_route_card из PostgreSQL (колонки в порядке ClickHouse-DDL)."""
    cols = ", ".join(MART_ROUTE_CARD_COLUMNS)
    dwh = dwh_engine(get_settings())
    return pd.read_sql(
        text(f"SELECT {cols} FROM mart.mart_route_card ORDER BY 1, 2, 3, 6, 7"),
        dwh,
    )


def _checksums(df: pd.DataFrame) -> dict:
    """Контрольные суммы для сверки PostgreSQL ↔ ClickHouse."""
    if df.empty:
        return {"rows": 0, "sum_trips_count": 0, "sum_median_trip": 0.0, "sum_avg_trip": 0.0}
    return {
        "rows": int(len(df)),
        "sum_trips_count": int(df["trips_count"].sum()),
        "sum_median_trip": float(df["median_trip"].sum()),
        "sum_avg_trip": float(df["avg_trip"].sum()),
    }


def _pg_ml_predictions_frame() -> pd.DataFrame:
    """Прочитать ml_price_predictions из PostgreSQL (колонки в порядке ClickHouse-DDL)."""
    cols = ", ".join(ML_PREDICTION_COLUMNS)
    dwh = dwh_engine(get_settings())
    df = pd.read_sql(
        text(f"SELECT {cols} FROM mart.ml_price_predictions ORDER BY price_id, model_name"),
        dwh,
    )
    # ClickHouse DateTime — без tz; Postgres timestamptz отдаёт aware-значения.
    # Приводим к наивному UTC (время сессии СУБД — UTC в контейнере).
    df["scored_at"] = pd.to_datetime(df["scored_at"]).dt.tz_localize(None)
    return df


def _ml_checksums(df: pd.DataFrame) -> dict:
    """Контрольные суммы для сверки прогнозов PostgreSQL ↔ ClickHouse."""
    if df.empty:
        return {"rows": 0, "sum_predicted": 0.0, "sum_predicted_lo": 0.0, "sum_predicted_hi": 0.0, "future_rows": 0}
    return {
        "rows": int(len(df)),
        "sum_predicted": float(df["predicted_price_rub"].sum()),
        "sum_predicted_lo": float(df["predicted_price_rub_lo"].sum()),
        "sum_predicted_hi": float(df["predicted_price_rub_hi"].sum()),
        "future_rows": int(df["actual_price_rub"].isna().sum()),
    }


def load_marts(as_of: str, *, host: str, port: int, user: str, password: str, database: str) -> dict:
    """Перелить витрины mart из PostgreSQL в ClickHouse (идемпотентно).

    Переливает ``mart.mart_route_card`` и ``mart.ml_price_predictions`` (если
    наполнена) и возвращает сводку: число строк и контрольные суммы в PostgreSQL
    и ClickHouse (для сверки). Загрузка идёт векторизованным ``client.insert``;
    ``ReplacingMergeTree`` + ``OPTIMIZE ... FINAL`` дедуплицирует повторы по ключу
    сортировки (grain витрины / ``(price_id, model_name)``).
    """
    clickhouse_connect = _ch_client()
    client = clickhouse_connect.get_client(
        host=host, port=port, username=user, password=password, database=database
    )

    # --- mart_route_card -----------------------------------------------------
    df = _pg_route_card_frame()
    pg = _checksums(df)
    log.info("clickhouse.load_marts.pg_read", table=CH_ROUTE_CARD_TABLE, checksum=pg)

    if not df.empty:
        rows = df_to_tuples(df, list(MART_ROUTE_CARD_COLUMNS), date_columns=("month",))
        # Векторизованная батч-вставка (без построчных INSERT в цикле).
        client.insert("mart_route_card", rows, column_names=list(MART_ROUTE_CARD_COLUMNS))
        log.info("clickhouse.load_marts.inserted", table=CH_ROUTE_CARD_TABLE, rows=len(rows))

    # Дедупликация ReplacingMergeTree по ключу сортировки (grain витрины).
    client.command(f"OPTIMIZE TABLE {CH_ROUTE_CARD_TABLE} FINAL")

    row = client.query(
        f"SELECT count(), toUInt64(sum(trips_count)), "
        f"round(sum(median_trip), 2), round(sum(avg_trip), 2) "
        f"FROM {CH_ROUTE_CARD_TABLE} FINAL"
    ).first_row
    ch = {
        "rows": int(row[0]),
        "sum_trips_count": int(row[1]),
        "sum_median_trip": float(row[2]),
        "sum_avg_trip": float(row[3]),
    }
    log.info("clickhouse.load_marts.ch_read", table=CH_ROUTE_CARD_TABLE, checksum=ch)

    # --- ml_price_predictions ------------------------------------------------
    ml_df = _pg_ml_predictions_frame()
    ml_pg = _ml_checksums(ml_df)
    log.info("clickhouse.load_marts.pg_read", table=CH_ML_PREDICTION_TABLE, checksum=ml_pg)

    if not ml_df.empty:
        ml_rows = df_to_tuples(
            ml_df,
            list(ML_PREDICTION_COLUMNS),
            date_columns=("loading_date",),
            ts_columns=("scored_at",),
        )
        client.insert("ml_price_predictions", ml_rows, column_names=list(ML_PREDICTION_COLUMNS))
        log.info("clickhouse.load_marts.inserted", table=CH_ML_PREDICTION_TABLE, rows=len(ml_rows))

    client.command(f"OPTIMIZE TABLE {CH_ML_PREDICTION_TABLE} FINAL")

    ml_row = client.query(
        f"SELECT count(), round(sum(predicted_price_rub), 2), "
        f"round(sum(predicted_price_rub_lo), 2), round(sum(predicted_price_rub_hi), 2), "
        f"toUInt64(countIf(isNull(actual_price_rub))) "
        f"FROM {CH_ML_PREDICTION_TABLE} FINAL"
    ).first_row
    ml_ch = {
        "rows": int(ml_row[0]),
        "sum_predicted": float(ml_row[1]),
        "sum_predicted_lo": float(ml_row[2]),
        "sum_predicted_hi": float(ml_row[3]),
        "future_rows": int(ml_row[4]),
    }
    log.info("clickhouse.load_marts.ch_read", table=CH_ML_PREDICTION_TABLE, checksum=ml_ch)

    summary = {
        "as_of": as_of,
        "mart_route_card_rows": pg["rows"],
        "ml_price_predictions_rows": ml_pg["rows"],
        "postgres": pg,
        "clickhouse": ch,
        "match": pg == ch,
        "ml_postgres": ml_pg,
        "ml_clickhouse": ml_ch,
        "ml_match": ml_pg == ml_ch,
    }
    log.info("clickhouse.load_marts.done", summary=summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clickhouse-loader",
        description="Переливка аналитических витрин PostgreSQL → ClickHouse (бонус ★).",
    )
    parser.add_argument("--as-of", default=None, help="Дата снимка (YYYY-MM-DD), по умолчанию AS_OF из env.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    as_of = args.as_of or os.getenv("AS_OF", "2026-10-26")
    summary = load_marts(
        as_of,
        host=os.getenv("CLICKHOUSE_HOST", "localhost"),
        port=int(os.getenv("CLICKHOUSE_NATIVE_PORT", "9000")),
        user=os.getenv("CLICKHOUSE_USER", "default"),
        password=os.getenv("CLICKHOUSE_PASSWORD", "clickhouse"),
        database=os.getenv("CLICKHOUSE_DB", "dwh"),
    )
    if not summary["match"] or not summary["ml_match"]:
        log.warning(
            "clickhouse.load_marts.mismatch",
            note="Контрольные суммы PostgreSQL и ClickHouse не совпали — проверьте данные.",
            postgres=summary["postgres"],
            clickhouse=summary["clickhouse"],
            ml_postgres=summary["ml_postgres"],
            ml_clickhouse=summary["ml_clickhouse"],
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
