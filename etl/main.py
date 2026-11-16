"""CLI ETL-пайплайна: извлечение -> DQ -> конвертация валют -> обогащение -> загрузка.

Примеры (этап 2)::

    # загрузить весь доступный период источника
    python -m etl.main --start-date 2023-06-01 --end-date 2026-10-26

    # backfill одного дня (используется Airflow DAG etl_dwh_ml)
    python -m etl.main --start-date 2026-10-26 --end-date 2026-10-26

    # повторный прогон за тот же период идемпотентен (UPSERT / DELETE+INSERT)
    python -m etl.main --start-date 2023-06-01 --end-date 2026-10-26
"""
from __future__ import annotations

import argparse
import datetime as dt
from typing import Optional, Sequence

from common.logging import configure_logging, get_logger
from etl.marts import rebuild_marts
from etl.pipeline import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="etl",
        description="Извлечение, валидация, очистка и идемпотентная загрузка цен в Core DWH.",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Нижняя граница окна (loading_date) включительно, формат YYYY-MM-DD "
        "(обязателен для ETL; не нужен при --rebuild-marts).",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Верхняя граница окна (loading_date) включительно, формат YYYY-MM-DD "
        "(обязателен для ETL; не нужен при --rebuild-marts).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Размер батча чтения источника (по умолчанию 10000).",
    )
    parser.add_argument(
        "--rebuild-marts",
        action="store_true",
        help="Пересчитать витрины mart (этап 3). Точка входа для Airflow DAG rebuild_marts.",
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help="Дата снимка (YYYY-MM-DD) для пересчёта витрин; по умолчанию AS_OF из окружения.",
    )
    return parser


def _parse_date(value: str, flag: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise SystemExit(f"Некорректная дата {flag}={value!r}: ожидается YYYY-MM-DD.") from exc


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_logging()
    log = get_logger("etl")
    args = build_parser().parse_args(argv)

    if args.rebuild_marts:
        stats = rebuild_marts(as_of=args.as_of)
        log.info(
            "etl.rebuild_marts.done",
            as_of=args.as_of or "(env AS_OF)",
            route_card_rows=stats.route_card_rows,
            fact_vs_corridor_rows=stats.fact_vs_corridor_rows,
            price_body_types_rows=stats.price_body_types_rows,
            price_dates_rows=stats.price_dates_rows,
        )
        return 0

    if not args.start_date or not args.end_date:
        log.error("etl.missing_dates", note="Для ETL укажите --start-date и --end-date.")
        return 2

    if args.batch_size <= 0:
        log.error("etl.invalid_batch_size", batch_size=args.batch_size)
        return 2

    start = _parse_date(args.start_date, "--start-date")
    end = _parse_date(args.end_date, "--end-date")
    if start > end:
        log.error("etl.invalid_window", start_date=args.start_date, end_date=args.end_date)
        return 2

    run(args.start_date, args.end_date, batch_size=args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
