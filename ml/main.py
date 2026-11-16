"""CLI ML-пайплайна: обучение CatBoost и batch-скоринг тарифов (этап 4).

Примеры::

    # полный цикл: обучение + оценка + скоринг (as_of из AS_OF окружения)
    python -m ml.main

    # явная дата снимка (используется DAG Airflow ml_train_score)
    python -m ml.main --as-of 2026-10-26

    # повторный скоринг без переобучения (загружает сохранённую модель)
    python -m ml.main --as-of 2026-10-26 --skip-train

    # подбор гиперпараметров (CLI переопределяет env): воспроизводимый прогон
    python -m ml.main --as-of 2026-10-26 --point-loss MAPE --depth 8 \\
        --l2-leaf-reg 5.0 --learning-rate 0.05

Повторный запуск идемпотентен: прогнозы пишутся UPSERT по ``price_id + model_name``.
"""
from __future__ import annotations

import argparse
import datetime as dt
from typing import Optional, Sequence

from common.config import get_settings
from common.logging import configure_logging, get_logger
from ml.pipeline import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml",
        description="Обучение и batch-скоринг модели прогнозирования тарифа (CatBoost).",
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help="Дата снимка (YYYY-MM-DD); по умолчанию берётся AS_OF из окружения.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Использовать уже обученную модель (только скоринг).",
    )
    parser.add_argument(
        "--point-loss",
        choices=["MAPE", "RMSE", "log_rmse"],
        default=None,
        help="Основной лосс точечной модели (переопределяет ML_POINT_LOSS).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Глубина деревьев CatBoost (переопределяет ML_DEPTH).",
    )
    parser.add_argument(
        "--l2-leaf-reg",
        type=float,
        default=None,
        help="L2-регуляризация листьев (переопределяет ML_L2_LEAF_REG).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Скорость обучения (переопределяет ML_LEARNING_RATE).",
    )
    return parser


def _parse_as_of(value: Optional[str]) -> str:
    if not value:
        return get_settings().as_of
    return dt.date.fromisoformat(value).isoformat()


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_logging()
    log = get_logger("ml")
    args = build_parser().parse_args(argv)

    try:
        as_of = _parse_as_of(args.as_of)
    except ValueError:
        log.error("ml.invalid_as_of", as_of=args.as_of, note="Ожидается YYYY-MM-DD.")
        return 2

    log.info(
        "ml.start",
        as_of=as_of,
        skip_train=args.skip_train,
        point_loss=args.point_loss,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        learning_rate=args.learning_rate,
    )
    stats = run(
        as_of,
        skip_train=args.skip_train,
        point_loss=args.point_loss,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        learning_rate=args.learning_rate,
    )
    log.info(
        "ml.done",
        as_of=stats.as_of,
        train_rows=stats.train_rows,
        val_rows=stats.val_rows,
        test_rows=stats.test_rows,
        future_rows=stats.future_rows,
        predictions_rows=stats.predictions_rows,
        metrics=stats.metrics,
        point_metrics=stats.point_metrics,
        picp=stats.picp,
        baselines=stats.baselines,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
