"""Grid-search тюнинг гиперпараметров точечной модели (этап 4, дополнение).

Перебирает сетку ``loss × depth × l2_leaf_reg × learning_rate``, для каждой ячейки
обучает точечную ``CatBoostRegressor`` с хронологическим ``val`` (ранняя остановка)
и оценивает метрики на ``test``. По итогу выводит отсортированную таблицу и
**рекомендованные параметры** (лучшая ячейка по выбранной метрике, по умолчанию
MAPE).

Скрипт read-only: данные читаются, модели обучаются в памяти, прогнозы в БД и
артефакты НЕ пишутся. Воспроизводимость обеспечивают фиксированный ``seed`` и
детерминированное хронологическое разбиение (см. ``ml/pipeline.py``).

Доступные лоссы точечной модели:
* ``RMSE`` — бейзлайн отчёта §1 (минимизирует абсолютную ошибку);
* ``MAPE`` — нативный лосс CatBoost (прямо оптимизирует MAPE; на логнормальных
  тарифах склонен к сильному систематическому занижению);
* ``log_rmse`` — RMSE по ``ln(y)`` с обратным ``exp`` и smearing-поправкой:
  оптимизирует относительную ошибку симметрично, без патологического смещения
  MAPE-лосса (рекомендуется для целевой метрики MAPE).

Примеры::

    python -m ml.tune --as-of 2026-10-26
    python -m ml.tune --as-of 2026-10-26 --metric mape_pct --top 10
    python -m ml.tune --as-of 2026-10-26 --depth 6,8 --l2-leaf-reg 3.0,5.0 \
        --learning-rate 0.03,0.1 --losses RMSE,log_rmse --out ml/artifacts/tune.csv
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from common.config import Settings, get_settings
from common.logging import configure_logging, get_logger
from etl.db import dwh_engine
from ml import features
from ml.data import load_ml_frame
from ml.pipeline import (
    EARLY_STOPPING_ROUNDS,
    VAL_FRACTION,
    _canonical_point_loss,
    _masks,
    evaluate,
    train_model,
)

log = get_logger("ml.tune")

# Метрики ``evaluate``: r2 выше лучше; остальные — ниже лучше.
METRIC_NAMES = ("mape_pct", "rmse", "mae", "r2")
_HIGHER_IS_BETTER = {"r2"}

# Сетка по умолчанию: компактная, чтобы прогон укладывался в несколько минут на
# ~19k обучающих строк. Расширяется флагами CLI (списки через запятую).
DEFAULT_LOSSES: Sequence[str] = ("RMSE", "log_rmse", "MAPE")
DEFAULT_DEPTHS: Sequence[int] = (4, 6, 8)
DEFAULT_L2: Sequence[float] = (1.0, 3.0, 10.0)
DEFAULT_LRS: Sequence[float] = (0.03, 0.1)

_RESULT_COLUMNS = (
    "loss", "depth", "l2_leaf_reg", "learning_rate",
    "r2", "mape_pct", "rmse", "mae", "bias_pct",
)


@dataclass
class TuneResult:
    """Сводка прогона: таблица результатов + рекомендованная ячейка."""
    as_of: str
    metric: str
    train_rows: int = 0
    val_rows: int = 0
    test_rows: int = 0
    rows: list = field(default_factory=list)
    best: dict = field(default_factory=dict)
    ranked: list = field(default_factory=list)


def _parse_ints(value: Optional[str], default: Sequence[int]) -> list[int]:
    if value is None:
        return list(default)
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _parse_floats(value: Optional[str], default: Sequence[float]) -> list[float]:
    if value is None:
        return list(default)
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _parse_losses(value: Optional[str], default: Sequence[str]) -> list[str]:
    if value is None:
        return list(default)
    allowed = {"RMSE", "MAPE", "log_rmse"}
    out = []
    bad = []
    for x in value.split(","):
        x = x.strip()
        if not x:
            continue
        canonical = _canonical_point_loss(x)
        if canonical not in allowed:
            bad.append(x)
            continue
        out.append(canonical)
    if bad:
        raise ValueError(f"Недопустимые лоссы {bad}; разрешены {sorted(allowed)}.")
    return out


def _fit_point(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    seed: int,
    *,
    loss: str,
    depth: int,
    l2_leaf_reg: float,
    learning_rate: float,
) -> np.ndarray:
    """Обучить точечную модель и вернуть прогноз на test (для ``log_rmse`` — в рублях)."""
    kwargs = dict(
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        learning_rate=learning_rate,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )

    if loss == "log_rmse":
        # RMSE по ln(y): симметричная оптимизация относительной ошибки + smearing.
        y_fit_log = np.log(y_fit)
        y_val_log = np.log(y_val)
        model = train_model(
            X_fit, y_fit_log, seed, loss="RMSE", eval_set=(X_val, y_val_log), **kwargs,
        )
        raw = np.asarray(model.predict(X_test), dtype="float64")
        resid = y_val_log.to_numpy(dtype="float64") - np.asarray(model.predict(X_val), dtype="float64")
        smear = float(np.mean(np.exp(resid)))  # поправка на смещение exp(log)
        return np.exp(raw) * smear

    model = train_model(
        X_fit, y_fit, seed, loss=loss, eval_set=(X_val, y_val), **kwargs,
    )
    return np.asarray(model.predict(X_test), dtype="float64")


def _print_summary(result: TuneResult, top: int) -> None:
    """Человекочитаемая таблица top-N и блок рекомендованных параметров."""
    ranked = result.ranked
    header = (
        f"{'loss':9} {'depth':>5} {'l2':>6} {'lr':>6} "
        f"{'R2':>8} {'MAPE%':>9} {'RMSE':>11} {'MAE':>11} {'bias%':>8}"
    )
    print("\n" + "=" * len(header))
    print(f"Результаты grid-search (metric={result.metric}, n={len(ranked)}):")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in ranked[:top]:
        print(
            f"{r['loss']:9} {r['depth']:>5} {r['l2_leaf_reg']:>6} {r['learning_rate']:>6} "
            f"{r['r2']:>8.4f} {r['mape_pct']:>9.4f} {r['rmse']:>11.2f} "
            f"{r['mae']:>11.2f} {r['bias_pct']:>8.4f}"
        )

    b = result.best
    print("-" * len(header))
    print("РЕКОМЕНДОВАННЫЕ ПАРАМЕТРЫ (лучшая ячейка по целевой метрике):")
    print(
        f"  --point-loss {b['loss']} --depth {b['depth']} "
        f"--l2-leaf-reg {b['l2_leaf_reg']} --learning-rate {b['learning_rate']}"
    )
    print(
        f"  R²={b['r2']:.6f}  MAPE={b['mape_pct']:.4f}%  "
        f"RMSE={b['rmse']:.2f} руб  MAE={b['mae']:.2f} руб  bias={b['bias_pct']:.4f}%"
    )
    print("=" * len(header) + "\n")


def run(
    as_of: str,
    *,
    settings: Optional[Settings] = None,
    metric: str = "mape_pct",
    losses: Sequence[str] = DEFAULT_LOSSES,
    depths: Sequence[int] = DEFAULT_DEPTHS,
    l2_leaf_regs: Sequence[float] = DEFAULT_L2,
    learning_rates: Sequence[float] = DEFAULT_LRS,
    seed: Optional[int] = None,
    top: int = 10,
    out: Optional[str] = None,
) -> TuneResult:
    """Выполнить grid-search и вернуть сводку с рекомендованной ячейкой."""
    if metric not in METRIC_NAMES:
        raise ValueError(f"Недопустимая метрика {metric}; разрешены {METRIC_NAMES}.")

    settings = settings or get_settings()
    seed = settings.ml_random_seed if seed is None else int(seed)
    as_of_date = dt.date.fromisoformat(as_of)
    dwh = dwh_engine(settings)

    wide = load_ml_frame(dwh, settings, as_of=as_of_date)
    train_mask, test_mask = _masks(wide, as_of_date)
    X_all = features.build_feature_matrix(wide)
    y_all = wide["price_rub"].astype("float64")

    # Границы отсечения выбросов — строго по train (отчёт §2.2).
    inlier_lo, inlier_hi = features.fit_inlier_bounds(y_all.loc[train_mask])
    train_mask = train_mask & (y_all >= inlier_lo) & (y_all <= inlier_hi)
    test_mask = test_mask & (y_all >= inlier_lo) & (y_all <= inlier_hi)

    X_train, y_train = X_all.loc[train_mask], y_all.loc[train_mask]
    X_test = X_all.loc[test_mask]
    y_test = y_all.loc[test_mask].to_numpy(dtype="float64")

    # Хронологический val (хвост train) под раннюю остановку — без shuffle.
    val_mask = features.chronological_val_mask(wide.loc[train_mask, "loading_date"], VAL_FRACTION)
    X_fit, y_fit = X_train.loc[~val_mask], y_train.loc[~val_mask]
    X_val, y_val = X_train.loc[val_mask], y_train.loc[val_mask]

    log.info(
        "ml.tune.start",
        as_of=as_of,
        metric=metric,
        seed=seed,
        train_rows=int(len(X_train)),
        val_rows=int(len(X_val)),
        test_rows=int(len(X_test)),
        grid_cells=len(losses) * len(depths) * len(l2_leaf_regs) * len(learning_rates),
    )

    rows: list[dict] = []
    for loss in losses:
        for depth in depths:
            for l2 in l2_leaf_regs:
                for lr in learning_rates:
                    try:
                        pred = _fit_point(
                            X_fit, y_fit, X_val, y_val, X_test, seed,
                            loss=loss, depth=depth, l2_leaf_reg=l2, learning_rate=lr,
                        )
                        m = evaluate(y_test, pred)
                    except Exception as exc:  # одна плохая ячейка не роняет прогон
                        log.warning(
                            "ml.tune.cell.error",
                            loss=loss, depth=depth, l2_leaf_reg=l2, learning_rate=lr,
                            error=str(exc),
                        )
                        continue
                    cell = {
                        "loss": loss,
                        "depth": depth,
                        "l2_leaf_reg": l2,
                        "learning_rate": lr,
                        **{k: float(v) for k, v in m.items()},
                    }
                    rows.append(cell)
                    log.info(
                        "ml.tune.cell",
                        loss=loss, depth=depth, l2_leaf_reg=l2, learning_rate=lr,
                        **{k: round(float(v), 6) for k, v in m.items()},
                    )

    if not rows:
        raise RuntimeError("Ни одна ячейка сетки не завершилась успешно.")

    # Ранжирование по целевой метрике (для r2 — по убыванию), tie-break — RMSE.
    if metric == "r2":
        ranked = sorted(rows, key=lambda r: (-r["r2"], r["rmse"]))
    else:
        ranked = sorted(rows, key=lambda r: (r[metric], r["rmse"]))

    result = TuneResult(
        as_of=as_of_date.isoformat(),
        metric=metric,
        train_rows=int(len(X_train)),
        val_rows=int(len(X_val)),
        test_rows=int(len(X_test)),
        rows=rows,
        best=ranked[0],
        ranked=ranked,
    )

    if out:
        pd.DataFrame(rows, columns=list(_RESULT_COLUMNS)).to_csv(out, index=False)
        log.info("ml.tune.saved", path=out, rows=len(rows))

    log.info(
        "ml.tune.best",
        metric=metric,
        **{k: (round(float(v), 6) if isinstance(v, float) else v) for k, v in result.best.items()},
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml.tune",
        description="Grid-search гиперпараметров точечной модели CatBoost.",
    )
    parser.add_argument("--as-of", default=None, help="Дата снимка (YYYY-MM-DD).")
    parser.add_argument(
        "--metric",
        choices=list(METRIC_NAMES),
        default="mape_pct",
        help="Целевая метрика ранжирования (по умолчанию mape_pct).",
    )
    parser.add_argument("--top", type=int, default=10, help="Сколько лучших ячеек показать.")
    parser.add_argument(
        "--losses",
        default=None,
        help="Лоссы через запятую: RMSE,log_rmse,MAPE (по умолчанию).",
    )
    parser.add_argument("--depth", default=None, help="Глубины через запятую (по умолчанию 4,6,8).")
    parser.add_argument("--l2-leaf-reg", default=None, help="L2-регуляризации через запятую (по умолчанию 1.0,3.0,10.0).")
    parser.add_argument("--learning-rate", default=None, help="Скорости обучения через запятую (по умолчанию 0.03,0.1).")
    parser.add_argument("--seed", type=int, default=None, help="Зерно (по умолчанию ML_RANDOM_SEED).")
    parser.add_argument("--out", default=None, help="Сохранить полную таблицу результатов в CSV.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    as_of = args.as_of or get_settings().as_of

    try:
        losses = _parse_losses(args.losses, DEFAULT_LOSSES)
        depths = _parse_ints(args.depth, DEFAULT_DEPTHS)
        l2s = _parse_floats(args.l2_leaf_reg, DEFAULT_L2)
        lrs = _parse_floats(args.learning_rate, DEFAULT_LRS)
        result = run(
            as_of,
            metric=args.metric,
            losses=losses,
            depths=depths,
            l2_leaf_regs=l2s,
            learning_rates=lrs,
            seed=args.seed,
            top=args.top,
            out=args.out,
        )
    except (ValueError, RuntimeError) as exc:
        log.error("ml.tune.error", error=str(exc))
        return 2

    _print_summary(result, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
