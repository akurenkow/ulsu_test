"""Оркестрация ML-пайплайна: обучение → оценка → batch-скоринг (этап 4).

Полный цикл ``python -m ml.main --as-of 2026-10-26``:

1. применяет DDL ``mart.ml_price_predictions`` идемпотентно;
2. читает очищенный слой ``core`` двумя запросами — факты до ``as_of``
   (``load_ml_frame``) и плановые заявки после ``as_of`` (``load_future_frame``);
3. строит признаки без утечек (``ml/features.py``) и выполняет **строго
   хронологическое** разбиение ``train``/``test`` по ``loading_date`` относительно
   ``as_of`` (системная дата запуска не используется);
4. отсекает выбросы таргета по IQR логарифма — границы считаются **только по
   ``train``** (отчёт §2.2); выделяет хронологический ``val`` (хвост train) под
   раннюю остановку;
5. обучает **точечные модели** всех поддерживаемых лоссов (``MAPE``/``RMSE``/
   ``log_rmse``) для честного сравнения, основную — через ``ML_POINT_LOSS``
   (по умолчанию ``RMSE``; для целевой метрики MAPE рекомендуется ``log_rmse``)
   и два квантильных ``CatBoostRegressor`` (q25/q75) для честного интервала
   прогноза; считает
   метрики ``R²``/``MAPE``/``RMSE``/``MAE``, покрытие интервала (PICP) и метрики
   в разрезе направлений (``origin_region → destination_region``), а также
   референсные бейзлайны ``Ridge``/``DecisionTree`` (отчёт §5);
6. скорирует ``test`` и плановые будущие заявки и идемпотентно пишет прогнозы в
   ``mart.ml_price_predictions`` (UPSERT по ``price_id + model_name``) — точку и
   интервал lo/hi. Для будущих заявок факт/ошибки — ``NULL``.
"""
from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

from common.config import Settings, get_settings, resolve_dwh_sql_dir
from common.logging import get_logger
from etl.db import apply_ddl, df_to_tuples, dwh_engine, executemany
from ml import features
from ml.data import load_future_frame, load_ml_frame

log = get_logger("ml.pipeline")

MODEL_NAME = "catboost"
DEFAULT_MODEL_FILE = "catboost.cbm"

# Точечная модель по умолчанию минимизирует RMSE — бейзлайн отчёта §1. На логнормальных
# тарифах нативный MAPE-лосс склонен к сильному занижению (см. README), поэтому
# по умолчанию используется RMSE; для оптимизации относительной ошибки (целевая
# метрика MAPE) рекомендуется ``log_rmse``. Выбор — через ``ML_POINT_LOSS``/``--point-loss``.
DEFAULT_POINT_LOSS = "RMSE"
# Все поддерживаемые точечные лоссы обучаются и оцениваются на test в каждом
# прогоне, чтобы лог всегда содержал честное сравнение вариантов независимо от
# выбранного основного лосса.
POINT_LOSS_VARIANTS: Sequence[str] = ("MAPE", "RMSE", "log_rmse")
INTERVAL_ALPHAS: Sequence[float] = (0.25, 0.75)

# Хвост train-окна под раннюю остановку (без перемешивания) и её терпение.
VAL_FRACTION = 0.2
EARLY_STOPPING_ROUNDS = 50

# Порядок колонок mart.ml_price_predictions (совпадает с DDL и ClickHouse-DDL).
PREDICTION_COLUMNS: Sequence[str] = (
    "price_id",
    "loading_date",
    "loading_city_gcid",
    "unloading_city_gcid",
    "body_group",
    "actual_price_rub",
    "predicted_price_rub",
    "predicted_price_rub_lo",
    "predicted_price_rub_hi",
    "abs_error_rub",
    "ape_pct",
    "model_name",
    "scored_at",
)

_PREDICTION_UPSERT_SQL = """
INSERT INTO mart.ml_price_predictions (
    price_id, loading_date, loading_city_gcid, unloading_city_gcid, body_group,
    actual_price_rub, predicted_price_rub, predicted_price_rub_lo,
    predicted_price_rub_hi, abs_error_rub, ape_pct,
    model_name, scored_at
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (price_id, model_name) DO UPDATE SET
    loading_date        = EXCLUDED.loading_date,
    loading_city_gcid   = EXCLUDED.loading_city_gcid,
    unloading_city_gcid = EXCLUDED.unloading_city_gcid,
    body_group          = EXCLUDED.body_group,
    actual_price_rub    = EXCLUDED.actual_price_rub,
    predicted_price_rub = EXCLUDED.predicted_price_rub,
    predicted_price_rub_lo = EXCLUDED.predicted_price_rub_lo,
    predicted_price_rub_hi = EXCLUDED.predicted_price_rub_hi,
    abs_error_rub       = EXCLUDED.abs_error_rub,
    ape_pct             = EXCLUDED.ape_pct,
    scored_at           = EXCLUDED.scored_at
"""


@dataclass
class PipelineStats:
    as_of: str
    rows_read: int
    train_rows: int
    val_rows: int = 0
    test_rows: int = 0
    future_rows: int = 0
    predictions_rows: int = 0
    metrics: dict = field(default_factory=dict)
    point_metrics: dict = field(default_factory=dict)  # {loss: metrics} — сравнение MAPE/RMSE
    baselines: dict = field(default_factory=dict)
    picp: Optional[float] = None
    inlier_lo: float = 0.0
    inlier_hi: float = 0.0
    direction_metrics: list = field(default_factory=list)


def _quantile_key(alpha: float) -> str:
    """Имя ключа квантиля: ``0.25 -> q25``."""
    return f"q{int(round(alpha * 100))}"


def _canonical_point_loss(value: str) -> str:
    """Привести имя точечного лосса к каноничному виду (регистронезависимо).

    Каноничные имена — ``MAPE``/``RMSE`` (верхний регистр) и ``log_rmse`` (нижний).
    Наивный ``.upper()`` превращает ``log_rmse`` в ``LOG_RMSE`` и ломает сравнение,
    поэтому регистр нормализуется сравнением по ``.lower()``. Неизвестное значение
    возвращается как есть (обработка/фолбэк — на стороне вызывающего).
    """
    v = str(value).strip().lower()
    for canonical in POINT_LOSS_VARIANTS:
        if canonical.lower() == v:
            return canonical
    return str(value).strip()


def _point_variants(primary_loss: str) -> list[tuple[str, str]]:
    """Список ``(kind, loss)`` всех точечных вариантов: основной — первым.

    Основной вариант (выбранный через ``ML_POINT_LOSS``/``--point-loss``) идёт в
    артефакт ``catboost_point.cbm`` и пишется в ``mart.ml_price_predictions``;
    остальные — в ``catboost_point_<loss>.cbm`` и используются только для сравнения
    метрик. Все варианты оцениваются на test, чтобы лог всегда содержал честное
    сравнение независимо от выбранного основного лосса.
    """
    primary = _canonical_point_loss(primary_loss)
    primary = primary if primary in POINT_LOSS_VARIANTS else DEFAULT_POINT_LOSS
    ordered = [primary] + [l for l in POINT_LOSS_VARIANTS if l != primary]
    return [(_point_kind(loss, primary), loss) for loss in ordered]


def _point_kind(loss: str, primary_loss: str) -> str:
    """Имя артефакта для точечного варианта: основной — ``point``, альтернативный — ``point_<loss>``."""
    return "point" if loss == primary_loss else f"point_{loss.lower()}"


def _point_targets(loss: str, y_fit: pd.Series, y_val: pd.Series) -> tuple[str, pd.Series, pd.Series]:
    """CatBoost-лосс и целевые векторы для точечного варианта.

    ``log_rmse`` обучается на ``ln(y)`` с RMSE-лоссом — это симметричная оптимизация
    относительной ошибки (≈ MAPE) без патологического смещения нативного MAPE-лосса.
    Остальные лоссы обучаются на исходных рублях без преобразования.
    """
    if loss == "log_rmse":
        return "RMSE", np.log(y_fit), np.log(y_val)
    return loss, y_fit, y_val


def _inverse_point_transform(pred, loss: str, *, smear: float = 1.0) -> np.ndarray:
    """Обратное преобразование прогноза точечной модели (``log_rmse`` → ``exp(pred)·smear``)."""
    pred = np.asarray(pred, dtype="float64")
    if loss == "log_rmse":
        return np.exp(pred) * smear
    return pred


def _smear_factor(model, X_val: pd.DataFrame, y_val: pd.Series, loss: str) -> float:
    """Smearing-поправка ``exp(log)`` для ``log_rmse``; для остальных — ``1.0``.

    Считается по хронологическому ``val`` детерминированно (та же выборка, что и
    при обучении), поэтому воспроизводима и не требует сохранения в артефакте.
    """
    if loss != "log_rmse":
        return 1.0
    resid = np.log(np.asarray(y_val, dtype="float64")) - np.asarray(model.predict(X_val), dtype="float64")
    return float(np.exp(np.mean(resid)))


def resolve_model_path(kind: str = "point") -> Path:
    """Путь к артефакту модели (``ML_MODEL_PATH`` или репо по умолчанию).

    ``kind`` ∈ {"point", "q25", "q75"} задаёт суффикс файла
    (``catboost_point.cbm``, ``catboost_q25.cbm`` и т. п.).
    """
    env = os.getenv("ML_MODEL_PATH")
    if env:
        path = Path(env)
        return path.with_name(f"{path.stem}_{kind}{path.suffix}")
    repo_root = Path(__file__).resolve().parents[1]
    stem, suffix = DEFAULT_MODEL_FILE.rsplit(".", 1)
    return repo_root / "ml" / "artifacts" / f"{stem}_{kind}.{suffix}"


def _apply_ml_ddl(dwh: Engine) -> None:
    ddl_dir = resolve_dwh_sql_dir()
    path = ddl_dir / "01_mart_schema.sql"
    if not path.exists():
        raise FileNotFoundError(f"Не найден DDL mart-слоя: {path}")
    apply_ddl(dwh, path.read_text(encoding="utf-8"))
    log.info("ml.ddl.applied", path=str(path))


def _masks(wide: pd.DataFrame, as_of: dt.date) -> tuple[pd.Series, pd.Series]:
    """Маски ``train``/``test`` строго по ``loading_date`` (без перемешивания).

    Возвращает ``(train_mask, test_mask)`` по фактам до ``as_of`` (будущие заявки
    читаются отдельно ``load_future_frame`` и масками не покрываются).
    ``outlier_mark`` используется как входной pre-filter источника; дополнительные
    границы IQR считаются по ``train`` в ``run``.
    """
    train_start, test_start, test_end = features.time_split_bounds(as_of)
    ts_train_start = pd.Timestamp(train_start)
    ts_test_start = pd.Timestamp(test_start)
    ts_test_end = pd.Timestamp(test_end)

    train_mask = (
        (wide["loading_date"] >= ts_train_start)
        & (wide["loading_date"] < ts_test_start)
        & (~wide["outlier_mark"])
    )
    test_mask = (
        (wide["loading_date"] >= ts_test_start)
        & (wide["loading_date"] <= ts_test_end)
        & (~wide["outlier_mark"])
    )
    return train_mask, test_mask


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
    *,
    loss: str,
    eval_set: Optional[tuple] = None,
    early_stopping_rounds: Optional[int] = None,
    depth: Optional[int] = None,
    l2_leaf_reg: Optional[float] = None,
    learning_rate: Optional[float] = None,
):
    """Обучить ``CatBoostRegressor`` детерминированно (loss + ранняя остановка).

    ``depth``/``l2_leaf_reg``/``learning_rate`` — настраиваемые гиперпараметры;
    ``None`` означает «использовать дефолт CatBoost» (6 / 3.0 / 0.03). Значения
    задаются через env ``ML_DEPTH``/``ML_L2_LEAF_REG``/``ML_LEARNING_RATE`` или
    флаги CLI ``--depth``/``--l2-leaf-reg``/``--learning-rate``.
    """
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "catboost не установлен; установите requirements-ml.txt."
        ) from exc

    model = CatBoostRegressor(
        loss_function=loss,
        iterations=1000,
        random_seed=int(seed),
        cat_features=list(features.CATEGORICAL_FEATURES),
        verbose=False,
        allow_writing_files=False,
        early_stopping_rounds=early_stopping_rounds,
        use_best_model=True,  # вернуть лучшую итерацию по val (не последнюю)
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        learning_rate=learning_rate,
    )
    model.fit(X_train, y_train, eval_set=eval_set)
    return model


def save_model(model, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))


def load_model(path: Path):
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "catboost не установлен; установите requirements-ml.txt."
        ) from exc
    model = CatBoostRegressor()
    model.load_model(str(path))
    return model


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Метрики качества на отложенной выборке: R², MAPE (%), RMSE (руб.), MAE (руб.)."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    # bias_pct — знаковое смещение (положительное = систематическое завышение),
    # помогает диагностировать асимметрию MAPE: она штрафует занижение сильнее.
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mape_pct": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias_pct": float(np.mean((y_pred - y_true) / y_true) * 100.0),
    }


def evaluate_per_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Метрики по направлению ``origin_region → destination_region``.

    Отчёт §1: на уровне одного направления основная метрика — ``MAPE``/``MAE``
    (дисперсия таргета мала, поэтому ``R²`` там закономерно падает). Возвращает
    DataFrame, отсортированный по числу сделок.
    """
    from sklearn.metrics import r2_score

    rows = []
    grouped = df.groupby(["origin_region", "destination_region"], sort=False)
    for (origin, destination), g in grouped:
        yt = g["y_true"].to_numpy(dtype="float64")
        yp = g["y_pred"].to_numpy(dtype="float64")
        rows.append(
            {
                "origin_region": str(origin),
                "destination_region": str(destination),
                "n": int(len(yt)),
                "mape_pct": float(np.mean(np.abs((yt - yp) / yt)) * 100.0),
                "mae": float(np.mean(np.abs(yt - yp))),
                "r2": float(r2_score(yt, yp)) if len(yt) > 1 else float("nan"),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["origin_region", "destination_region", "n", "mape_pct", "mae", "r2"]
        )
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def prediction_interval_coverage(y_true, lo, hi) -> float:
    """Покрытие интервала (PICP): доля фактов внутри ``[lo, hi]``."""
    yt = np.asarray(y_true, dtype="float64")
    return float(np.mean((yt >= np.asarray(lo, dtype="float64")) & (yt <= np.asarray(hi, dtype="float64"))))


def _fit_baselines(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test, seed: int) -> dict:
    """Референсные бейзлайны отчёта §5: ``Ridge`` и ``DecisionTree``.

    Обе модели обучаются на одной и той же one-hot/масштабированной матрице
    (``SimpleImputer`` + ``StandardScaler`` для чисел, ``OneHotEncoder`` для
    категорий) — это линейный и древовидный ориентиры рядом с CatBoost.
    """
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.tree import DecisionTreeRegressor
    except ImportError:  # pragma: no cover
        log.warning("ml.baselines.skipped", reason="scikit-learn не установлен")
        return {}

    num_cols = list(features.NUMERIC_FEATURES)
    cat_cols = list(features.CATEGORICAL_FEATURES)

    def _as_object(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in cat_cols:
            out[c] = out[c].astype("object")
        return out

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
                num_cols,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    results = {}
    for name, est in (
        ("ridge", Ridge(alpha=1.0)),
        ("decision_tree", DecisionTreeRegressor(max_depth=10, random_state=int(seed))),
    ):
        pipe = Pipeline([("pre", pre), ("model", est)])
        pipe.fit(_as_object(X_train), y_train)
        results[name] = evaluate(y_test, np.asarray(pipe.predict(_as_object(X_test)), dtype="float64"))
    return results


def _feature_importance(model) -> dict:
    """Имена признаков → важность (CatBoost ``get_feature_importance``)."""
    names = getattr(model, "feature_names_", None) or list(features.FEATURE_COLUMNS)
    imp = model.get_feature_importance()
    return {str(n): float(v) for n, v in zip(names, imp)}


def _honest_interval(lo: np.ndarray, med: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Гарантировать ``lo <= med <= hi``, **сохраняя** точечный прогноз ``med``.

    Прежняя реализация сортировала тройку ``(lo, med, hi)`` и подменяла точечный
    прогноз медианой трёх значений. Это некорректно, когда точечная модель
    минимизирует MAPE (её прогноз смещён относительно квантилей q25/q75 и может
    выходить за их границы): ``predicted_price_rub`` обязана быть прогнозом именно
    точечной модели, а не перестановкой квантилей. Здесь границы интервала
    расширяются до точечного прогноза: ``lo = min(lo, med)``, ``hi = max(hi, med)``,
    поэтому инвариант ``lo <= med <= hi`` выполняется всегда, а покрытие интервала
    (PICP) при расширении не уменьшается.
    """
    lo = np.asarray(lo, dtype="float64")
    med = np.asarray(med, dtype="float64")
    hi = np.asarray(hi, dtype="float64")
    lo_out = np.minimum(np.minimum(lo, med), hi)
    hi_out = np.maximum(np.maximum(lo, med), hi)
    return lo_out, med, hi_out


def _prediction_block(
    keys: pd.DataFrame,
    body_group: pd.Series,
    actual,
    lo: np.ndarray,
    med: np.ndarray,
    hi: np.ndarray,
    scored_at: dt.datetime,
) -> pd.DataFrame:
    """Собрать блок строк ``mart.ml_price_predictions``.

    ``actual`` — факт (массив) или ``None`` (будущие заявки → NULL в ошибках).
    """
    n = len(keys)
    lo = np.asarray(lo, dtype="float64")
    med = np.asarray(med, dtype="float64")
    hi = np.asarray(hi, dtype="float64")
    if actual is None:
        actual_arr = np.full(n, np.nan)
        abs_err = np.full(n, np.nan)
        ape = np.full(n, np.nan)
    else:
        actual_arr = np.asarray(actual, dtype="float64")
        abs_err = np.abs(actual_arr - med)
        ape = np.abs((actual_arr - med) / actual_arr) * 100.0
    return pd.DataFrame(
        {
            "price_id": keys["price_id"].to_numpy(),
            "loading_date": keys["loading_date"].to_numpy(),
            "loading_city_gcid": keys["loading_city_gcid"].to_numpy(),
            "unloading_city_gcid": keys["unloading_city_gcid"].to_numpy(),
            "body_group": body_group.to_numpy(),
            "actual_price_rub": actual_arr,
            "predicted_price_rub": med,
            "predicted_price_rub_lo": lo,
            "predicted_price_rub_hi": hi,
            "abs_error_rub": abs_err,
            "ape_pct": ape,
            "model_name": MODEL_NAME,
            "scored_at": scored_at,
        }
    )


def _predictions_frame(
    wide: pd.DataFrame,
    body_group_test: pd.Series,
    test_mask: pd.Series,
    y_test: np.ndarray,
    test_lo: np.ndarray,
    test_med: np.ndarray,
    test_hi: np.ndarray,
    future_wide: pd.DataFrame,
    body_group_future: pd.Series,
    future_lo: np.ndarray,
    future_med: np.ndarray,
    future_hi: np.ndarray,
    scored_at: dt.datetime,
) -> pd.DataFrame:
    """Собрать все строки ``mart.ml_price_predictions`` (test: факт; future: NULL)."""
    key_cols = ["price_id", "loading_date", "loading_city_gcid", "unloading_city_gcid"]
    blocks = [
        _prediction_block(
            wide.loc[test_mask, key_cols],
            body_group_test,
            y_test,
            test_lo,
            test_med,
            test_hi,
            scored_at,
        )
    ]
    if future_wide is not None and len(future_wide):
        blocks.append(
            _prediction_block(
                future_wide[key_cols],
                body_group_future,
                None,
                future_lo,
                future_med,
                future_hi,
                scored_at,
            )
        )
    return pd.concat(blocks, ignore_index=True)[list(PREDICTION_COLUMNS)]


def _upsert_predictions(dwh: Engine, pred_df: pd.DataFrame) -> int:
    rows = df_to_tuples(
        pred_df,
        PREDICTION_COLUMNS,
        date_columns=("loading_date",),
        ts_columns=("scored_at",),
    )
    executemany(dwh, _PREDICTION_UPSERT_SQL, rows)
    return len(rows)


def run(
    as_of: str,
    *,
    skip_train: bool = False,
    settings: Optional[Settings] = None,
    point_loss: Optional[str] = None,
    depth: Optional[int] = None,
    l2_leaf_reg: Optional[float] = None,
    learning_rate: Optional[float] = None,
) -> PipelineStats:
    """Обучение и batch-скоринг; возвращает сводку и метрики.

    Гиперпараметры передаются из CLI (переопределяют env) или читаются из
    ``Settings``; ``None`` означает «дефолт CatBoost». Все точечные лоссы
    (MAPE/RMSE/log_rmse) обучаются и оцениваются на test в каждом прогоне — лог
    всегда содержит честное сравнение вариантов.
    """
    settings = settings or get_settings()
    as_of_date = dt.date.fromisoformat(as_of)
    dwh = dwh_engine(settings)

    # Разрешение гиперпараметров: CLI > env (Settings) > дефолт CatBoost (None).
    depth = settings.ml_depth if depth is None else depth
    l2_leaf_reg = settings.ml_l2_leaf_reg if l2_leaf_reg is None else l2_leaf_reg
    learning_rate = settings.ml_learning_rate if learning_rate is None else learning_rate
    requested_loss = _canonical_point_loss(point_loss or settings.ml_point_loss or DEFAULT_POINT_LOSS)
    primary_loss = requested_loss if requested_loss in POINT_LOSS_VARIANTS else DEFAULT_POINT_LOSS
    if primary_loss != requested_loss:
        log.warning(
            "ml.point_loss.invalid",
            value=requested_loss,
            fallback=primary_loss,
            note=f"Допустимы {list(POINT_LOSS_VARIANTS)}.",
        )

    _apply_ml_ddl(dwh)

    # Факты (train/test) и плановые будущие заявки (скоринг) — отдельными запросами.
    wide = load_ml_frame(dwh, settings, as_of=as_of_date)
    future_wide = load_future_frame(dwh, settings, as_of=as_of_date)
    log.info("ml.data.read", rows_read=int(len(wide)), future_rows_read=int(len(future_wide)))

    train_mask, test_mask = _masks(wide, as_of_date)
    X_all = features.build_feature_matrix(wide)
    y_all = wide["price_rub"].astype("float64")

    # Отсечение выбросов: границы IQR по ln(y) считаются строго по train (отчёт §2.2).
    inlier_lo, inlier_hi = features.fit_inlier_bounds(y_all.loc[train_mask])
    train_mask = train_mask & (y_all >= inlier_lo) & (y_all <= inlier_hi)
    test_mask = test_mask & (y_all >= inlier_lo) & (y_all <= inlier_hi)

    X_train = X_all.loc[train_mask]
    y_train = y_all.loc[train_mask]
    X_test = X_all.loc[test_mask]
    y_test = y_all.loc[test_mask].to_numpy(dtype="float64")
    body_group_test = wide.loc[test_mask, "body_type"].map(features.body_group).astype("string")

    # Хронологический val (хвост train) под раннюю остановку — без shuffle.
    val_mask = features.chronological_val_mask(wide.loc[train_mask, "loading_date"], VAL_FRACTION)
    X_fit, y_fit = X_train.loc[~val_mask], y_train.loc[~val_mask]
    X_val, y_val = X_train.loc[val_mask], y_train.loc[val_mask]

    # Будущие заявки: признаки и группа кузова (факт для них неизвестен).
    if len(future_wide):
        X_future = features.build_feature_matrix(future_wide)
        body_group_future = future_wide["body_type"].map(features.body_group).astype("string")
    else:
        X_future, body_group_future = None, None

    log.info(
        "ml.split",
        as_of=as_of_date.isoformat(),
        train_rows=int(len(X_train)),
        val_rows=int(len(X_val)),
        test_rows=int(len(X_test)),
        future_rows=int(len(future_wide)),
        inlier_lo=round(inlier_lo, 2) if np.isfinite(inlier_lo) else None,
        inlier_hi=round(inlier_hi, 2) if np.isfinite(inlier_hi) else None,
    )

    # Все точечные варианты (основной первым) + квантили интервала q25/q75.
    point_specs = _point_variants(primary_loss)
    specs = point_specs + [
        (_quantile_key(a), f"Quantile:alpha={a}") for a in INTERVAL_ALPHAS
    ]

    models: dict[str, object] = {}
    feature_importance: dict = {}
    for kind, loss in specs:
        model_path = resolve_model_path(kind)
        if skip_train:
            log.info("ml.model.load", kind=kind, path=str(model_path))
            model = load_model(model_path)
        else:
            train_loss, y_fit_t, y_val_t = _point_targets(loss, y_fit, y_val)
            log.info(
                "ml.model.train.start",
                model=MODEL_NAME,
                kind=kind,
                loss=loss,
                train_loss=train_loss,
                seed=settings.ml_random_seed,
                depth=depth,
                l2_leaf_reg=l2_leaf_reg,
                learning_rate=learning_rate,
                train_rows=int(len(X_fit)),
                val_rows=int(len(X_val)),
                features=len(features.FEATURE_COLUMNS),
            )
            model = train_model(
                X_fit,
                y_fit_t,
                settings.ml_random_seed,
                loss=train_loss,
                eval_set=(X_val, y_val_t),
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                depth=depth,
                l2_leaf_reg=l2_leaf_reg,
                learning_rate=learning_rate,
            )
            try:
                save_model(model, model_path)
                log.info("ml.model.saved", kind=kind, path=str(model_path))
            except OSError as exc:  # ro-монтирование в Airflow — не фатально
                log.warning("ml.model.save_skipped", kind=kind, path=str(model_path), error=str(exc))

        models[kind] = model
        if kind == "point":
            feature_importance = _feature_importance(model)

    # Smearing-поправки для log_rmse-вариантов (по val, детерминированно).
    smears: dict[str, float] = {
        kind: _smear_factor(models[kind], X_val, y_val, loss)
        for kind, loss in point_specs
    }

    # Основной точечный прогноз (с обратным преобразованием) + честный интервал q25/q75.
    primary_pred = _inverse_point_transform(
        np.asarray(models["point"].predict(X_test), dtype="float64"),
        primary_loss,
        smear=smears["point"],
    )
    test_lo, test_med, test_hi = _honest_interval(
        np.asarray(models["q25"].predict(X_test), dtype="float64"),
        primary_pred,
        np.asarray(models["q75"].predict(X_test), dtype="float64"),
    )

    metrics = evaluate(y_test, test_med)
    # Метрики всех точечных вариантов (основной + сравнение).
    point_metrics = {}
    for kind, loss in point_specs:
        pred = _inverse_point_transform(
            np.asarray(models[kind].predict(X_test), dtype="float64"),
            loss,
            smear=smears[kind],
        )
        point_metrics[loss] = evaluate(y_test, pred)

    picp = prediction_interval_coverage(y_test, test_lo, test_hi)
    direction_df = evaluate_per_direction(
        pd.DataFrame(
            {
                "origin_region": X_test["origin_region"].to_numpy(),
                "destination_region": X_test["destination_region"].to_numpy(),
                "y_true": y_test,
                "y_pred": test_med,
            }
        )
    )
    baselines = _fit_baselines(X_fit, y_fit, X_test, y_test, settings.ml_random_seed)

    hyper = {"depth": depth, "l2_leaf_reg": l2_leaf_reg, "learning_rate": learning_rate}
    for kind, loss in point_specs:
        m = point_metrics[loss]
        role = "primary" if loss == primary_loss else "compare"
        extra = {"picp": round(picp, 4)} if role == "primary" else {}
        log.info(
            "ml.metrics",
            loss=loss,
            role=role,
            **hyper,
            r2=round(m["r2"], 6),
            mape_pct=round(m["mape_pct"], 4),
            rmse=round(m["rmse"], 2),
            mae=round(m["mae"], 2),
            bias_pct=round(m["bias_pct"], 4),
            **extra,
        )

    # Лучший альтернативный вариант по MAPE — для сводной строки ml.compare.
    other_losses = [loss for _, loss in point_specs if loss != primary_loss]
    best_other = min(other_losses, key=lambda l: point_metrics[l]["mape_pct"])
    log.info(
        "ml.compare",
        primary_loss=primary_loss,
        primary_mape_pct=round(metrics["mape_pct"], 4),
        best_other_loss=best_other,
        best_other_mape_pct=round(point_metrics[best_other]["mape_pct"], 4),
        # delta > 0 → основной вариант лучше (MAPE ниже).
        delta_mape_pct=round(point_metrics[best_other]["mape_pct"] - metrics["mape_pct"], 4),
        **hyper,
    )
    for name, m in baselines.items():
        log.info(
            "ml.baseline",
            model=name,
            r2=round(m["r2"], 6),
            mape_pct=round(m["mape_pct"], 4),
            rmse=round(m["rmse"], 2),
            mae=round(m["mae"], 2),
        )
    for name, imp in sorted(feature_importance.items(), key=lambda kv: -kv[1])[:10]:
        log.info("ml.feature_importance", feature=name, importance=round(imp, 4))

    # Прогноз будущих заявок (основной точечный вариант + интервал q25/q75).
    if X_future is not None:
        future_primary = _inverse_point_transform(
            np.asarray(models["point"].predict(X_future), dtype="float64"),
            primary_loss,
            smear=smears["point"],
        )
        future_lo, future_med, future_hi = _honest_interval(
            np.asarray(models["q25"].predict(X_future), dtype="float64"),
            future_primary,
            np.asarray(models["q75"].predict(X_future), dtype="float64"),
        )
    else:
        future_lo = future_med = future_hi = np.array([], dtype="float64")

    scored_at = dt.datetime.now(dt.timezone.utc)
    pred_df = _predictions_frame(
        wide,
        body_group_test,
        test_mask,
        y_test,
        test_lo,
        test_med,
        test_hi,
        future_wide,
        body_group_future,
        future_lo,
        future_med,
        future_hi,
        scored_at,
    )
    rows = _upsert_predictions(dwh, pred_df)
    log.info(
        "ml.scoring.done",
        predictions_rows=rows,
        test_scored=int(int(test_mask.sum())),
        future_scored=int(int(len(future_wide))),
        model_name=MODEL_NAME,
    )

    return PipelineStats(
        as_of=as_of_date.isoformat(),
        rows_read=int(len(wide)),
        train_rows=int(len(X_train)),
        val_rows=int(len(X_val)),
        test_rows=int(len(X_test)),
        future_rows=int(len(future_wide)),
        predictions_rows=rows,
        metrics=metrics,
        point_metrics=point_metrics,
        baselines=baselines,
        picp=picp,
        inlier_lo=inlier_lo,
        inlier_hi=inlier_hi,
        direction_metrics=direction_df.head(20).to_dict("records"),
    )
