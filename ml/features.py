"""Инженерия признаков модели прогнозирования тарифа (этап 4).

Чистые функции над ``pandas``-объектами без I/O, чтобы правила было просто
покрывать юнит-тестами (``tests/test_ml_features.py``).

Ключевой инвариант — **защита от утечек (Data Leakage)**. Матрица признаков
строится только из разрешённых колонок ``FEATURE_COLUMNS``; любые производные
целевой переменной (``price``/``price_rub``, ``rate_value``, удельные тарифы,
``price_group_stats``, ``outlier_mark`` и т. д.) запрещены и проверяются
``assert_no_leakage``. Целевая переменная ``price_rub`` и флаг ``outlier_mark``
используются **только** вне матрицы признаков (таргет и очистка выборки).
"""
from __future__ import annotations

import datetime as dt
from typing import Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Группы кузова (тот же маппинг, что и fallback-ветка sql/dwh/02_mart_route_card.sql:
# prices.body_type уже хранит std_id, дочерние id остались в price_body_types).
# ---------------------------------------------------------------------------
BODY_GROUP_MAP = {56: "Тент", 4: "Рефрижератор", 2: "Изотерм", 15: "Борт"}
BODY_GROUP_OTHER = "Прочие"
BODY_GROUP_NONE = "Не указан"

# Числовые признаки (порядок важен: CatBoost при predict сопоставляет по именам,
# но фиксированный порядок делает вывод детерминированным и прозрачным).
NUMERIC_FEATURES: Sequence[str] = (
    "distance",
    "total_weight",
    "total_volume",
    "origin_lat",
    "origin_lon",
    "dest_lat",
    "dest_lon",
    "year",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
)

# Категориальные признаки: география кодируется раздельно (отправление/назначение,
# регион и город), а не склеенной строкой маршрута высокой кардинальности.
CATEGORICAL_FEATURES: Sequence[str] = (
    "origin_region",
    "destination_region",
    "origin_city",
    "destination_city",
    "body_group",
    "weight_category",
    "volume_category",
)

FEATURE_COLUMNS: Sequence[str] = tuple(NUMERIC_FEATURES) + tuple(CATEGORICAL_FEATURES)

# Колонки, которых НЕ должно быть в матрице признаков (прямые/косвенные
# производные целевой переменной). Часть из них отсутствует в схеме источника,
# но перечислена, чтобы тест ловил будущее добавление таких полей.
FORBIDDEN_FEATURES: frozenset = frozenset(
    {
        "price",
        "price_rub",
        "price_with_vat",
        "price_with_nds",
        "price_nds",
        "rate_value",
        "currency_code",
        "nearest_rate_date",
        "outlier_mark",
        "price_type",
        "is_fact",
        "complex",
        "price_group_id",
        "customer_cost",
        "carrier_cost",
        "transportation_cost",
        "profit",
        "avg_price",
        "min_km_price",
        "max_km_price",
        "price_km",
        "price_per_weight",
    }
)


def extract_region(full_name_local) -> str:
    """Извлечь регион из ``full_name_local`` формата «Город, Регион».

    Примеры:
        'Казань, Республика Татарстан' -> 'Республика Татарстан'
        'Москва, г. Москва'            -> 'г. Москва'
        'Тверь'                        -> 'Тверь' (нет запятой — вся строка)

    Пустые/``NULL`` значения возвращаются как ``""``.
    """
    if full_name_local is None or full_name_local is pd.NaT or full_name_local is pd.NA:
        return ""
    if isinstance(full_name_local, float) and np.isnan(full_name_local):
        return ""
    text = str(full_name_local).strip()
    if not text or text in {"nan", "None", "<NA>"}:
        return ""
    if "," in text:
        return text.split(",", 1)[1].strip()
    return text


def body_group(std_id) -> str:
    """Привести ``std_id`` кузова к стандартизированной бизнес-группе.

    ``NULL``/``NaN`` -> «Не указан»; неизвестный код -> «Прочие».
    """
    if std_id is None or std_id is pd.NaT or std_id is pd.NA:
        return BODY_GROUP_NONE
    if isinstance(std_id, float) and np.isnan(std_id):
        return BODY_GROUP_NONE
    return BODY_GROUP_MAP.get(int(std_id), BODY_GROUP_OTHER)


def cyclical_time_features(dates: pd.Series) -> pd.DataFrame:
    """Циклическое кодирование даты погрузки (сохраняет близость 31.12 ↔ 01.01).

    Возвращает DataFrame с колонками:
    * ``year`` — числовой долгосрочный инфляционный тренд;
    * ``month_sin/cos`` (T = 12), ``dow_sin/cos`` (T = 7), ``doy_sin/cos``
      (T = 365.25) — гармонические признаки по формуле sin(2π·t/T), cos(2π·t/T).
    """
    d = pd.to_datetime(dates)
    month = d.dt.month.astype("float64")
    dow = d.dt.dayofweek.astype("float64")  # 0=Пн .. 6=Вс
    doy = d.dt.dayofyear.astype("float64")
    return pd.DataFrame(
        {
            "year": d.dt.year.astype("float64"),
            "month_sin": np.sin(2 * np.pi * month / 12.0),
            "month_cos": np.cos(2 * np.pi * month / 12.0),
            "dow_sin": np.sin(2 * np.pi * dow / 7.0),
            "dow_cos": np.cos(2 * np.pi * dow / 7.0),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        },
        index=dates.index,
    )


def build_feature_matrix(wide: pd.DataFrame) -> pd.DataFrame:
    """Построить матрицу признаков (ровно ``FEATURE_COLUMNS``) из wide-кадра.

    ``wide`` — слой core с уже присоединёнными справочниками городов: содержит
    ``distance``, ``total_weight``, ``total_volume``, ``loading_date``,
    ``body_type``, ``weight_category``, ``volume_category`` и префиксованные
    колонки городов ``origin_*`` / ``dest_*``.
    """
    out = pd.DataFrame(index=wide.index)

    for col in (
        "distance",
        "total_weight",
        "total_volume",
        "origin_lat",
        "origin_lon",
        "dest_lat",
        "dest_lon",
    ):
        out[col] = pd.to_numeric(wide[col], errors="coerce").astype("float64")

    time = cyclical_time_features(wide["loading_date"])
    for col in (
        "year",
        "month_sin",
        "month_cos",
        "dow_sin",
        "dow_cos",
        "doy_sin",
        "doy_cos",
    ):
        out[col] = time[col].astype("float64")

    out["origin_region"] = wide["origin_full_name_local"].map(extract_region).astype("string")
    out["destination_region"] = wide["dest_full_name_local"].map(extract_region).astype("string")
    out["origin_city"] = wide["origin_name"].astype("string")
    out["destination_city"] = wide["dest_name"].astype("string")
    out["body_group"] = wide["body_type"].map(body_group).astype("string")
    out["weight_category"] = wide["weight_category"].astype("string")
    out["volume_category"] = wide["volume_category"].astype("string")

    out = out[list(FEATURE_COLUMNS)]
    assert_no_leakage(out)
    return out


def assert_no_leakage(features: pd.DataFrame) -> None:
    """Бросить ``ValueError``, если в матрице есть запрещённые колонки-утечки."""
    bad = sorted(set(features.columns) & FORBIDDEN_FEATURES)
    if bad:
        raise ValueError(f"Обнаружены колонки-утечки в матрице признаков: {bad}")


def assign_split(loading_date, as_of: dt.date) -> str:
    """Отнести дату погрузки к train/test/future (границы задания, вопрос 3).

    * ``before_train`` : дата раньше train-окна (в ML-выборку не входит);
    * ``train``        : [as_of - 3 мес, as_of - 3 нед) — ``test_start`` не входит;
    * ``test``         : [as_of - 3 нед, as_of] — обе границы включительно;
    * ``future``       : (as_of, +∞).
    """
    train_start, test_start, test_end = time_split_bounds(as_of)
    if isinstance(loading_date, pd.Timestamp):
        d = loading_date.date()
    elif isinstance(loading_date, dt.datetime):
        d = loading_date.date()
    else:
        d = loading_date
    if d < train_start:
        return "before_train"
    if train_start <= d < test_start:
        return "train"
    if test_start <= d <= test_end:
        return "test"
    return "future"


def time_split_bounds(as_of: dt.date) -> tuple[dt.date, dt.date, dt.date]:
    """Детерминированные границы временного разбиения для даты снимка ``as_of``.

    Возвращает ``(train_start, test_start, test_end)``, где
    ``train_start = as_of - 3 месяца``, ``test_start = as_of - 3 недели``,
    ``test_end = as_of``. Для ``as_of = 2026-10-26`` даёт эталонные границы
    ``2026-07-26`` / ``2026-10-05`` / ``2026-10-26``.
    """
    train_start = _sub_months(as_of, 3)
    test_start = as_of - dt.timedelta(days=21)
    return train_start, test_start, as_of


def _sub_months(d: dt.date, months: int) -> dt.date:
    """Вычесть ``months`` календарных месяцев с ограничением дня длиной месяца."""
    m = d.month - months
    y = d.year
    while m <= 0:
        m += 12
        y -= 1
    day = min(d.day, _days_in_month(y, m))
    return dt.date(y, m, day)


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        return 31
    return (dt.date(year, month + 1, 1) - dt.timedelta(days=1)).day


def fit_inlier_bounds(y: pd.Series) -> tuple[float, float]:
    """Границы отсечения выбросов таргета по IQR логарифма (строго по ``train``).

    Отчёт §2.2: тариф логнормален, поэтому аномалии удобно резать по ``ln(y)``,
    а границы обязаны считаться **только по train** (защита от утечки). Здесь
    возвращаются границы в исходных рублях ``(lo, hi)``:

    * ``lo = exp(Q1(ln y) - 1.5·IQR)``, ``hi = exp(Q3(ln y) + 1.5·IQR)``;
    * при вырожденном IQR (<=0) или слишком малой выборке (<4) возвращается
      ``(-inf, +inf)`` — то есть отсечение не применяется.

    Функция чистая и детерминированная; вызывается только на train-маске,
    результат применяется к train/test без пересчёта на них.
    """
    y = pd.to_numeric(y, errors="coerce").dropna()
    y = y[y > 0]
    if len(y) < 4:
        return (float("-inf"), float("inf"))
    log_y = np.log(y)
    q1, q3 = log_y.quantile(0.25), log_y.quantile(0.75)
    iqr = q3 - q1
    if iqr <= 0:
        return (float("-inf"), float("inf"))
    return (float(np.exp(q1 - 1.5 * iqr)), float(np.exp(q3 + 1.5 * iqr)))


def chronological_val_mask(dates: pd.Series, val_fraction: float = 0.2) -> pd.Series:
    """Хвост train-окна (по ``loading_date``) под ``val`` — без перемешивания.

    Возвращает булев ``pd.Series``, выровненный по ``dates.index``, где ``True``
    помечены последние ``val_fraction`` записей в хронологическом порядке. Это
    сохраняет схему Time-Based Split отчёта §4: ранняя история → обучение,
    поздняя → ранняя остановка/подбор гиперпараметров, никогда не «подглядывая»
    в будущее относительно валидационного хвоста.
    """
    dates = pd.Series(pd.to_datetime(dates), index=dates.index)
    n = len(dates)
    if n == 0:
        return pd.Series(dtype=bool, index=dates.index)
    n_val = max(1, int(round(n * val_fraction)))
    n_val = min(n_val, n - 1) if n > 1 else n_val
    order = dates.sort_values(kind="mergesort").index
    mask = pd.Series(False, index=dates.index)
    mask.loc[order[-n_val:]] = True
    return mask
