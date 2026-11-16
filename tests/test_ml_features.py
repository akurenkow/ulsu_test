"""Юнит-тесты ML-признаков и защиты от утечек (этап 4).

Покрывают чистые функции ``ml/features.py``: извлечение регионов, группы кузова,
циклическое кодирование дат, состав матрицы признаков (без утечек) и границы
временного разбиения. Не требуют БД и не импортируют CatBoost/scikit-learn.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from ml import features


# ---------------------------------------------------------------------------
# Регион из full_name_local
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Казань, Республика Татарстан", "Республика Татарстан"),
        ("Москва, г. Москва", "г. Москва"),
        ("Екатеринбург, Свердловская обл.", "Свердловская обл."),
        ("Тверь", "Тверь"),  # без запятой — вся строка
        ("  Казань , Республика Татарстан ", "Республика Татарстан"),
        (None, ""),
        (float("nan"), ""),
    ],
)
def test_extract_region(value, expected):
    assert features.extract_region(value) == expected


# ---------------------------------------------------------------------------
# Группа кузова
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("std_id", "expected"),
    [
        (56, "Тент"),
        (4, "Рефрижератор"),
        (2, "Изотерм"),
        (15, "Борт"),
        (3, "Прочие"),
        (50, "Прочие"),
        (68, "Прочие"),
        (None, "Не указан"),
        (float("nan"), "Не указан"),
    ],
)
def test_body_group(std_id, expected):
    assert features.body_group(std_id) == expected


# ---------------------------------------------------------------------------
# Циклическое кодирование дат
# ---------------------------------------------------------------------------
def _date(s: str) -> pd.Series:
    return pd.Series([pd.Timestamp(s)])


def test_cyclical_time_features_shape_and_range():
    dates = pd.Series(pd.to_datetime(["2026-01-15", "2026-07-01", "2026-12-31"]))
    out = features.cyclical_time_features(dates)
    expected_cols = {
        "year", "month_sin", "month_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
    }
    assert set(out.columns) == expected_cols
    # гармонические признаки лежат в [-1; 1]
    for col in expected_cols - {"year"}:
        assert ((out[col] >= -1.0) & (out[col] <= 1.0)).all(), col
    assert list(out["year"]) == [2026, 2026, 2026]


def test_cyclical_year_boundary_closeness():
    # 31 декабря и 1 января близки на окружности дня года (T = 365.25).
    dec31 = features.cyclical_time_features(pd.Series([pd.Timestamp("2025-12-31")]))
    jan01 = features.cyclical_time_features(pd.Series([pd.Timestamp("2026-01-01")]))
    assert abs(dec31["doy_sin"].iloc[0] - jan01["doy_sin"].iloc[0]) < 0.05
    assert abs(dec31["doy_cos"].iloc[0] - jan01["doy_cos"].iloc[0]) < 0.05


# ---------------------------------------------------------------------------
# Матрица признаков и защита от утечек
# ---------------------------------------------------------------------------
def _wide_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "price_id": [1, 2],
            "loading_date": pd.to_datetime(["2026-09-15", "2026-10-20"]),
            "price_rub": [84500.0, 90000.0],
            "outlier_mark": [False, False],
            "distance": [897.0, 1200.0],
            "total_weight": [20.0, 18.5],
            "total_volume": [80.0, 75.0],
            "body_type": [56.0, 4.0],
            "weight_category": ["FTL", "FTL"],
            "volume_category": ["FTL", "FTL"],
            "origin_name": ["Казань", "Москва"],
            "origin_full_name_local": ["Казань, Республика Татарстан", "Москва, г. Москва"],
            "origin_lat": [55.7961, 55.7558],
            "origin_lon": [49.1064, 37.6173],
            "dest_name": ["Москва", "Казань"],
            "dest_full_name_local": ["Москва, г. Москва", "Казань, Республика Татарстан"],
            "dest_lat": [55.7558, 55.7961],
            "dest_lon": [37.6173, 49.1064],
        }
    )


def test_build_feature_matrix_exact_columns_and_no_leakage():
    X = features.build_feature_matrix(_wide_frame())
    assert list(X.columns) == list(features.FEATURE_COLUMNS)
    assert not (set(X.columns) & features.FORBIDDEN_FEATURES)
    assert X["origin_region"].tolist() == ["Республика Татарстан", "г. Москва"]
    assert X["body_group"].tolist() == ["Тент", "Рефрижератор"]
    assert X["distance"].tolist() == [897.0, 1200.0]


def test_assert_no_leakage_raises_on_forbidden_column():
    X = features.build_feature_matrix(_wide_frame())
    X["price_rub"] = 1.0  # инъекция запрещённой колонки
    with pytest.raises(ValueError):
        features.assert_no_leakage(X)


def test_feature_columns_do_not_contain_forbidden():
    assert not (set(features.FEATURE_COLUMNS) & features.FORBIDDEN_FEATURES)


# ---------------------------------------------------------------------------
# Временное разбиение (Time-Based Split)
# ---------------------------------------------------------------------------
def test_time_split_bounds_default_as_of():
    train_start, test_start, test_end = features.time_split_bounds(dt.date(2026, 10, 26))
    assert train_start == dt.date(2026, 7, 26)
    assert test_start == dt.date(2026, 10, 5)
    assert test_end == dt.date(2026, 10, 26)


@pytest.mark.parametrize(
    ("loading_date", "expected"),
    [
        (dt.date(2026, 7, 25), "before_train"),   # раньше train-окна (не исторический тест)
        (dt.date(2026, 7, 26), "train"),    # нижняя граница train включительно
        (dt.date(2026, 10, 4), "train"),    # день перед границей — train
        (dt.date(2026, 10, 5), "test"),     # граница входит только в test
        (dt.date(2026, 10, 26), "test"),    # верхняя граница test включительно
        (dt.date(2026, 10, 27), "future"),  # завтра после снимка — future
    ],
)
def test_assign_split_boundaries(loading_date, expected):
    assert features.assign_split(loading_date, dt.date(2026, 10, 26)) == expected


# ---------------------------------------------------------------------------
# Отсечение выбросов по IQR логарифма (строго по train) и хронологический val
# ---------------------------------------------------------------------------
def test_fit_inlier_bounds_rejects_extreme_anomalies():
    # Логнормальный таргет с разбросом ~100 и одной экстремальной аномалией.
    y = pd.Series(
        [95.0, 100.0, 105.0, 110.0, 98.0, 102.0, 97.0, 108.0, 101.0, 99.0,
         104.0, 96.0, 103.0, 107.0, 100.0, 94.0, 106.0, 102.0, 98.0, 101.0]
        + [10_000_000.0]
    )
    lo, hi = features.fit_inlier_bounds(y)
    assert lo < 100.0 < hi
    assert hi < 10_000_000.0  # аномалия выходит за верхнюю границу


def test_fit_inlier_bounds_degenerate_returns_infinite():
    y = pd.Series([100.0, 100.0, 100.0])
    lo, hi = features.fit_inlier_bounds(y)
    assert lo == float("-inf")
    assert hi == float("inf")


def test_fit_inlier_bounds_ignores_nonpositive():
    y = pd.Series([0.0, -5.0, 100.0, 120.0, 130.0, 140.0])
    lo, hi = features.fit_inlier_bounds(y)
    assert lo < 100.0 < hi


def test_chronological_val_mask_takes_tail_without_shuffle():
    dates = pd.Series(
        pd.to_datetime(
            ["2026-07-26", "2026-09-01", "2026-10-01", "2026-10-04", "2026-08-15"]
        )
    )
    mask = features.chronological_val_mask(dates, val_fraction=0.4)
    # 40% от 5 = 2 последние по дате записи: 2026-10-01 и 2026-10-04.
    assert set(dates[mask].dt.date.astype(str)) == {"2026-10-01", "2026-10-04"}
    assert mask.index.equals(dates.index)


def test_chronological_val_mask_keeps_at_least_one_train_row():
    dates = pd.Series(pd.to_datetime(["2026-09-01", "2026-09-02"]))
    mask = features.chronological_val_mask(dates, val_fraction=1.0)
    # даже при 100% хвост не забирает всю выборку — остаётся хотя бы одна строка в train.
    assert int(mask.sum()) <= len(dates) - 1
