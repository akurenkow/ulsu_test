"""Юнит-тесты хронологического разбиения и построчной инженерии признаков
(этап 5, задача 5.4).

Дополняют ``test_ml_features.py``: проверяют, что маски ``train/test/future``
не пересекаются и границы соответствуют заданию, а построение матрицы признаков
— чистая построчная трансформация (никакие статистики не подгоняются на выборке,
значит утечки через импутацию/масштабирование нет).
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from ml import features
from ml.pipeline import (
    _canonical_point_loss,
    _honest_interval,
    _inverse_point_transform,
    _masks,
    _point_kind,
    _point_targets,
    _point_variants,
    _smear_factor,
    evaluate_per_direction,
    prediction_interval_coverage,
)


def test_masks_are_disjoint_and_strictly_chronological():
    as_of = dt.date(2026, 10, 26)
    wide = pd.DataFrame(
        {
            "price_id": range(6),
            "loading_date": pd.to_datetime(
                ["2026-07-20", "2026-07-26", "2026-10-04", "2026-10-05", "2026-10-26", "2026-10-27"]
            ),
            "outlier_mark": [False] * 6,
        }
    )
    train, test = _masks(wide, as_of)

    # Периоды не пересекаются.
    assert int((train & test).sum()) == 0

    # Строго по loading_date: train = [07-26, 10-05), test = [10-05, 10-26].
    # 2026-07-20 — раньше train-окна; 2026-10-27 — будущее (читается отдельно) —
    # в факты train/test не входят.
    assert set(wide.loc[train, "loading_date"].dt.date.astype(str)) == {
        "2026-07-26", "2026-10-04"
    }
    assert set(wide.loc[test, "loading_date"].dt.date.astype(str)) == {
        "2026-10-05", "2026-10-26"
    }


def test_feature_matrix_is_rowwise_transform():
    # Инженерия признаков не зависит от остальной выборки: матрица для одной
    # строки совпадает с соответствующей строкой матрицы для всей выборки.
    # Это гарантирует, что при обучении нет статистик, подогнанных на train
    # (SimpleImputer/StandardScaler/TargetEncoder отсутствуют в пайплайне).
    wide = pd.DataFrame(
        {
            "loading_date": pd.to_datetime(["2026-09-15", "2026-10-20"]),
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

    X_all = features.build_feature_matrix(wide)
    X_one = features.build_feature_matrix(wide.iloc[[0]])

    pd.testing.assert_frame_equal(X_one, X_all.iloc[[0]], check_dtype=False)


def test_honest_interval_preserves_point_and_brackets():
    lo = np.array([3.0, 5.0, 1.0])
    med = np.array([2.0, 5.0, 2.0])
    hi = np.array([4.0, 5.0, 3.0])

    a, b, c = _honest_interval(lo, med, hi)

    # Точечный прогноз не подменяется медианой тройки — сохраняется как есть.
    np.testing.assert_array_equal(b, med)
    # Инвариант lo <= med <= hi выполняется всегда.
    assert np.all(a <= b)
    assert np.all(b <= c)


def test_canonical_point_loss_is_case_insensitive():
    assert _canonical_point_loss("log_rmse") == "log_rmse"
    assert _canonical_point_loss("LOG_RMSE") == "log_rmse"  # регрессия: .upper() ломал log_rmse
    assert _canonical_point_loss(" Log_Rmse ") == "log_rmse"
    assert _canonical_point_loss("rmse") == "RMSE"
    assert _canonical_point_loss("mape") == "MAPE"
    # неизвестное возвращается как есть (фолбэк — на вызывающем)
    assert _canonical_point_loss("poisson") == "poisson"


def test_point_variants_order_and_fallback():
    assert _point_variants("MAPE") == [
        ("point", "MAPE"),
        ("point_rmse", "RMSE"),
        ("point_log_rmse", "log_rmse"),
    ]
    assert _point_variants("RMSE") == [
        ("point", "RMSE"),
        ("point_mape", "MAPE"),
        ("point_log_rmse", "log_rmse"),
    ]
    # log_rmse — основной вариант (регрессия: раньше .upper() ронял его на RMSE)
    assert _point_variants("log_rmse") == [
        ("point", "log_rmse"),
        ("point_mape", "MAPE"),
        ("point_rmse", "RMSE"),
    ]
    assert _point_variants("LOG_RMSE") == _point_variants("log_rmse")
    # регистр не важен
    assert _point_variants("rmse") == _point_variants("RMSE")
    # недопустимое значение детерминированно падает на дефолт (RMSE) первым.
    assert _point_variants("not-a-loss")[0] == ("point", "RMSE")


def test_point_kind_stable_artifact_names():
    assert _point_kind("MAPE", "MAPE") == "point"
    assert _point_kind("RMSE", "MAPE") == "point_rmse"
    assert _point_kind("log_rmse", "MAPE") == "point_log_rmse"
    assert _point_kind("RMSE", "RMSE") == "point"


def test_point_targets_log_rmse_transforms_to_log():
    y_fit = pd.Series([100.0, 200.0])
    y_val = pd.Series([150.0, 300.0])
    loss, yf, yv = _point_targets("log_rmse", y_fit, y_val)
    assert loss == "RMSE"
    np.testing.assert_allclose(yf, np.log(y_fit))
    np.testing.assert_allclose(yv, np.log(y_val))


def test_point_targets_passthrough_others():
    y_fit = pd.Series([100.0, 200.0])
    y_val = pd.Series([150.0, 300.0])
    loss, yf, yv = _point_targets("RMSE", y_fit, y_val)
    assert loss == "RMSE"
    assert yf.equals(y_fit)
    assert yv.equals(y_val)


def test_inverse_point_transform_log_rmse_exp_and_smear():
    pred = np.array([np.log(100.0), np.log(200.0)])
    out = _inverse_point_transform(pred, "log_rmse", smear=1.1)
    np.testing.assert_allclose(out, [110.0, 220.0])


def test_inverse_point_transform_passthrough_others():
    pred = np.array([1.0, 2.0])
    np.testing.assert_array_equal(_inverse_point_transform(pred, "RMSE", smear=1.1), pred)


class _PerfectLogModel:
    def predict(self, X):
        return np.asarray([np.log(150.0), np.log(300.0)])


def test_smear_factor_log_rmse_perfect_is_one():
    y_val = pd.Series([150.0, 300.0])
    assert _smear_factor(_PerfectLogModel(), None, y_val, "log_rmse") == pytest.approx(1.0)


def test_smear_factor_non_log_rmse_is_one():
    assert _smear_factor(None, None, pd.Series([1.0]), "RMSE") == 1.0


def test_honest_interval_expands_when_point_outside_quantiles():
    # Точечная MAPE-модель может предсказывать ниже q25 (систематическое занижение
    # из-за асимметрии MAPE) — интервал обязан расшириться до точечного прогноза.
    lo = np.array([100.0, 90.0])
    med = np.array([80.0, 120.0])   # 80 < q25=100 (занижение); 120 > q75=90 (завышение)
    hi = np.array([110.0, 95.0])

    a, b, c = _honest_interval(lo, med, hi)

    np.testing.assert_array_equal(b, med)
    assert np.all(a <= b)
    assert np.all(b <= c)
    # Нижняя граница расширяется до точечного прогноза, когда он ниже q25.
    assert a[0] == 80.0
    # Верхняя граница расширяется до точечного прогноза, когда он выше q75.
    assert c[1] == 120.0


def test_prediction_interval_coverage():
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    lo = np.array([90.0, 190.0, 350.0, 390.0])
    hi = np.array([110.0, 210.0, 390.0, 410.0])
    # внутрь попадают 1-я, 2-я и 4-я точки; 3-я (300) ниже lo (350).
    assert prediction_interval_coverage(y_true, lo, hi) == pytest.approx(0.75)


def test_evaluate_per_direction_uses_mape_mae_primary():
    df = pd.DataFrame(
        {
            "origin_region": ["Республика Татарстан", "Республика Татарстан", "г. Москва"],
            "destination_region": ["г. Москва", "г. Москва", "Республика Татарстан"],
            "y_true": [100.0, 200.0, 300.0],
            "y_pred": [110.0, 180.0, 300.0],
        }
    )
    out = evaluate_per_direction(df)
    assert set(out.columns) == {
        "origin_region", "destination_region", "n", "mape_pct", "mae", "r2",
    }
    # направлений два; отсортировано по убыванию числа сделок.
    assert out["n"].tolist() == [2, 1]
    kzn = out[out["origin_region"] == "Республика Татарстан"].iloc[0]
    assert kzn["mae"] == pytest.approx((10.0 + 20.0) / 2.0)
    assert kzn["mape_pct"] == pytest.approx((10.0 / 100.0 + 20.0 / 200.0) / 2.0 * 100.0)
