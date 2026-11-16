"""Юнит-тесты парсинга сетки grid-search тюнинга (этап 4, дополнение).

Проверяют чистые хелперы ``ml/tune.py`` без БД и без CatBoost/scikit-learn.
"""
from __future__ import annotations

import pytest

from ml import tune


def test_parse_ints_uses_default_when_none():
    assert tune._parse_ints(None, (4, 6, 8)) == [4, 6, 8]


def test_parse_ints_splits_and_strips():
    assert tune._parse_ints(" 6 , 8 ,10", ()) == [6, 8, 10]


def test_parse_floats_uses_default_when_none():
    assert tune._parse_floats(None, (1.0, 3.0)) == [1.0, 3.0]


def test_parse_floats_splits_and_strips():
    assert tune._parse_floats("3.0, 5.0", ()) == [3.0, 5.0]


def test_parse_losses_uses_default_when_none():
    assert tune._parse_losses(None, ("RMSE", "MAPE")) == ["RMSE", "MAPE"]


def test_parse_losses_strips_and_keeps_order():
    assert tune._parse_losses(" RMSE , log_rmse , MAPE ", ()) == ["RMSE", "log_rmse", "MAPE"]


def test_parse_losses_canonicalizes_case():
    # LOG_RMSE -> log_rmse (регрессия: регистр не должен ломать log_rmse)
    assert tune._parse_losses("LOG_RMSE, mape, rmse", ()) == ["log_rmse", "MAPE", "RMSE"]


def test_parse_losses_rejects_unknown():
    with pytest.raises(ValueError):
        tune._parse_losses("RMSE,poisson", ())
