"""Чтение очищенного слоя ``core`` для ML-пайплайна (этап 4).

Формирует единый wide-кадр: сделки из ``core.prices`` (нормализованная цена
``price_rub``, категории груза) + дедуплицированный справочник городов
``core.cities`` (регион/название/координаты отправления и назначения).

Два разных набора строк (граница — ``as_of``):

* ``load_ml_frame`` — **фактические** прямые перевозки до ``as_of`` включительно
  (``is_fact``, ``complex=false``, ``price_rub > 0``, ``distance > 0``). Это
  универсум обучения и отложенного теста.
* ``load_future_frame`` — **плановые/будущие** заявки после ``as_of``
  (``complex=false``, ``distance > 0``, без требования ``is_fact``/``price_rub``),
  которые нужно скорить. Факт и ошибки для них — ``NULL``.

Разбиение train/test и очистка выбросов выполняются в ``ml/pipeline.py``.
"""
from __future__ import annotations

import datetime as dt
from typing import Optional

import pandas as pd
from sqlalchemy import Engine, text

from common.config import Settings, get_settings
from etl.db import dwh_engine

# Префиксы origin_*/dest_* согласованы с ml/features.py (build_feature_matrix).
_SELECT_CORE = """
SELECT
    p.price_id,
    p.loading_date,
    p.loading_city_gcid,
    p.unloading_city_gcid,
    p.price_rub,
    p.outlier_mark,
    p.distance,
    p.total_weight,
    p.total_volume,
    p.body_type,
    p.weight_category,
    p.volume_category,
    cl.name              AS origin_name,
    cl.full_name_local   AS origin_full_name_local,
    cl.latitude          AS origin_lat,
    cl.longitude         AS origin_lon,
    cu.name              AS dest_name,
    cu.full_name_local   AS dest_full_name_local,
    cu.latitude          AS dest_lat,
    cu.longitude         AS dest_lon
FROM core.prices p
JOIN core.cities cl ON cl.gc_id = p.loading_city_gcid
JOIN core.cities cu ON cu.gc_id = p.unloading_city_gcid
"""

# Факты для train/test: только закрытые прямые сделки с валидной ценой/дистанцией,
# не позже даты снимка (будущее читается отдельным запросом load_future_frame).
_FEATURE_SQL = _SELECT_CORE + """
WHERE p.is_fact
  AND NOT p.complex
  AND p.price_rub > 0
  AND p.distance > 0
  AND p.loading_date <= :as_of
ORDER BY p.price_id
"""

# Плановые/будущие заявки для batch-скоринга: не требуем is_fact и цену (факт
# ещё неизвестен), но требуем дистанцию и не-комплексность для построения признаков.
_FUTURE_SQL = _SELECT_CORE + """
WHERE NOT p.complex
  AND p.distance > 0
  AND p.loading_date > :as_of
ORDER BY p.price_id
"""


def _normalize_date(df: pd.DataFrame) -> pd.DataFrame:
    df["loading_date"] = pd.to_datetime(df["loading_date"])
    return df


def load_ml_frame(
    engine: Optional[Engine] = None,
    settings: Optional[Settings] = None,
    as_of: Optional[dt.date] = None,
) -> pd.DataFrame:
    """Прочитать wide-кадр фактов для ML (train/test, ``loading_date <= as_of``)."""
    eng = engine or dwh_engine(settings or get_settings())
    if as_of is None:
        as_of = dt.date.fromisoformat((settings or get_settings()).as_of)
    df = pd.read_sql(text(_FEATURE_SQL), eng, params={"as_of": as_of})
    return _normalize_date(df)


def load_future_frame(
    engine: Optional[Engine] = None,
    settings: Optional[Settings] = None,
    as_of: Optional[dt.date] = None,
) -> pd.DataFrame:
    """Прочитать wide-кадр будущих заявок для скоринга (``loading_date > as_of``)."""
    eng = engine or dwh_engine(settings or get_settings())
    if as_of is None:
        as_of = dt.date.fromisoformat((settings or get_settings()).as_of)
    df = pd.read_sql(text(_FUTURE_SQL), eng, params={"as_of": as_of})
    return _normalize_date(df)
