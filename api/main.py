"""HTTP-сервис (FastAPI) демо-дашборда «Рынок ставок».

Отдаёт реальные агрегаты рынка ставок из DWH (``core.prices``) в формате
самодостаточной демо-страницы ``bigdata-demo.html``: еженедельный ряд ставок
(медиана/коридор/объём) по срезам направление/кузов/тоннаж/плечо, плюс
heatmap, бары по кузовам и топ направлений.

Запуск: ``uvicorn api.main:app`` (в Docker — см. ``Dockerfile.api``).
"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from api import queries

app = FastAPI(title="Smart Logistics — Market Rates API", version="1.0.0")

# Демо-страница раздаётся отдельно (статика), поэтому разрешаем кросс-доменные
# запросы с любого origin (file://, локальный dev-сервер и т.п.).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine: Optional[Engine] = None


def dwh_dsn() -> str:
    """DSN к хранилищу ``dwh`` (переменные окружения, dev-дефолты)."""
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    db = os.getenv("DWH_DB", "dwh")
    user = os.getenv("DWH_USER", "dwh_user")
    password = os.getenv("DWH_PASSWORD", "dwh_password")
    return f"postgresql+psycopg://{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{db}"


def engine() -> Engine:
    """Ленивый пул соединений (создаётся при первом запросе)."""
    global _engine
    if _engine is None:
        _engine = create_engine(dwh_dsn(), pool_pre_ping=True, future=True)
    return _engine


def _rows(sql: str, params: Optional[dict] = None) -> list[dict]:
    """Выполнить SELECT и вернуть список строк-словарей."""
    try:
        with engine().connect() as conn:
            result = conn.execute(text(sql), params or {})
            return [dict(row) for row in result.mappings()]
    except SQLAlchemyError as exc:  # БД недоступна — 503, а не 500.
        raise HTTPException(
            status_code=503, detail=f"DWH unavailable: {exc.__class__.__name__}"
        ) from exc


def _f(value, nd: int = 2) -> Optional[float]:
    """Число → округлённый float (None остаётся None)."""
    if value is None:
        return None
    return round(float(value), nd)


def _delta(cur, prev) -> Optional[float]:
    """Относительное изменение cur/prev − 1 (None при нехватке данных)."""
    if cur is None or prev is None or not prev:
        return None
    return round(cur / prev - 1, 4)


def _round4(value) -> Optional[float]:
    return None if value is None else round(value, 4)


def _kind(n: int, n_planned) -> str:
    """Класс недели: ``history`` | ``planned`` | ``mixed`` (по доле плановых)."""
    planned = int(n_planned or 0)
    if planned == 0:
        return "history"
    if planned >= int(n):
        return "planned"
    return "mixed"


def _window_end(points: list[dict], start, period: int) -> int:
    """Индекс (исключительный) конца окна ``period`` недель от ``start``.

    ``start`` — ``datetime.date`` или None. Без ``start`` окно прижато к концу
    ряда (последние ``period`` недель). Неделя начала окна — понедельник, в
    который попадает ``start``.
    """
    if start is None:
        return len(points)
    start_week = start - dt.timedelta(days=start.weekday())
    for i, p in enumerate(points):
        if dt.date.fromisoformat(p["d"]) >= start_week:
            return min(i + period, len(points))
    return 0


def _parse_dir(direction: Optional[str]):
    """``'all'`` → (None, None); ``'<gcid>:<gcid>'`` → (lc, uc)."""
    if not direction or direction == "all":
        return None, None
    try:
        lc, uc = direction.split(":")
        return int(lc), int(uc)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="dir must be 'all' or '<loading_gcid>:<unloading_gcid>'",
        )


def _windows():
    """(max_week, cur_start, prev_start) — окна «последние 4 / прошлые 4 недели»."""
    rows = _rows(queries.max_week_sql(), {})
    if not rows or rows[0]["max_week"] is None:
        return None, None, None
    max_week = rows[0]["max_week"]
    return max_week, max_week - dt.timedelta(days=21), max_week - dt.timedelta(days=49)


def _compute_stats(points: list[dict], period: int, median_trip, km, end: Optional[int] = None) -> dict:
    """KPI демо-страницы из недельного ряда (``end`` — конец окна, не включительно).

    Без ``end`` окно прижато к концу ряда (последние ``period`` недель), как и
    раньше; с ``end`` — к произвольной стартовой дате (см. ``_window_end``).
    """
    end = len(points) if end is None else min(end, len(points))
    if end <= 0:
        return {
            "median_km": None, "week_pct": None, "m4_pct": None,
            "ytd_pct": None, "trips": 0, "median_trip": median_trip, "km": km,
        }

    meds = [p["med"] for p in points[:end]]
    ns = [p["n"] for p in points[:end]]
    dates = [p["d"] for p in points[:end]]

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    cur = mean(meds[-4:])
    prev = mean(meds[-8:-4])
    week_pct = meds[-1] / meds[-2] - 1 if len(meds) >= 2 and meds[-2] else None
    m4_pct = cur / prev - 1 if cur and prev else None

    latest_year = dates[-1][:4]
    jan = [m for m, d in zip(meds, dates) if d[:4] == latest_year and d[5:7] == "01"]
    jan_med = mean(jan)
    ytd_pct = cur / jan_med - 1 if cur and jan_med else None

    return {
        "median_km": round(cur, 2) if cur is not None else None,
        "week_pct": _round4(week_pct),
        "m4_pct": _round4(m4_pct),
        "ytd_pct": _round4(ytd_pct),
        "trips": int(sum(ns[-period:])),
        "median_trip": median_trip,
        "km": km,
    }


@app.get("/health")
def health():
    try:
        with engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    return {"status": "ok" if db_ok else "degraded", "db": db_ok}


@app.get("/api/dims")
def dims(limit: int = Query(30, ge=1, le=200)):
    """Справочники срезов: направления (топ по сделкам) + body/ton/leg."""
    rows = _rows(queries.directions_sql(limit), {})
    directions = [
        {
            "key": f"{r['loading_city_gcid']}:{r['unloading_city_gcid']}",
            "from": r["from_name"],
            "to": r["to_name"],
            "km": _f(r["km"], 0),
            "trips": int(r["trips"]),
        }
        for r in rows
    ]
    return {
        "directions": directions,
        "bodies": queries.BODIES,
        "tons": queries.TONS,
        "legs": queries.LEGS,
    }


@app.get("/api/series")
def series(
    direction: Optional[str] = Query("all", alias="dir"),
    body: Optional[str] = Query("all"),
    ton: Optional[str] = Query("all"),
    leg: Optional[str] = Query("all"),
    period: int = Query(26, ge=1, le=52),
    start: Optional[dt.date] = Query(None, description="Стартовая дата окна (YYYY-MM-DD)"),
):
    """Еженедельный ряд ставок по выбранному срезу + KPI.

    ``start`` задаёт начало окна для KPI (``period`` недель от этой даты);
    без него окно прижато к концу данных.
    """
    lc, uc = _parse_dir(direction)
    params = queries.dim_params(
        lc, uc,
        None if body in (None, "all") else body,
        None if ton in (None, "all") else ton,
        None if leg in (None, "all") else leg,
    )

    rows = _rows(queries.series_sql(), params)
    points = [
        {
            "i": i,
            "d": r["week"].isoformat(),
            "med": _f(r["med"]),
            "lo": _f(r["lo"]),
            "hi": _f(r["hi"]),
            "n": int(r["n"]),
            "n_planned": int(r["n_planned"] or 0),
            "kind": _kind(r["n"], r["n_planned"]),
        }
        for i, r in enumerate(rows)
    ]

    srows = _rows(queries.stats_sql(), params)
    median_trip = _f(srows[0]["median_trip"]) if srows else None
    km = _f(srows[0]["km"], 0) if srows else None

    return {
        "points": points,
        "stats": _compute_stats(points, period, median_trip, km, end=_window_end(points, start, period)),
        "as_of": queries.AS_OF,
    }


@app.get("/api/forecast")
def forecast(
    direction: Optional[str] = Query("all", alias="dir"),
    body: Optional[str] = Query("all"),
    ton: Optional[str] = Query("all"),
    leg: Optional[str] = Query("all"),
):
    """Прогноз ставки ₽/км (med — точечная MAPE-модель; lo/hi — q25/q75) на 4 недели вперёд."""
    lc, uc = _parse_dir(direction)
    params = queries.dim_params(
        lc, uc,
        None if body in (None, "all") else body,
        None if ton in (None, "all") else ton,
        None if leg in (None, "all") else leg,
    )
    as_of = dt.date.fromisoformat(queries.AS_OF)
    params.update(
        as_of=as_of,
        horizon=as_of + dt.timedelta(days=queries.FORECAST_HORIZON_DAYS),
        model_name=queries.FORECAST_MODEL_NAME,
    )
    rows = _rows(queries.forecast_sql(), params)
    return {
        "points": [
            {
                "d": r["week"].isoformat(),
                "lo": _f(r["lo"]),
                "med": _f(r["med"]),
                "hi": _f(r["hi"]),
                "n": int(r["n"]),
            }
            for r in rows
        ],
        "as_of": queries.AS_OF,
        "horizon": (as_of + dt.timedelta(days=queries.FORECAST_HORIZON_DAYS)).isoformat(),
    }


@app.get("/api/heat")
def heat(
    direction: Optional[str] = Query("all", alias="dir"),
    ton: Optional[str] = Query("all"),
):
    """Heatmap «кузов × плечо»: ставка за 4 недели, Δ, число сделок."""
    lc, uc = _parse_dir(direction)
    _, cur_start, prev_start = _windows()
    if cur_start is None:
        return {"cells": []}

    params = queries.dim_params(lc, uc, None, None if ton in (None, "all") else ton, None)
    params.update(cur_start=cur_start, prev_start=prev_start)
    rows = _rows(queries.heat_sql(), params)
    return {
        "cells": [
            {
                "body": r["body_group"],
                "leg": r["leg"],
                "v": _f(r["v"]),
                "delta": _delta(r["v"], r["v_prev"]),
                "n": int(r["n"] or 0),
            }
            for r in rows
        ]
    }


@app.get("/api/bars")
def bars(
    direction: Optional[str] = Query("all", alias="dir"),
    ton: Optional[str] = Query("all"),
    leg: Optional[str] = Query("all"),
):
    """Бары по кузовам: ставка за 4 недели, Δ, число сделок."""
    lc, uc = _parse_dir(direction)
    _, cur_start, prev_start = _windows()
    if cur_start is None:
        return {"bars": []}

    params = queries.dim_params(
        lc, uc, None,
        None if ton in (None, "all") else ton,
        None if leg in (None, "all") else leg,
    )
    params.update(cur_start=cur_start, prev_start=prev_start)
    rows = _rows(queries.bars_sql(), params)
    return {
        "bars": [
            {
                "body": r["body_group"],
                "v": _f(r["v"]),
                "delta": _delta(r["v"], r["v_prev"]),
                "n": int(r["n"] or 0),
            }
            for r in rows
        ]
    }


@app.get("/api/top")
def top(
    body: Optional[str] = Query("all"),
    ton: Optional[str] = Query("all"),
    leg: Optional[str] = Query("all"),
    limit: int = Query(10, ge=1, le=50),
):
    """Топ направлений по сделкам за 4 недели + спарклайн за 13 недель."""
    max_week, cur_start, prev_start = _windows()
    if cur_start is None:
        return {"rows": []}

    params = queries.dim_params(
        None, None,
        None if body in (None, "all") else body,
        None if ton in (None, "all") else ton,
        None if leg in (None, "all") else leg,
    )
    params.update(cur_start=cur_start, prev_start=prev_start)
    rows = _rows(queries.top_sql(limit), params)

    sparks: dict = {}
    if rows:
        pairs = [(r["loading_city_gcid"], r["unloading_city_gcid"]) for r in rows]
        spark_start = max_week - dt.timedelta(days=84)  # 13 недель
        for r in _rows(queries.spark_sql(pairs), {"spark_start": spark_start}):
            key = (r["loading_city_gcid"], r["unloading_city_gcid"])
            sparks.setdefault(key, []).append({"d": r["week"].isoformat(), "v": _f(r["med"])})

    return {
        "rows": [
            {
                "key": f"{r['loading_city_gcid']}:{r['unloading_city_gcid']}",
                "from": r["from_name"],
                "to": r["to_name"],
                "km": _f(r["km"], 0),
                "n": int(r["n_cur"] or 0),
                "v": _f(r["v_cur"]),
                "delta": _delta(r["v_cur"], r["v_prev"]),
                "spark": sparks.get(
                    (r["loading_city_gcid"], r["unloading_city_gcid"]), []
                ),
            }
            for r in rows
        ]
    }


# ---------------------------------------------------------------------------
# Раздача статического дашборда (web/index.html) на корневом пути.
# Монтируется ПОСЛЕДНИМ: маршруты /health, /api/* и /docs приоритетнее.
# ---------------------------------------------------------------------------
_WEB_DIR = Path(__file__).resolve().parent.parent / "web"
app.mount("/", StaticFiles(directory=str(_WEB_DIR), html=True), name="web")
