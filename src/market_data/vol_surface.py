"""Построение и нормализация поверхности волатильности."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class VolSurfacePoint:
    """Одна точка поверхности волатильности."""

    tenor: str
    strike: float
    implied_vol: float


@dataclass(slots=True)
class VolSurfaceSnapshot:
    """Нормализованная поверхность волатильности по базовому активу."""

    snapshot_date: str
    underlier: str
    points: list[VolSurfacePoint] = field(default_factory=list)
    source: str = "moex"


def _rows_from_block(payload: dict[str, Any], block_name: str) -> list[dict[str, Any]]:
    """Преобразует табличный блок MOEX в список словарей."""
    block = payload.get(block_name, {})
    columns = block.get("columns", [])
    data = block.get("data", [])
    return [dict(zip(columns, row, strict=False)) for row in data]


def _to_float(value: Any, default: float = 0.0) -> float:
    """Приводит числовые значения к float."""
    if value in (None, "", "nan"):
        return default
    return float(value)


def normalize_option_quotes_to_surface(
    payload: dict[str, Any],
    *,
    snapshot_date: str,
    underlier: str,
    source: str = "moex",
) -> VolSurfaceSnapshot:
    """Нормализует сырые опционные котировки в заготовку vol surface."""
    rows = _rows_from_block(payload, "history")
    points: list[VolSurfacePoint] = []

    for row in rows:
        strike = row.get("STRIKE")
        implied_vol = row.get("IMPLIEDVOLATILITY") or row.get("VOLATILITY")
        tenor = row.get("TENOR") or row.get("MATURITY") or row.get("EXPIRATION")

        if strike in (None, "") or implied_vol in (None, ""):
            continue

        tenor_value = str(tenor) if tenor not in (None, "") else "UNKNOWN"
        points.append(
            VolSurfacePoint(
                tenor=tenor_value,
                strike=_to_float(strike),
                implied_vol=_to_float(implied_vol),
            )
        )

    return VolSurfaceSnapshot(
        snapshot_date=snapshot_date,
        underlier=underlier,
        points=points,
        source=source,
    )


def vol_surface_to_dict(snapshot: VolSurfaceSnapshot) -> dict[str, Any]:
    """Преобразует поверхность волатильности в JSON-friendly словарь."""
    return {
        "snapshot_date": snapshot.snapshot_date,
        "underlier": snapshot.underlier,
        "points": [
            {
                "tenor": point.tenor,
                "strike": point.strike,
                "implied_vol": point.implied_vol,
            }
            for point in snapshot.points
        ],
        "source": snapshot.source,
    }


def build_mock_vol_surface(
    snapshot_date: str,
    *,
    underlier: str = "SBER",
    source: str = "mock",
) -> dict[str, Any]:
    """Возвращает mock-поверхность волатильности для демо и fallback."""
    snapshot = VolSurfaceSnapshot(
        snapshot_date=snapshot_date,
        underlier=underlier,
        points=[
            VolSurfacePoint(tenor="1M", strike=280.0, implied_vol=0.24),
            VolSurfacePoint(tenor="3M", strike=290.0, implied_vol=0.26),
        ],
        source=source,
    )
    return vol_surface_to_dict(snapshot)
