"""Заготовка для построения vol surface в quant-слое."""

from __future__ import annotations

from typing import Any


def build_vol_surface_from_option_quotes(
    option_quotes: dict[str, Any],
    *,
    snapshot_date: str,
    underlier: str,
) -> dict[str, Any]:
    """Заготовка для quant-логики построения поверхности волатильности."""
    raise NotImplementedError(
        "Vol surface should be derived by Quant Core from option_quotes."
    )
