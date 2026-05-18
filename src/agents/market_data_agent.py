"""Market Data Agent scaffold for pipeline orchestration."""

from __future__ import annotations

from typing import Any

from src.agents.state import IPVState


def extract_market_request(position: dict[str, Any]) -> dict[str, Any]:
    """Готовит минимальный запрос к Market Data Layer из PositionInput."""
    as_of_dates = position.get("as_of_dates", {})
    instrument = position.get("instrument", {})
    return {
        "t0": as_of_dates.get("t0"),
        "t1": as_of_dates.get("t1"),
        "security": instrument.get("underlier"),
    }


def run_market_data_agent(state: IPVState) -> IPVState:
    """Заполняет state полями market snapshot.

    Полноценный вызов `load_or_fetch_market_snapshots_for_period(...)`
    будет добавлен на следующем шаге реализации.
    """
    _ = state
    raise NotImplementedError("Market data wiring will be implemented in the next step.")
