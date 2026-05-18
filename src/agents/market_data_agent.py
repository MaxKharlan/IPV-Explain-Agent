"""Market Data Agent scaffold for pipeline orchestration."""

from __future__ import annotations

from typing import Any

from src.agents.state import IPVState
from src.market_data.moex_client import load_or_fetch_market_snapshots_for_period


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
    """Заполняет state полями market snapshot."""
    position = state.get("position")
    if position is None:
        raise ValueError("Market data agent requires position payload in state.")

    request = extract_market_request(position)
    if not request["t0"] or not request["t1"] or not request["security"]:
        raise ValueError("Position must include as_of_dates.t0, as_of_dates.t1 and instrument.underlier.")

    snapshot_t0, snapshot_t1 = load_or_fetch_market_snapshots_for_period(
        request["t0"],
        request["t1"],
        security=request["security"],
    )
    state["market_snapshot_t0"] = snapshot_t0
    state["market_snapshot_t1"] = snapshot_t1

    quality_flags_t0 = snapshot_t0.get("quality_flags", {})
    quality_flags_t1 = snapshot_t1.get("quality_flags", {})
    state["fallback_flags"]["used_mock_market_data"] = bool(
        quality_flags_t0.get("used_mock_data") or quality_flags_t1.get("used_mock_data")
    )
    return state
