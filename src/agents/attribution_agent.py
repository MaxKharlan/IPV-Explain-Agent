"""Attribution Agent scaffold for pipeline orchestration."""

from __future__ import annotations

from src.agents.state import IPVState


def build_risk_factor_inputs(state: IPVState) -> dict[str, object]:
    """Собирает заготовку risk-factor входов из state для Attribution Engine."""
    return {
        "position": state.get("position"),
        "pricing_result": state.get("pricing_result"),
        "market_snapshot_t0": state.get("market_snapshot_t0"),
        "market_snapshot_t1": state.get("market_snapshot_t1"),
    }


def run_attribution_agent(state: IPVState) -> IPVState:
    """Заполняет state полем attribution_result.

    Полный bridge к `src.attribution.run_attribution(...)`
    будет добавлен на следующем шаге реализации.
    """
    _ = state
    raise NotImplementedError("Attribution wiring will be implemented in the next step.")
