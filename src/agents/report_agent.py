"""Report Agent scaffold for pipeline orchestration."""

from __future__ import annotations

from src.agents.state import IPVState


def build_report_payload(state: IPVState) -> dict[str, object]:
    """Собирает заготовку report payload из текущего state."""
    return {
        "position": state.get("position"),
        "market_snapshot_t0": state.get("market_snapshot_t0"),
        "market_snapshot_t1": state.get("market_snapshot_t1"),
        "pricing_result": state.get("pricing_result"),
        "attribution_result": state.get("attribution_result"),
        "narrative_result": state.get("narrative_result"),
    }


def run_report_agent(state: IPVState) -> IPVState:
    """Заполняет state полем report_result.

    Полная сборка итогового отчёта будет сделана на следующем шаге реализации.
    """
    _ = state
    raise NotImplementedError("Report assembly will be implemented in the next step.")
