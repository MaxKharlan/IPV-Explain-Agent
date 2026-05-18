"""Report Agent scaffold for pipeline orchestration."""

from __future__ import annotations

from src.agents.state import IPVState


def build_report_payload(state: IPVState) -> dict[str, object]:
    """Собирает заготовку report payload из текущего state."""
    position = state.get("position") or {}
    attribution_result = state.get("attribution_result") or {}
    narrative_result = state.get("narrative_result") or {}
    return {
        "position": position,
        "market_snapshot_t0": state.get("market_snapshot_t0"),
        "market_snapshot_t1": state.get("market_snapshot_t1"),
        "pricing_result": state.get("pricing_result"),
        "attribution_result": attribution_result,
        "narrative_result": narrative_result,
        "report_summary": {
            "position_id": position.get("position_id"),
            "instrument_type": position.get("instrument_type"),
            "total_pnl": attribution_result.get("total_pnl"),
            "summary": narrative_result.get("summary"),
            "used_mock_market_data": state["fallback_flags"]["used_mock_market_data"],
            "used_template_narrative": state["fallback_flags"]["used_template_narrative"],
        },
    }


def run_report_agent(state: IPVState) -> IPVState:
    """Заполняет state полем report_result."""
    state["report_result"] = build_report_payload(state)
    return state
