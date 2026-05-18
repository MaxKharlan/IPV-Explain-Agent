"""Typed state contract for pipeline orchestration."""

from __future__ import annotations

from typing import Any, TypedDict


class FallbackFlags(TypedDict):
    """Флаги fallback-сценариев в agent pipeline."""

    used_mock_market_data: bool
    used_template_narrative: bool


class IPVState(TypedDict):
    """Общее состояние пайплайна между market/pricing/attribution/report узлами."""

    position: dict[str, Any] | None
    market_snapshot_t0: dict[str, Any] | None
    market_snapshot_t1: dict[str, Any] | None
    pricing_result: dict[str, Any] | None
    attribution_result: dict[str, Any] | None
    narrative_result: dict[str, Any] | None
    report_result: dict[str, Any] | None
    errors: list[str]
    fallback_flags: FallbackFlags


def create_initial_state(position: dict[str, Any] | None = None) -> IPVState:
    """Создаёт начальное состояние для запуска pipeline."""
    return {
        "position": position,
        "market_snapshot_t0": None,
        "market_snapshot_t1": None,
        "pricing_result": None,
        "attribution_result": None,
        "narrative_result": None,
        "report_result": None,
        "errors": [],
        "fallback_flags": {
            "used_mock_market_data": False,
            "used_template_narrative": False,
        },
    }


IPV_STATE_SHAPE: IPVState = create_initial_state()
