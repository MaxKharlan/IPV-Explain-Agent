"""Общий контракт состояния для оркестрации в LangGraph."""

IPV_STATE_SHAPE = {
    "position": None,
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
