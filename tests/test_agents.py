"""Smoke-tests for agent orchestration pipeline."""

from src.agents.orchestrator import run_pipeline_until_attribution


def test_pipeline_until_attribution_option_flow(monkeypatch) -> None:
    """Pipeline до attribution должен проходить на option position."""
    snapshot_t0 = {
        "snapshot_id": "SNAP-2026-05-01",
        "snapshot_date": "2026-05-01",
        "source": "mock",
        "spot_prices": {"SBER": 301.55},
        "yield_curve": {
            "snapshot_date": "2026-05-01",
            "currency": "RUB",
            "points": [{"tenor": "1Y", "rate": 16.5}],
            "source": "mock",
        },
        "option_quotes": {
            "snapshot_date": "2026-05-01",
            "underlier": "SBER",
            "points": [],
            "source": "mock",
        },
        "quality_flags": {
            "used_mock_data": False,
            "missing_curve_points": False,
            "used_mock_option_quotes": False,
        },
    }
    snapshot_t1 = {
        "snapshot_id": "SNAP-2026-05-02",
        "snapshot_date": "2026-05-02",
        "source": "mock",
        "spot_prices": {"SBER": 307.10},
        "yield_curve": {
            "snapshot_date": "2026-05-02",
            "currency": "RUB",
            "points": [{"tenor": "1Y", "rate": 16.8}],
            "source": "mock",
        },
        "option_quotes": {
            "snapshot_date": "2026-05-02",
            "underlier": "SBER",
            "points": [],
            "source": "mock",
        },
        "quality_flags": {
            "used_mock_data": False,
            "missing_curve_points": False,
            "used_mock_option_quotes": False,
        },
    }

    def _mock_load_snapshots(*args, **kwargs):
        return snapshot_t0, snapshot_t1

    monkeypatch.setattr(
        "src.agents.market_data_agent.load_or_fetch_market_snapshots_for_period",
        _mock_load_snapshots,
    )

    position = {
        "position_id": "POS-SBER-CALL-001",
        "instrument_type": "option",
        "book": "EQ_VOL",
        "currency": "RUB",
        "quantity": 1000.0,
        "as_of_dates": {"t0": "2026-05-01", "t1": "2026-05-02"},
        "instrument": {
            "underlier": "SBER",
            "option_type": "call",
            "strike": 280.0,
            "maturity_date": "2026-09-20",
            "vol_t0": 0.24,
            "vol_t1": 0.27,
        },
    }

    state = run_pipeline_until_attribution(position)

    assert state["market_snapshot_t0"]["spot_prices"]["SBER"] == 301.55
    assert state["pricing_result"]["position_id"] == position["position_id"]
    assert state["attribution_result"]["position_id"] == position["position_id"]
    assert "components" in state["attribution_result"]
