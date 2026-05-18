"""Smoke-tests for agent orchestration pipeline."""

import pytest

from src.agents.narrative_agent import (
    GigaChatClient,
    generate_gigachat_narrative,
    generate_template_narrative,
    run_narrative_agent,
)
from src.agents.orchestrator import run_pipeline, run_pipeline_until_attribution


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


def test_full_pipeline_builds_narrative_and_report(monkeypatch) -> None:
    """Полный pipeline должен дойти до report_result."""
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

    monkeypatch.setattr(
        "src.agents.market_data_agent.load_or_fetch_market_snapshots_for_period",
        lambda *args, **kwargs: (snapshot_t0, snapshot_t1),
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

    state = run_pipeline(position)

    assert state["narrative_result"]["position_id"] == position["position_id"]
    assert state["narrative_result"]["fallback_used"] is True
    assert state["report_result"]["report_summary"]["position_id"] == position["position_id"]
    assert "summary" in state["report_result"]["report_summary"]


def test_template_narrative_prioritizes_largest_drivers() -> None:
    """Template narrative должен выбирать крупнейшие драйверы по модулю."""
    state = {
        "position": {
            "position_id": "POS-001",
            "instrument_type": "option",
            "currency": "RUB",
        },
        "market_snapshot_t0": None,
        "market_snapshot_t1": None,
        "pricing_result": None,
        "attribution_result": {
            "position_id": "POS-001",
            "currency": "RUB",
            "total_pnl": 1.61,
            "components": {
                "delta_effect": 0.92,
                "gamma_effect": 0.13,
                "vega_effect": 0.71,
                "theta_effect": -0.09,
                "residual": -0.06,
            },
            "residual_threshold_passed": True,
        },
        "narrative_result": None,
        "report_result": None,
        "errors": [],
        "fallback_flags": {
            "used_mock_market_data": False,
            "used_template_narrative": False,
        },
    }

    narrative = generate_template_narrative(state)

    assert narrative["top_drivers"][0]["name"] == "delta_effect"
    assert narrative["top_drivers"][1]["name"] == "vega_effect"
    assert "delta" in narrative["summary"]
    assert "vega" in narrative["summary"]
    assert narrative["validation_status"] == "passed"


def test_template_narrative_rejects_inconsistent_position_id() -> None:
    """Narrative layer должен падать на битом attribution payload."""
    state = {
        "position": {
            "position_id": "POS-001",
            "instrument_type": "option",
            "currency": "RUB",
        },
        "market_snapshot_t0": None,
        "market_snapshot_t1": None,
        "pricing_result": None,
        "attribution_result": {
            "position_id": "POS-OTHER",
            "currency": "RUB",
            "total_pnl": 1.0,
            "components": {
                "delta_effect": 0.5,
                "gamma_effect": 0.1,
                "vega_effect": 0.2,
                "theta_effect": -0.1,
                "residual": 0.3,
            },
            "residual_threshold_passed": False,
        },
        "narrative_result": None,
        "report_result": None,
        "errors": [],
        "fallback_flags": {
            "used_mock_market_data": False,
            "used_template_narrative": False,
        },
    }

    with pytest.raises(ValueError, match="position_id mismatch"):
        generate_template_narrative(state)


def test_gigachat_narrative_uses_structured_output(monkeypatch) -> None:
    """GigaChat path должен принимать валидный structured JSON."""
    state = {
        "position": {
            "position_id": "POS-001",
            "instrument_type": "option",
            "currency": "RUB",
        },
        "market_snapshot_t0": None,
        "market_snapshot_t1": None,
        "pricing_result": None,
        "attribution_result": {
            "position_id": "POS-001",
            "currency": "RUB",
            "total_pnl": 1.61,
            "components": {
                "delta_effect": 0.92,
                "gamma_effect": 0.13,
                "vega_effect": 0.71,
                "theta_effect": -0.09,
                "residual": -0.06,
            },
            "residual_threshold_passed": True,
        },
        "narrative_result": None,
        "report_result": None,
        "errors": [],
        "fallback_flags": {
            "used_mock_market_data": False,
            "used_template_narrative": False,
        },
    }

    class FakeGigaChatClient(GigaChatClient):
        def is_configured(self) -> bool:
            return True

        def generate_narrative(self, prompt: str) -> dict[str, object]:
            assert "components=" in prompt
            return {
                "position_id": "POS-001",
                "summary": "Позиция изменилась в основном за счёт delta и vega.",
                "detailed_explanation": "Наибольший вклад дали delta и vega компоненты.",
                "top_drivers": [
                    {"name": "delta_effect", "value": 0.92},
                    {"name": "vega_effect", "value": 0.71},
                ],
                "residual_comment": "Residual находится в допустимом диапазоне.",
                "validation_status": "passed",
                "fallback_used": False,
            }

    narrative = generate_gigachat_narrative(state, client=FakeGigaChatClient())

    assert narrative["position_id"] == "POS-001"
    assert narrative["fallback_used"] is False
    assert narrative["validation_status"] == "passed"


def test_run_narrative_agent_falls_back_when_gigachat_fails(monkeypatch) -> None:
    """При ошибке GigaChat агент должен откатываться на template narrative."""
    state = {
        "position": {
            "position_id": "POS-001",
            "instrument_type": "option",
            "currency": "RUB",
        },
        "market_snapshot_t0": None,
        "market_snapshot_t1": None,
        "pricing_result": None,
        "attribution_result": {
            "position_id": "POS-001",
            "currency": "RUB",
            "total_pnl": 1.61,
            "components": {
                "delta_effect": 0.92,
                "gamma_effect": 0.13,
                "vega_effect": 0.71,
                "theta_effect": -0.09,
                "residual": -0.06,
            },
            "residual_threshold_passed": True,
        },
        "narrative_result": None,
        "report_result": None,
        "errors": [],
        "fallback_flags": {
            "used_mock_market_data": False,
            "used_template_narrative": False,
        },
    }

    class BrokenGigaChatClient(GigaChatClient):
        def is_configured(self) -> bool:
            return True

        def generate_narrative(self, prompt: str) -> dict[str, object]:
            _ = prompt
            raise RuntimeError("upstream unavailable")

    monkeypatch.setattr(
        "src.agents.narrative_agent.GigaChatClient",
        BrokenGigaChatClient,
    )

    result_state = run_narrative_agent(state)

    assert result_state["fallback_flags"]["used_template_narrative"] is True
    assert result_state["narrative_result"]["fallback_used"] is True
    assert result_state["errors"]
    assert "narrative_fallback" in result_state["errors"][0]
