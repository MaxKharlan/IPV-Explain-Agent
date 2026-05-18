"""Актуальные API-контракты и adapter helpers."""

from __future__ import annotations

from typing import Any


POSITION_INPUT_EXAMPLE = {
    "position_id": "POS-SBER-CALL-001",
    "instrument_type": "option",
    "book": "EQ_VOL",
    "currency": "RUB",
    "quantity": 1000.0,
    "counterparty": "internal",
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


MARKET_SNAPSHOT_EXAMPLE = {
    "snapshot_id": "SNAP-2025-05-05",
    "snapshot_date": "2025-05-05",
    "source": "moex",
    "spot_prices": {"SBER": 307.10},
    "yield_curve": {
        "snapshot_date": "2025-05-05",
        "currency": "RUB",
        "points": [
            {"tenor": "1Y", "rate": 16.5},
            {"tenor": "2Y", "rate": 16.8},
            {"tenor": "5Y", "rate": 17.1},
        ],
        "source": "moex",
    },
    "option_quotes": {
        "snapshot_date": "2025-05-05",
        "underlier": "SBER",
        "points": [
            {
                "option_type": "call",
                "strike": 280.0,
                "expiry": "2026-09-20",
                "settlement_price": 12.4,
                "instrument_id": "SBERC280",
            },
            {
                "option_type": "put",
                "strike": 290.0,
                "expiry": "2026-09-20",
                "settlement_price": 8.1,
                "instrument_id": "SBERP290",
            },
        ],
        "source": "moex",
    },
    "quality_flags": {
        "used_mock_data": False,
        "missing_curve_points": False,
        "used_mock_option_quotes": False,
    },
}


PRICING_RESULT_MIN_EXAMPLE = {
    "position_id": "POS-SBER-CALL-001",
    "price_t0": 12.41,
    "price_t1": 13.86,
    "greeks_t0": {
        "delta": 0.53,
        "gamma": 0.012,
        "vega": 4.80,
        "theta": -0.15,
        "rho": 0.90,
    },
}


ATTRIBUTION_RESULT_EXAMPLE = {
    "position_id": "POS-SBER-CALL-001",
    "currency": "RUB",
    "price_t0": 12.41,
    "price_t1": 13.86,
    "total_pnl": 1.45,
    "components": {
        "delta_effect": 0.17,
        "gamma_effect": -0.01,
        "vega_effect": 1.54,
        "theta_effect": -0.13,
        "rho_effect": -0.07,
        "residual": 0.05,
    },
    "explained_pnl": 1.40,
    "explained_ratio": 0.9655,
    "residual_threshold_passed": True,
    "residual_threshold": 0.05,
    "notes": [],
}


NARRATIVE_OUTPUT_EXAMPLE = {
    "position_id": "POS-SBER-CALL-001",
    "summary": "Изменение стоимости составило 1.45 RUB. Главный фактор — вега-фактор размером 1.54 RUB. Второй по значимости фактор — тета-фактор размером -0.13 RUB.",
    "detailed_explanation": "Общее изменение стоимости позиции за период составило 1.45 RUB. Наибольший вклад внес фактор вега, отражающий чувствительность стоимости к изменению волатильности. Следующим по вкладу стал тета-фактор, связанный с временным распадом опциона. Остальные заметные факторы внесли ограниченный вклад. Остаток находится в допустимом диапазоне и подтверждает корректность разложения.",
    "top_drivers": [
        {"name": "vega_effect", "value": 1.54},
        {"name": "theta_effect", "value": -0.13},
    ],
    "residual_comment": "Остаток составляет 0.05 RUB и находится в допустимых пределах.",
    "validation_status": "passed",
    "fallback_used": False,
}


def _infer_model_name(instrument_type: str) -> str:
    """Возвращает presentation-friendly model name по instrument_type."""
    model_names = {
        "option": "black_scholes",
        "bond": "bond_discounting",
        "swap": "swap_npv",
    }
    return model_names.get(instrument_type, "unknown_model")


def build_public_pricing_result(
    position: dict[str, Any],
    pricing_result_min: dict[str, Any],
    market_t0: dict[str, Any] | None = None,
    market_t1: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Адаптирует внутренний PricingResultMin во внешний presentation-format."""
    instrument = position.get("instrument", {})
    underlier = instrument.get("underlier") if isinstance(instrument, dict) else None
    greeks_t0 = pricing_result_min.get("greeks_t0", {})

    model_inputs_summary: dict[str, Any] = {}
    if underlier and market_t0 and market_t1:
        model_inputs_summary["spot_t0"] = market_t0.get("spot_prices", {}).get(underlier)
        model_inputs_summary["spot_t1"] = market_t1.get("spot_prices", {}).get(underlier)
    if isinstance(instrument, dict):
        if "vol_t0" in instrument:
            model_inputs_summary["vol_t0"] = instrument["vol_t0"]
        if "vol_t1" in instrument:
            model_inputs_summary["vol_t1"] = instrument["vol_t1"]

    return {
        "position_id": pricing_result_min["position_id"],
        "instrument_type": position.get("instrument_type"),
        "price_t0": pricing_result_min["price_t0"],
        "price_t1": pricing_result_min["price_t1"],
        "currency": position.get("currency", "RUB"),
        "sensitivities": {
            "delta": greeks_t0.get("delta"),
            "gamma": greeks_t0.get("gamma"),
            "vega": greeks_t0.get("vega"),
            "theta": greeks_t0.get("theta"),
            "rho": greeks_t0.get("rho"),
        },
        "model_name": _infer_model_name(str(position.get("instrument_type", ""))),
        "model_inputs_summary": model_inputs_summary,
    }


def build_public_attribution_output(attribution_result: dict[str, Any]) -> dict[str, Any]:
    """Адаптирует внутренний AttributionResult во внешний presentation-format."""
    components = attribution_result.get("components", {})
    waterfall_components = [
        {"label": "Delta", "value": components.get("delta_effect", 0.0)},
        {"label": "Gamma", "value": components.get("gamma_effect", 0.0)},
        {"label": "Vega", "value": components.get("vega_effect", 0.0)},
        {"label": "Theta", "value": components.get("theta_effect", 0.0)},
        {"label": "Rho", "value": components.get("rho_effect", 0.0)},
        {"label": "Residual", "value": components.get("residual", 0.0)},
    ]
    return {
        "position_id": attribution_result["position_id"],
        "currency": attribution_result["currency"],
        "price_t0": attribution_result["price_t0"],
        "price_t1": attribution_result["price_t1"],
        "total_pnl": attribution_result["total_pnl"],
        "components": components,
        "waterfall_components": waterfall_components,
        "validation": {
            "residual_threshold_passed": attribution_result.get("residual_threshold_passed", False),
            "residual_threshold": attribution_result.get("residual_threshold"),
            "notes": attribution_result.get("notes", []),
        },
    }
