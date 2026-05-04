"""Примеры общих API-контрактов."""

POSITION_INPUT_EXAMPLE = {
    "position_id": "POS-001",
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
    },
}

MARKET_SNAPSHOT_EXAMPLE = {
    "snapshot_id": "SNAP-2026-05-01",
    "snapshot_date": "2026-05-01",
    "source": "moex",
    "spot_prices": {"SBER": 301.55},
    "yield_curve": {
        "currency": "RUB",
        "points": [{"tenor": "1M", "rate": 0.165}],
    },
    "vol_surface": {
        "underlier": "SBER",
        "points": [{"tenor": "1M", "strike": 280.0, "implied_vol": 0.24}],
    },
    "quality_flags": {
        "used_mock_data": False,
        "missing_curve_points": False,
        "surface_interpolated": True,
    },
}

ATTRIBUTION_OUTPUT_EXAMPLE = {
    "position_id": "POS-001",
    "currency": "RUB",
    "price_t0": 12.41,
    "price_t1": 14.02,
    "total_pnl": 1.61,
    "components": {
        "delta_effect": 0.92,
        "gamma_effect": 0.13,
        "vega_effect": 0.71,
        "theta_effect": -0.09,
        "residual": -0.06,
    },
    "stress_results": [{"scenario_name": "parallel_shift_1bp", "pnl": -0.01}],
    "waterfall_components": [{"label": "Delta", "value": 0.92}],
    "validation": {"residual_threshold_passed": True, "notes": []},
}

NARRATIVE_OUTPUT_EXAMPLE = {
    "position_id": "POS-001",
    "summary": "Рост справедливой стоимости связан с движением спота и implied volatility.",
    "detailed_explanation": "Основной вклад внесли delta и vega компоненты.",
    "top_drivers": [
        {"name": "delta_effect", "value": 0.92},
        {"name": "vega_effect", "value": 0.71},
    ],
    "residual_comment": "Residual находится в допустимом диапазоне.",
    "validation_status": "passed",
    "fallback_used": False,
}
