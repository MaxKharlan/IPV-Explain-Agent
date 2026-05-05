"""End-to-end интеграционный тест Attribution Engine.

Проходит весь pipeline на реалистичных рыночных данных:
    Position → Pricing → Attribution → Stress → Backtest.

Цель: убедиться, что три модуля корректно стыкуются и выдают
осмысленные числа на типовом IPV-кейсе. Это smoke-тест поверх
unit-тестов из test_attribution.py.
"""

from __future__ import annotations

from datetime import date

import pytest

from src.attribution import (
    BacktestCase,
    CurveSensitivities,
    Greeks,
    Position,
    PricingResultMin,
    RiskFactorSnapshot,
    backtest_attribution,
    parallel_shift,
    run_attribution,
    run_stress_scenarios,
    standard_scenario_set,
    twist,
)


# ---------------------------------------------------------------------------
# Реалистичный кейс: SBER 280 Call (по примеру из contracts.md)
# ---------------------------------------------------------------------------


def _sber_call_position() -> Position:
    return Position(
        position_id="POS-SBER-CALL-001",
        instrument_type="option",
        quantity=1000.0,
        currency="RUB",
        underlier="SBER",
        book="EQ_VOL",
    )


def _sber_call_pricing(price_t0: float, price_t1: float) -> PricingResultMin:
    """Греки — raw, BS-derived для SBER call со страйком 280, T≈4.5 мес."""
    return PricingResultMin(
        position_id="POS-SBER-CALL-001",
        price_t0=price_t0,
        price_t1=price_t1,
        greeks_t0=Greeks(
            delta=0.53,
            gamma=0.012,
            vega=120.0,    # ∂V/∂σ raw, per 1.0 σ unit
            theta=-15.0,   # ∂V/∂t per year, для long call < 0
        ),
    )


# ---------------------------------------------------------------------------
# E2E: Attribution + Stress
# ---------------------------------------------------------------------------


def test_e2e_sber_call_attribution_then_stress() -> None:
    """От снапшотов через attribution к стресс-сценариям — единый flow."""
    position = _sber_call_position()
    rf_t0 = RiskFactorSnapshot(
        snapshot_date=date(2026, 5, 1), spot=301.55, vol=0.24,
    )
    rf_t1 = RiskFactorSnapshot(
        snapshot_date=date(2026, 5, 2), spot=307.10, vol=0.27,
    )

    # ---------- Attribution ----------
    # Ожидаемые компоненты:
    #   delta_effect:  0.53 × 5.55       = 2.9415
    #   gamma_effect:  0.5 × 0.012 × 5.55² = 0.18486
    #   vega_effect:   120  × 0.03       = 3.6
    #   theta_effect:  -15  × (1/365)    ≈ -0.04110
    # Итого explained ≈ 6.685
    explained_target = (
        0.53 * 5.55
        + 0.5 * 0.012 * 5.55 ** 2
        + 120.0 * 0.03
        + (-15.0) * (1.0 / 365.0)
    )
    # Зададим actual PnL близко к explained (residual = 0.05, ~0.7% от total)
    actual_pnl = explained_target + 0.05
    pricing = _sber_call_pricing(price_t0=12.41, price_t1=12.41 + actual_pnl)

    attribution = run_attribution(position, pricing, rf_t0, rf_t1)

    # Проверка компонент
    assert attribution.components.delta_effect == pytest.approx(2.9415, rel=1e-3)
    assert attribution.components.gamma_effect == pytest.approx(0.18486, rel=1e-3)
    assert attribution.components.vega_effect == pytest.approx(3.6, rel=1e-3)
    assert attribution.components.theta_effect == pytest.approx(-15.0 / 365.0, rel=1e-3)

    # Residual внутри threshold
    assert abs(attribution.components.residual) < 0.1
    assert attribution.residual_threshold_passed is True


    # Sanity по иерархии: при vol shock +3pp / spot move +1.8% именно vega
    # доминирует над delta. Это типично для дней значительной vol-волатильности
    # и хорошая иллюстрация regime-dependent attribution.
    # Ожидаемая иерархия: vega > delta > gamma > theta
    assert (
            abs(attribution.components.vega_effect)
            > abs(attribution.components.delta_effect)
    )
    assert (
            abs(attribution.components.delta_effect)
            > abs(attribution.components.gamma_effect)
    )
    assert (
            abs(attribution.components.gamma_effect)
            > abs(attribution.components.theta_effect)
    )

    # ---------- Stress: без curve_sens (rho=None) ----------
    # Все стрессы должны skipped — у опциона rho не задан
    stress_no_curve = run_stress_scenarios(
        position, pricing.greeks_t0,
        base_price=pricing.price_t0,
        scenarios=standard_scenario_set(),
    )
    assert len(stress_no_curve.results) == 10
    assert all(r.method == "skipped" for r in stress_no_curve.results)
    assert stress_no_curve.pnl_dict() == {}

    # ---------- Stress: с PV01 curve sensitivities ----------
    # Допустим у нас оценены rate sensitivities по тенрам
    sens = CurveSensitivities(
        pv01_by_tenor={
            "1M": -0.5, "3M": -1.0, "6M": -2.0, "1Y": -4.0,
            "2Y": -7.0, "5Y": -15.0, "10Y": -25.0, "20Y": -40.0,
        }
    )
    stress = run_stress_scenarios(
        position, pricing.greeks_t0,
        base_price=pricing.price_t0,
        scenarios=standard_scenario_set(),
        curve_sensitivities=sens,
    )

    assert all(r.method == "curve_pv01" for r in stress.results)
    # Симметрия first-order: ±10bp дают противоположные PnL
    pnl_dict = stress.pnl_dict()
    assert pnl_dict["parallel_shift_+10bp"] == pytest.approx(
        -pnl_dict["parallel_shift_-10bp"], rel=1e-10,
    )
    # Знаки: при отрицательных pv01 (длинная позиция в облигации-like rate exposure)
    # параллельный +10bp сдвиг должен дать отрицательный PnL
    assert pnl_dict["parallel_shift_+10bp"] < 0


# ---------------------------------------------------------------------------
# E2E: Multi-book backtest
# ---------------------------------------------------------------------------


def test_e2e_multi_book_backtest_with_realistic_distribution() -> None:
    """Бэктест на смеси опционов и бондов из 2 books — проверяем срезы."""
    cases: list[BacktestCase] = []

    # 5 опционов из EQ_VOL, все с perfect attribution
    for i in range(5):
        cases.append(BacktestCase(
            position=Position(
                position_id=f"OPT-{i}", instrument_type="option",
                quantity=100.0, currency="RUB", book="EQ_VOL",
            ),
            pricing=PricingResultMin(
                position_id=f"OPT-{i}",
                price_t0=10.0, price_t1=12.0,
                greeks_t0=Greeks(delta=1.0, gamma=0.0, vega=0.0, theta=0.0),
            ),
            rf_t0=RiskFactorSnapshot(snapshot_date=date(2026, 5, 1), spot=100.0, vol=0.20),
            rf_t1=RiskFactorSnapshot(snapshot_date=date(2026, 5, 1), spot=102.0, vol=0.20),
        ))

    # 3 облигации из RATES: 2 perfect, 1 с big residual
    bond_specs = [(1.0, 11.0), (1.0, 11.0), (0.3, 11.0)]  # последняя с bad delta
    for i, (delta, p1) in enumerate(bond_specs):
        cases.append(BacktestCase(
            position=Position(
                position_id=f"BND-{i}", instrument_type="bond",
                quantity=1000.0, currency="RUB", book="RATES",
            ),
            pricing=PricingResultMin(
                position_id=f"BND-{i}",
                price_t0=10.0, price_t1=p1,
                greeks_t0=Greeks(delta=delta, gamma=0.0, vega=0.0, theta=0.0),
            ),
            rf_t0=RiskFactorSnapshot(snapshot_date=date(2026, 5, 1), spot=100.0, vol=0.0),
            rf_t1=RiskFactorSnapshot(snapshot_date=date(2026, 5, 1), spot=101.0, vol=0.0),
        ))

    report = backtest_attribution(cases)

    # Overall: 7/8 passed (5 опционов + 2 хороших бонда), 1 breached
    assert report.overall.n_total == 8
    assert report.overall.n_passed == 7
    assert report.overall.n_breached == 1
    assert report.overall.pass_rate == pytest.approx(7 / 8)

    # By instrument: options 100%, bonds 2/3
    assert report.by_instrument_type["option"].pass_rate == 1.0
    assert report.by_instrument_type["bond"].pass_rate == pytest.approx(2 / 3)

    # By book — то же самое в этом примере
    assert report.by_book["EQ_VOL"].pass_rate == 1.0
    assert report.by_book["RATES"].pass_rate == pytest.approx(2 / 3)

    # Проверка breached_positions
    assert len(report.breached_positions) == 1
    assert report.breached_positions[0].position_id == "BND-2"
    assert report.breached_positions[0].instrument_type == "bond"

    # Никаких error'ов от валидации
    assert len(report.errors) == 0


# ---------------------------------------------------------------------------
# E2E: Attribution → Backtest → Report (конец конца)
# ---------------------------------------------------------------------------


def test_e2e_full_pipeline_smoke() -> None:
    """Полный flow — каждый этап выдаёт next stage'у валидные данные."""
    # Только один кейс, минимальный — проверяем сам факт стыковки
    position = _sber_call_position()
    rf_t0 = RiskFactorSnapshot(snapshot_date=date(2026, 5, 1), spot=301.55, vol=0.24)
    rf_t1 = RiskFactorSnapshot(snapshot_date=date(2026, 5, 2), spot=307.10, vol=0.27)
    pricing = _sber_call_pricing(price_t0=12.41, price_t1=12.41 + 6.685)  # ≈ explained

    # Attribution
    attribution = run_attribution(position, pricing, rf_t0, rf_t1)
    assert attribution.position_id == position.position_id
    assert attribution.currency == position.currency

    # Stress (через PV01)
    sens = CurveSensitivities(pv01_by_tenor={"1Y": -2.0, "10Y": -20.0})
    stress = run_stress_scenarios(
        position, pricing.greeks_t0,
        base_price=pricing.price_t0,
        scenarios=[parallel_shift(10), twist(25, tenors=("1Y", "10Y"))],
        curve_sensitivities=sens,
    )
    assert len(stress.results) == 2
    assert all(r.method == "curve_pv01" for r in stress.results)

    # Backtest (на этом одном кейсе)
    backtest_report = backtest_attribution([
        BacktestCase(position=position, pricing=pricing, rf_t0=rf_t0, rf_t1=rf_t1)
    ])
    assert backtest_report.overall.n_total == 1
    assert backtest_report.overall.pass_rate == 1.0

    # Финальная sanity: все артефакты сериализуются в JSON (важно для API/Report Agent)
    _ = attribution.model_dump_json()
    _ = stress.model_dump_json()
    _ = backtest_report.model_dump_json()