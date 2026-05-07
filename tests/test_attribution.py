"""Unit-тесты для src/attribution/taylor_decomp.py.

Покрытие:
    - чистые однофакторные сценарии (delta-only, gamma-only, vega-only, theta-only)
    - синтетический полиномиальный кейс с residual = 0 (точная Taylor-формула)
    - rho-вклад при наличии данных и graceful skip при partial data
    - edge cases: нулевые сдвиги, near-zero total_pnl, инверсия дат
    - валидация: position_id mismatch, выход за residual threshold
    - smoke-тест на «реалистичной» SBER-call позиции
"""

from __future__ import annotations

from datetime import date

import pytest

from src.attribution.schemas import (
    Greeks,
    Position,
    PricingResultMin,
    RiskFactorSnapshot,
    BacktestCase,
    BacktestReport,
)
from src.attribution.backtesting import backtest_attribution
from src.attribution.taylor_decomp import (
    DEFAULT_RESIDUAL_THRESHOLD,
    run_attribution,
)

# ----------------------------- Хелперы фикстур -----------------------------


def _make_position(pid: str = "POS-001") -> Position:
    return Position(
        position_id=pid,
        instrument_type="option",
        quantity=1000.0,
        currency="RUB",
        underlier="SBER",
    )


def _make_snapshots(
    spot_t0: float = 100.0,
    spot_t1: float = 100.0,
    vol_t0: float = 0.20,
    vol_t1: float = 0.20,
    days_between: int = 1,
    rate_t0: float | None = None,
    rate_t1: float | None = None,
) -> tuple[RiskFactorSnapshot, RiskFactorSnapshot]:
    rf0 = RiskFactorSnapshot(
        snapshot_date=date(2026, 5, 1),
        spot=spot_t0,
        vol=vol_t0,
        rate=rate_t0,
    )
    rf1 = RiskFactorSnapshot(
        snapshot_date=date(2026, 5, 1 + days_between) if days_between < 30 else date(2026, 6, 1),
        spot=spot_t1,
        vol=vol_t1,
        rate=rate_t1,
    )
    return rf0, rf1


def _make_pricing(
    pid: str = "POS-001",
    price_t0: float = 10.0,
    price_t1: float = 10.0,
    greeks: Greeks | None = None,
) -> PricingResultMin:
    if greeks is None:
        greeks = Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0)
    return PricingResultMin(
        position_id=pid,
        price_t0=price_t0,
        price_t1=price_t1,
        greeks_t0=greeks,
    )


# ----------------------------- Однофакторные тесты -----------------------------


def test_pure_delta_attribution() -> None:
    """Только спот меняется, греки кроме delta = 0. delta_effect = Δ·ΔS."""
    rf0, rf1 = _make_snapshots(spot_t0=100.0, spot_t1=105.0, days_between=0)
    greeks = Greeks(delta=0.5, gamma=0.0, vega=0.0, theta=0.0)
    pricing = _make_pricing(price_t0=10.0, price_t1=12.5, greeks=greeks)  # ΔPnL = 2.5

    result = run_attribution(_make_position(), pricing, rf0, rf1)

    assert result.components.delta_effect == pytest.approx(0.5 * 5.0)
    assert result.components.gamma_effect == 0.0
    assert result.components.vega_effect == 0.0
    assert result.components.theta_effect == 0.0
    assert result.components.residual == pytest.approx(0.0, abs=1e-12)
    assert result.residual_threshold_passed is True


def test_pure_gamma_synthetic_quadratic() -> None:
    """Pricing-функция P = b·S². Тогда Δ = 2bS, Γ = 2b. Taylor должен дать
    точное разложение: ΔP = 2bS·ΔS + b·ΔS² (без residual)."""
    s0, s1, b = 100.0, 105.0, 0.01
    p0, p1 = b * s0**2, b * s1**2  # 100.0 vs 110.25 → ΔPnL = 10.25
    greeks = Greeks(delta=2 * b * s0, gamma=2 * b, vega=0.0, theta=0.0)
    rf0, rf1 = _make_snapshots(spot_t0=s0, spot_t1=s1, days_between=0)
    pricing = _make_pricing(price_t0=p0, price_t1=p1, greeks=greeks)

    result = run_attribution(_make_position(), pricing, rf0, rf1)

    # Точная декомпозиция: ΔPnL = 2bS·ΔS + b·ΔS²
    expected_delta = 2 * b * s0 * (s1 - s0)
    expected_gamma = 0.5 * 2 * b * (s1 - s0) ** 2  # = b·ΔS²
    assert result.components.delta_effect == pytest.approx(expected_delta)
    assert result.components.gamma_effect == pytest.approx(expected_gamma)
    assert result.components.residual == pytest.approx(0.0, abs=1e-10)


def test_pure_vega_attribution() -> None:
    """Только vol меняется, vega ≠ 0. vega_effect = ν·Δσ."""
    rf0, rf1 = _make_snapshots(vol_t0=0.20, vol_t1=0.25, days_between=0)
    greeks = Greeks(delta=0.0, gamma=0.0, vega=8.0, theta=0.0)
    pricing = _make_pricing(price_t0=10.0, price_t1=10.4, greeks=greeks)  # ΔPnL = 0.4

    result = run_attribution(_make_position(), pricing, rf0, rf1)

    assert result.components.vega_effect == pytest.approx(8.0 * 0.05)
    assert result.components.residual == pytest.approx(0.0, abs=1e-12)


def test_pure_theta_attribution() -> None:
    """Только время идёт, theta ≠ 0. theta_effect = θ·Δt (calendar, ACT/365)."""
    rf0, rf1 = _make_snapshots(days_between=10)  # 10/365 года
    greeks = Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=-3.65)  # «теряем» 0.1/день
    expected_theta_effect = -3.65 * (10.0 / 365.0)
    pricing = _make_pricing(price_t0=10.0, price_t1=10.0 + expected_theta_effect, greeks=greeks)

    result = run_attribution(_make_position(), pricing, rf0, rf1)

    assert result.components.theta_effect == pytest.approx(expected_theta_effect)
    assert result.components.residual == pytest.approx(0.0, abs=1e-12)


# --------------------------- Комбинированные тесты ---------------------------


def test_combined_synthetic_polynomial_zero_residual() -> None:
    """P(S, σ, t) = a·S + b·S² + c·σ + d·t — точное Taylor-разложение → residual = 0."""
    a, b, c, d = 0.4, 0.005, 5.0, -2.0
    s0, s1 = 100.0, 103.0
    sigma0, sigma1 = 0.20, 0.22
    t_days = 5
    dt = t_days / 365.0

    p0 = a * s0 + b * s0**2 + c * sigma0
    p1 = a * s1 + b * s1**2 + c * sigma1 + d * dt
    # Производные на t0:
    greeks = Greeks(
        delta=a + 2 * b * s0,
        gamma=2 * b,
        vega=c,
        theta=d,
    )
    rf0, rf1 = _make_snapshots(
        spot_t0=s0, spot_t1=s1, vol_t0=sigma0, vol_t1=sigma1, days_between=t_days
    )
    pricing = _make_pricing(price_t0=p0, price_t1=p1, greeks=greeks)

    result = run_attribution(_make_position(), pricing, rf0, rf1)

    assert result.components.residual == pytest.approx(0.0, abs=1e-10)
    assert result.residual_threshold_passed is True
    assert result.explained_ratio == pytest.approx(1.0, abs=1e-6)


# --------------------------------- Rho-тесты ---------------------------------


def test_rho_contribution_when_provided() -> None:
    """При наличии rho и обеих ставок rho_effect = ρ·Δr."""
    rf0, rf1 = _make_snapshots(rate_t0=0.16, rate_t1=0.17, days_between=0)
    greeks = Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=12.0)
    pricing = _make_pricing(price_t0=10.0, price_t1=10.12, greeks=greeks)  # ΔPnL = 0.12

    result = run_attribution(_make_position(), pricing, rf0, rf1)

    assert result.components.rho_effect == pytest.approx(12.0 * 0.01)
    assert result.components.residual == pytest.approx(0.0, abs=1e-12)


def test_rho_partial_data_adds_note() -> None:
    """Если rho задан, но ставка только в одном снапшоте — rho_effect = 0 + note."""
    rf0, rf1 = _make_snapshots(rate_t0=0.16, rate_t1=None, days_between=0)
    greeks = Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=12.0)
    pricing = _make_pricing(price_t0=10.0, price_t1=10.0, greeks=greeks)

    result = run_attribution(_make_position(), pricing, rf0, rf1)

    assert result.components.rho_effect == 0.0
    assert any("rho" in n.lower() or "rate" in n.lower() for n in result.notes)


# ------------------------------- Edge cases -------------------------------


def test_zero_changes_all_components_zero() -> None:
    """Никакие риск-факторы не меняются → все эффекты 0, total_pnl = 0."""
    rf0, rf1 = _make_snapshots(days_between=0)  # та же дата, тот же spot/vol
    greeks = Greeks(delta=0.5, gamma=0.01, vega=5.0, theta=-2.0)
    pricing = _make_pricing(price_t0=10.0, price_t1=10.0, greeks=greeks)

    result = run_attribution(_make_position(), pricing, rf0, rf1)

    assert result.total_pnl == pytest.approx(0.0)
    assert result.components.delta_effect == 0.0
    assert result.components.gamma_effect == 0.0
    assert result.components.vega_effect == 0.0
    assert result.components.theta_effect == 0.0
    assert result.components.residual == pytest.approx(0.0)
    assert result.residual_threshold_passed is True
    assert result.explained_ratio == 1.0


def test_near_zero_total_pnl_uses_absolute_test() -> None:
    """При |ΔPnL| ≈ 0 и совпадающем explained — passed по абсолютному тесту."""
    rf0, rf1 = _make_snapshots(days_between=0)
    greeks = Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0)
    pricing = _make_pricing(price_t0=10.0, price_t1=10.0, greeks=greeks)

    result = run_attribution(_make_position(), pricing, rf0, rf1)

    assert result.residual_threshold_passed is True
    assert any("zero" in n.lower() or "absolute" in n.lower() for n in result.notes)


# -------------------------- Валидация и ошибки --------------------------


def test_inverted_dates_raises() -> None:
    """t1 < t0 → ValueError."""
    rf0 = RiskFactorSnapshot(snapshot_date=date(2026, 5, 10), spot=100.0, vol=0.20)
    rf1 = RiskFactorSnapshot(snapshot_date=date(2026, 5, 1), spot=100.0, vol=0.20)
    pricing = _make_pricing()

    with pytest.raises(ValueError, match="Inverted snapshot order"):
        run_attribution(_make_position(), pricing, rf0, rf1)


def test_position_id_mismatch_raises() -> None:
    """position_id у Position и Pricing не совпадают → ValueError."""
    rf0, rf1 = _make_snapshots(days_between=1)
    pricing = _make_pricing(pid="POS-OTHER")

    with pytest.raises(ValueError, match="position_id mismatch"):
        run_attribution(_make_position(pid="POS-001"), pricing, rf0, rf1)


def test_invalid_residual_threshold_raises() -> None:
    """Threshold вне (0, 1) → ValueError."""
    rf0, rf1 = _make_snapshots(days_between=1)
    pricing = _make_pricing()

    with pytest.raises(ValueError, match="residual_threshold"):
        run_attribution(_make_position(), pricing, rf0, rf1, residual_threshold=1.5)


def test_residual_threshold_breach_flag() -> None:
    """Греки врут на 50% → residual >> threshold, флаг = False."""
    rf0, rf1 = _make_snapshots(spot_t0=100.0, spot_t1=110.0, days_between=0)
    # «правильный» delta был бы ~1.0, но Pricing Agent ошибочно прислал 0.5
    greeks = Greeks(delta=0.5, gamma=0.0, vega=0.0, theta=0.0)
    pricing = _make_pricing(price_t0=10.0, price_t1=20.0, greeks=greeks)
    # ΔPnL = 10, explained = 0.5 * 10 = 5, residual = 5, ratio = 50% > 5%

    result = run_attribution(_make_position(), pricing, rf0, rf1)

    assert result.components.residual == pytest.approx(5.0)
    assert result.residual_threshold_passed is False


def test_custom_threshold_overrides_default() -> None:
    """Custom threshold = 60% делает breach из предыдущего теста проходящим."""
    rf0, rf1 = _make_snapshots(spot_t0=100.0, spot_t1=110.0, days_between=0)
    greeks = Greeks(delta=0.5, gamma=0.0, vega=0.0, theta=0.0)
    pricing = _make_pricing(price_t0=10.0, price_t1=20.0, greeks=greeks)

    result = run_attribution(_make_position(), pricing, rf0, rf1, residual_threshold=0.60)

    assert result.residual_threshold_passed is True
    assert result.residual_threshold == 0.60


# -------------------------------- Smoke -------------------------------------


def test_realistic_sber_call_smoke() -> None:
    """Реалистичные значения SBER 280 call: small move, residual в пределах 5%."""
    rf0, rf1 = _make_snapshots(
        spot_t0=301.55, spot_t1=307.10,
        vol_t0=0.24, vol_t1=0.27,
        days_between=1,
    )
    greeks = Greeks(delta=0.53, gamma=0.012, vega=120.0, theta=-15.0)  # raw greeks
    # explained = 0.53*5.55 + 0.5*0.012*5.55² + 120*0.03 + (-15)*(1/365)
    #           = 2.94 + 0.185 + 3.60 + (-0.041) ≈ 6.68
    # ΔPnL ставим близкий, чтобы residual был в пределах 5%
    pricing = _make_pricing(price_t0=12.41, price_t1=12.41 + 6.65, greeks=greeks)

    result = run_attribution(_make_position(), pricing, rf0, rf1)

    assert result.residual_threshold_passed is True
    assert result.explained_ratio > 0.9
    # Sanity: delta_effect доминирует для такого сдвига спота
    assert abs(result.components.delta_effect) > abs(result.components.theta_effect)


def test_attribution_result_is_immutable() -> None:
    """Pydantic frozen=True — попытка мутации должна падать."""
    rf0, rf1 = _make_snapshots(days_between=1)
    pricing = _make_pricing()
    result = run_attribution(_make_position(), pricing, rf0, rf1)

    with pytest.raises((TypeError, ValueError, Exception)):  # ValidationError в pydantic v2
        result.total_pnl = 999.0  # type: ignore[misc]


# В блок импортов добавить:
from src.attribution.schemas import (
    CurveSensitivities,
    CurveShiftScenario,
    StressReport,
)
from src.attribution.stress_tests import (
    BP,
    DEFAULT_TENORS,
    butterfly,
    parallel_shift,
    run_stress_scenarios,
    standard_scenario_set,
    twist,
)


# ===========================================================================
#                              STRESS TESTS
# ===========================================================================


# --------------------------- Конструкторы сценариев ---------------------------


def test_parallel_shift_construction() -> None:
    """parallel_shift(10bp) на 3 тенра → у всех одинаковый shift = 0.0010."""
    scen = parallel_shift(10, tenors=("1M", "1Y", "10Y"))
    assert scen.name == "parallel_shift_+10bp"
    assert scen.shifts_by_tenor == {"1M": 10 * BP, "1Y": 10 * BP, "10Y": 10 * BP}
    assert scen.is_parallel() is True
    assert scen.parallel_size() == pytest.approx(0.0010)


def test_parallel_shift_negative() -> None:
    scen = parallel_shift(-100, tenors=("1Y", "10Y"))
    assert scen.name == "parallel_shift_-100bp"
    assert all(v == pytest.approx(-0.01) for v in scen.shifts_by_tenor.values())


def test_parallel_shift_empty_tenors_raises() -> None:
    with pytest.raises(ValueError, match="tenors must not be empty"):
        parallel_shift(10, tenors=())


def test_twist_construction_linear_weights() -> None:
    """twist(+25bp) на [1Y, 5Y, 10Y]: shifts = [-25bp, 0, +25bp]."""
    scen = twist(25, tenors=("1Y", "5Y", "10Y"))
    assert scen.name == "twist_steepener_25bp"
    assert scen.shifts_by_tenor["1Y"] == pytest.approx(-25 * BP)
    assert scen.shifts_by_tenor["5Y"] == pytest.approx(0.0)
    assert scen.shifts_by_tenor["10Y"] == pytest.approx(+25 * BP)
    assert scen.is_parallel() is False


def test_twist_flattener_negative_slope() -> None:
    scen = twist(-25, tenors=("1Y", "10Y"))
    assert "flattener" in scen.name
    assert scen.shifts_by_tenor["1Y"] == pytest.approx(+25 * BP)
    assert scen.shifts_by_tenor["10Y"] == pytest.approx(-25 * BP)


def test_twist_too_few_tenors_raises() -> None:
    with pytest.raises(ValueError, match=">= 2 tenors"):
        twist(25, tenors=("1Y",))


def test_butterfly_construction() -> None:
    """butterfly(+25bp) на [1Y, 5Y, 10Y]: wings +25bp, mid -50bp."""
    scen = butterfly(25, tenors=("1Y", "5Y", "10Y"))
    assert "wings_up" in scen.name
    assert scen.shifts_by_tenor["1Y"] == pytest.approx(+25 * BP)
    assert scen.shifts_by_tenor["5Y"] == pytest.approx(-50 * BP)
    assert scen.shifts_by_tenor["10Y"] == pytest.approx(+25 * BP)


def test_butterfly_too_few_tenors_raises() -> None:
    with pytest.raises(ValueError, match=">= 3 tenors"):
        butterfly(25, tenors=("1Y", "10Y"))


def test_standard_scenario_set_completeness() -> None:
    """Стандартный набор содержит ровно 10 сценариев из заданного списка."""
    scenarios = standard_scenario_set()
    names = {s.name for s in scenarios}
    assert len(scenarios) == 10
    # 6 параллельных
    for bps in (-100, -10, -1, 1, 10, 100):
        sign = "+" if bps >= 0 else "-"
        assert f"parallel_shift_{sign}{abs(bps)}bp" in names
    # 2 twist
    assert "twist_steepener_25bp" in names
    assert "twist_flattener_25bp" in names
    # 2 butterfly
    assert "butterfly_wings_up_25bp" in names
    assert "butterfly_wings_down_25bp" in names


# --------------------------- run_stress_scenarios ---------------------------


def _stress_position() -> Position:
    return Position(
        position_id="POS-BOND-001",
        instrument_type="bond",
        quantity=1_000_000.0,
        currency="RUB",
    )


def test_stress_parallel_via_rho() -> None:
    """Скалярный rho + параллельный сдвиг → ΔPnL = rho × shift."""
    pos = _stress_position()
    greeks = Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=-700.0)  # bond-like
    scen = parallel_shift(10, tenors=("1Y", "10Y"))

    report = run_stress_scenarios(pos, greeks, base_price=100.0, scenarios=[scen])

    assert isinstance(report, StressReport)
    result = report.results[0]
    assert result.method == "scalar_rho"
    # rho × shift = -700 × 0.001 = -0.7
    assert result.pnl == pytest.approx(-700.0 * 0.0010)


def test_stress_full_curve_via_pv01() -> None:
    """CurveSensitivities + twist → корректное скалярное произведение."""
    pos = _stress_position()
    greeks = Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=-100.0)
    sens = CurveSensitivities(
        pv01_by_tenor={"1Y": -50.0, "5Y": -250.0, "10Y": -700.0}
    )
    scen = twist(25, tenors=("1Y", "5Y", "10Y"))
    # twist: 1Y -25bp, 5Y 0, 10Y +25bp
    # PnL = (-50)·(-0.0025) + (-250)·0 + (-700)·(0.0025)
    #     = 0.125 + 0 - 1.75 = -1.625
    expected = (-50.0) * (-25 * BP) + (-700.0) * (25 * BP)

    report = run_stress_scenarios(
        pos, greeks, base_price=100.0,
        scenarios=[scen], curve_sensitivities=sens,
    )

    result = report.results[0]
    assert result.method == "curve_pv01"
    assert result.pnl == pytest.approx(expected)


def test_stress_pv01_takes_priority_over_rho() -> None:
    """Если задан и rho, и PV01 — используется PV01 даже для параллельного сдвига."""
    pos = _stress_position()
    greeks = Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=-1000.0)
    sens = CurveSensitivities(pv01_by_tenor={"1Y": -100.0, "10Y": -500.0})
    scen = parallel_shift(10, tenors=("1Y", "10Y"))

    report = run_stress_scenarios(
        pos, greeks, base_price=100.0,
        scenarios=[scen], curve_sensitivities=sens,
    )

    result = report.results[0]
    assert result.method == "curve_pv01"
    expected = (-100.0 + -500.0) * (10 * BP)  # = -0.6
    assert result.pnl == pytest.approx(expected)


def test_stress_skips_non_parallel_when_only_rho() -> None:
    """Twist без CurveSensitivities + только rho → skipped с note."""
    pos = _stress_position()
    greeks = Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=-100.0)
    scen = twist(25, tenors=("1Y", "10Y"))

    report = run_stress_scenarios(pos, greeks, base_price=100.0, scenarios=[scen])

    result = report.results[0]
    assert result.method == "skipped"
    assert result.pnl is None
    assert any("non-parallel" in n.lower() for n in result.notes)
    assert scen.name not in report.pnl_dict()


def test_stress_skips_parallel_when_no_rho() -> None:
    """Параллельный сдвиг без rho и без PV01 → skipped."""
    pos = _stress_position()
    greeks = Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=None)
    scen = parallel_shift(10)

    report = run_stress_scenarios(pos, greeks, base_price=100.0, scenarios=[scen])

    assert report.results[0].method == "skipped"
    assert report.results[0].pnl is None


def test_stress_partial_pv01_logs_missing_tenors() -> None:
    """Сценарий касается тенров, которых нет в PV01 → пропускаются + note."""
    pos = _stress_position()
    greeks = Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0)
    sens = CurveSensitivities(pv01_by_tenor={"1Y": -100.0})  # нет 10Y
    scen = parallel_shift(10, tenors=("1Y", "10Y"))

    report = run_stress_scenarios(
        pos, greeks, base_price=100.0,
        scenarios=[scen], curve_sensitivities=sens,
    )

    result = report.results[0]
    assert result.method == "curve_pv01"
    # учитывается только 1Y: -100 × 0.001 = -0.1
    assert result.pnl == pytest.approx(-100.0 * 0.0010)
    assert any("10Y" in n for n in result.notes)


def test_stress_zero_shift_yields_zero_pnl() -> None:
    """parallel_shift(0) → PnL ровно 0, метод scalar_rho."""
    pos = _stress_position()
    greeks = Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=-700.0)
    scen = parallel_shift(0)

    report = run_stress_scenarios(pos, greeks, base_price=100.0, scenarios=[scen])

    assert report.results[0].pnl == pytest.approx(0.0)
    assert report.results[0].method == "scalar_rho"


def test_stress_signs_long_bond() -> None:
    """Длинная позиция в bond: rho < 0 → +shift даёт -PnL."""
    pos = _stress_position()
    greeks = Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=-700.0)

    up = run_stress_scenarios(pos, greeks, 100.0, [parallel_shift(+10)]).results[0].pnl
    down = run_stress_scenarios(pos, greeks, 100.0, [parallel_shift(-10)]).results[0].pnl

    assert up < 0  # ставка вверх → bond вниз
    assert down > 0  # ставка вниз → bond вверх
    assert up == pytest.approx(-down)  # симметрия first-order


def test_stress_report_pnl_dict_excludes_skipped() -> None:
    """pnl_dict не содержит сценариев, которые были пропущены."""
    pos = _stress_position()
    greeks = Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=-100.0)
    scenarios = [parallel_shift(10), twist(25)]  # второй без curve_sens пропустится

    report = run_stress_scenarios(pos, greeks, base_price=100.0, scenarios=scenarios)
    d = report.pnl_dict()

    assert "parallel_shift_+10bp" in d
    assert "twist_steepener_25bp" not in d


def test_stress_empty_scenarios_raises() -> None:
    pos = _stress_position()
    greeks = Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0)
    with pytest.raises(ValueError, match="scenarios must not be empty"):
        run_stress_scenarios(pos, greeks, base_price=100.0, scenarios=[])


def test_stress_full_standard_set_with_pv01() -> None:
    """Smoke: стандартный набор из 10 сценариев + полные PV01 → все 10 успешны."""
    pos = _stress_position()
    greeks = Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=-700.0)
    # PV01 для всех дефолтных тенров
    sens = CurveSensitivities(
        pv01_by_tenor={t: -50.0 * (i + 1) for i, t in enumerate(DEFAULT_TENORS)}
    )

    report = run_stress_scenarios(
        pos, greeks, base_price=100.0,
        scenarios=standard_scenario_set(),
        curve_sensitivities=sens,
    )

    assert len(report.results) == 10
    assert all(r.method == "curve_pv01" for r in report.results)
    assert all(r.pnl is not None for r in report.results)
    # parallel_shift(0) не в наборе, но parallel ±1bp/10bp/100bp должны быть симметричны
    pnl_pos_10 = next(r.pnl for r in report.results if r.scenario_name == "parallel_shift_+10bp")
    pnl_neg_10 = next(r.pnl for r in report.results if r.scenario_name == "parallel_shift_-10bp")
    assert pnl_pos_10 == pytest.approx(-pnl_neg_10)

# ===========================================================================
#                              BACKTESTING
# ===========================================================================


def _make_case(
    pid: str,
    *,
    instrument_type: str = "option",
    currency: str = "RUB",
    book: str | None = None,
    spot_t0: float = 100.0,
    spot_t1: float = 100.0,
    delta: float = 0.0,
    price_t0: float = 10.0,
    price_t1: float = 10.0,
) -> BacktestCase:
    """Хелпер: собирает BacktestCase с минимальной механикой."""
    position = Position(
        position_id=pid,
        instrument_type=instrument_type,  # type: ignore[arg-type]
        quantity=1000.0,
        currency=currency,
        book=book,
    )
    rf0, rf1 = _make_snapshots(spot_t0=spot_t0, spot_t1=spot_t1, days_between=0)
    greeks = Greeks(delta=delta, gamma=0.0, vega=0.0, theta=0.0)
    pricing = PricingResultMin(
        position_id=pid, price_t0=price_t0, price_t1=price_t1, greeks_t0=greeks
    )
    return BacktestCase(position=position, pricing=pricing, rf_t0=rf0, rf_t1=rf1)


# ----------------------------- Базовые тесты -----------------------------


def test_backtest_empty_raises() -> None:
    with pytest.raises(ValueError, match="cases must not be empty"):
        backtest_attribution([])


def test_backtest_invalid_threshold_raises() -> None:
    case = _make_case("POS-1")
    with pytest.raises(ValueError, match="residual_threshold"):
        backtest_attribution([case], residual_threshold=2.0)


def test_backtest_single_passing_case() -> None:
    """Один perfect-attribution кейс → pass_rate=1.0, residual≈0."""
    case = _make_case(
        "POS-1", spot_t0=100.0, spot_t1=105.0,
        delta=0.5, price_t0=10.0, price_t1=12.5,  # Δ·ΔS = 2.5 = ΔPnL
    )

    report = backtest_attribution([case])

    assert isinstance(report, BacktestReport)
    assert report.overall.n_total == 1
    assert report.overall.n_passed == 1
    assert report.overall.pass_rate == 1.0
    assert len(report.breached_positions) == 0
    assert len(report.errors) == 0


def test_backtest_mixed_pass_fail() -> None:
    """3 проходящих + 1 breach → pass_rate = 0.75."""
    good = [
        _make_case(f"GOOD-{i}", spot_t0=100, spot_t1=105,
                   delta=0.5, price_t0=10, price_t1=12.5)
        for i in range(3)
    ]
    # Bad: greeks говорят delta=0.1, а реальное движение price = 5 → residual огромный
    bad = _make_case("BAD-1", spot_t0=100, spot_t1=110,
                     delta=0.1, price_t0=10, price_t1=15)

    report = backtest_attribution([*good, bad])

    assert report.overall.n_total == 4
    assert report.overall.n_passed == 3
    assert report.overall.n_breached == 1
    assert report.overall.pass_rate == 0.75
    assert len(report.breached_positions) == 1
    assert report.breached_positions[0].position_id == "BAD-1"


# ---------------------------- Fail-soft ошибки ----------------------------


def test_backtest_collects_errors_does_not_raise() -> None:
    """Кейс с position_id mismatch → попадает в errors, остальные считаются."""
    good = _make_case("GOOD-1", spot_t0=100, spot_t1=105,
                      delta=0.5, price_t0=10, price_t1=12.5)

    # Фабрикуем кейс с рассинхроном position_id ↔ pricing
    bad_position = Position(position_id="P-A", instrument_type="option", quantity=1.0)
    bad_pricing = PricingResultMin(
        position_id="P-B",  # mismatch
        price_t0=10.0, price_t1=10.0,
        greeks_t0=Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0),
    )
    rf0, rf1 = _make_snapshots(days_between=0)
    bad_case = BacktestCase(position=bad_position, pricing=bad_pricing, rf_t0=rf0, rf_t1=rf1)

    report = backtest_attribution([good, bad_case])

    assert report.overall.n_total == 1  # только good попал в результаты
    assert len(report.errors) == 1
    assert report.errors[0].position_id == "P-A"
    assert "mismatch" in report.errors[0].message.lower()


# ----------------------------- Группировки -----------------------------


def test_backtest_grouping_by_instrument_type() -> None:
    """Срез by_instrument_type: 2 опциона passed, 1 bond breached."""
    cases = [
        _make_case("OPT-1", instrument_type="option",
                   spot_t0=100, spot_t1=101, delta=1.0, price_t0=10, price_t1=11),
        _make_case("OPT-2", instrument_type="option",
                   spot_t0=100, spot_t1=102, delta=1.0, price_t0=10, price_t1=12),
        _make_case("BOND-1", instrument_type="bond",
                   spot_t0=100, spot_t1=110, delta=0.0,  # никак не объяснит ΔPnL=5
                   price_t0=10, price_t1=15),
    ]

    report = backtest_attribution(cases)

    assert "option" in report.by_instrument_type
    assert "bond" in report.by_instrument_type
    assert report.by_instrument_type["option"].n_total == 2
    assert report.by_instrument_type["option"].pass_rate == 1.0
    assert report.by_instrument_type["bond"].n_total == 1
    assert report.by_instrument_type["bond"].pass_rate == 0.0


def test_backtest_grouping_by_currency() -> None:
    cases = [
        _make_case("RUB-1", currency="RUB",
                   spot_t0=100, spot_t1=101, delta=1.0, price_t0=10, price_t1=11),
        _make_case("USD-1", currency="USD",
                   spot_t0=100, spot_t1=101, delta=1.0, price_t0=10, price_t1=11),
    ]

    report = backtest_attribution(cases)

    assert set(report.by_currency.keys()) == {"RUB", "USD"}
    assert report.by_currency["RUB"].n_total == 1
    assert report.by_currency["USD"].n_total == 1


def test_backtest_grouping_by_book_skips_none() -> None:
    """Позиции без book не попадают в by_book срез."""
    cases = [
        _make_case("WITH-BOOK", book="EQ_VOL",
                   spot_t0=100, spot_t1=101, delta=1.0, price_t0=10, price_t1=11),
        _make_case("NO-BOOK", book=None,
                   spot_t0=100, spot_t1=101, delta=1.0, price_t0=10, price_t1=11),
    ]

    report = backtest_attribution(cases)

    assert "EQ_VOL" in report.by_book
    assert report.by_book["EQ_VOL"].n_total == 1
    assert None not in report.by_book  # type: ignore[comparison-overlap]
    assert len(report.by_book) == 1


# ---------------------------- Edge: zero-PnL ----------------------------


def test_backtest_zero_pnl_excluded_from_ratio_stats() -> None:
    """Позиция с ΔPnL≈0 → n_zero_pnl=1, попадает в total и в signed stats."""
    zero_case = _make_case("ZERO-1", spot_t0=100, spot_t1=100,
                           delta=0.5, price_t0=10, price_t1=10)
    good_case = _make_case("GOOD-1", spot_t0=100, spot_t1=105,
                           delta=0.5, price_t0=10, price_t1=12.5)

    report = backtest_attribution([zero_case, good_case])

    assert report.overall.n_total == 2
    assert report.overall.n_zero_pnl == 1
    # Percentiles считаются только по валидным ratios (1 кейс) — не None
    assert report.overall.mean_abs_residual_ratio is not None
    assert report.overall.mean_abs_residual_ratio == pytest.approx(0.0, abs=1e-10)


def test_backtest_all_zero_pnl_yields_none_percentiles() -> None:
    """Все позиции zero-PnL → percentile-метрики = None."""
    cases = [
        _make_case(f"Z-{i}", spot_t0=100, spot_t1=100,
                   delta=0.0, price_t0=10, price_t1=10)
        for i in range(3)
    ]

    report = backtest_attribution(cases)

    assert report.overall.n_zero_pnl == 3
    assert report.overall.mean_abs_residual_ratio is None
    assert report.overall.p95_abs_residual_ratio is None


# ----------------------------- Метрики -----------------------------


def test_backtest_signed_residual_bias_detection() -> None:
    """Все residuals одного знака → mean_signed_residual ≠ 0 (детектит bias)."""
    # Случай: pricing систематически недооценивает ΔPnL на 0.5
    cases = [
        _make_case(f"BIAS-{i}", spot_t0=100, spot_t1=105,
                   delta=0.5, price_t0=10, price_t1=12.5 + 0.5)  # ΔPnL = 3.0, explained = 2.5
        for i in range(5)
    ]

    report = backtest_attribution(cases, residual_threshold=0.5)  # высокий threshold чтобы пропустить

    # signed residual = +0.5 для каждого → mean ≈ +0.5
    assert report.overall.mean_signed_residual == pytest.approx(0.5, abs=1e-10)
    # Все одинаковые → std ≈ 0
    assert report.overall.std_signed_residual == pytest.approx(0.0, abs=1e-10)


def test_backtest_p95_p99_reflect_distribution() -> None:
    """100 кейсов с линейно растущим residual_ratio → p95 ≈ 95-й перцентиль."""
    cases = []
    for i in range(100):
        # ΔPnL = 10, residual = 0.01·(i+1) → ratio растёт от 0.001 до 0.10
        # explained = 10 - 0.01·(i+1); подбираем delta так, чтобы explained = delta·ΔS
        true_pnl = 10.0
        residual = 0.01 * (i + 1)
        explained_target = true_pnl - residual
        # ΔS = 10, delta = explained_target / 10
        cases.append(_make_case(
            f"P-{i:03d}", spot_t0=100.0, spot_t1=110.0,
            delta=explained_target / 10.0, price_t0=10.0, price_t1=20.0,
        ))

    # threshold=0.99 чтобы все прошли (нам важно посмотреть на статистики)
    report = backtest_attribution(cases, residual_threshold=0.99)

    # 95-й перцентиль ≈ 0.0955, 99-й ≈ 0.0995
    assert report.overall.p95_abs_residual_ratio == pytest.approx(0.0955, abs=1e-3)
    assert report.overall.p99_abs_residual_ratio == pytest.approx(0.0995, abs=1e-3)


def test_backtest_custom_threshold_propagates_to_pass_rate() -> None:
    """Threshold 1% делает кейс с residual_ratio=3% breached, threshold 5% — passed."""
    # ΔPnL = 10, explained = 9.7 → residual = 0.3, ratio = 3%
    case = _make_case("EDGE-1", spot_t0=100, spot_t1=110,
                      delta=0.97, price_t0=10, price_t1=20)

    tight = backtest_attribution([case], residual_threshold=0.01)
    loose = backtest_attribution([case], residual_threshold=0.05)

    assert tight.overall.pass_rate == 0.0
    assert loose.overall.pass_rate == 1.0
    assert tight.residual_threshold == 0.01
    assert loose.residual_threshold == 0.05


def test_backtest_breached_positions_complete() -> None:
    """breached_positions содержит все непрошедшие кейсы с правильными id."""
    bad_ids = ["B-1", "B-2", "B-3"]
    cases = [
        _make_case(pid, spot_t0=100, spot_t1=110,
                   delta=0.0, price_t0=10, price_t1=15)  # ничего не объяснено
        for pid in bad_ids
    ]
    cases.append(_make_case("OK-1", spot_t0=100, spot_t1=105,
                            delta=0.5, price_t0=10, price_t1=12.5))

    report = backtest_attribution(cases)

    breached_ids = {r.position_id for r in report.breached_positions}
    assert breached_ids == set(bad_ids)


def test_backtest_n_zero_pnl_counted_correctly() -> None:
    """Smoke: 2 zero-PnL, 3 non-zero — n_zero_pnl = 2."""
    cases = [
        _make_case("Z-1", spot_t0=100, spot_t1=100, price_t0=10, price_t1=10),
        _make_case("Z-2", spot_t0=100, spot_t1=100, price_t0=10, price_t1=10),
        _make_case("NZ-1", spot_t0=100, spot_t1=105, delta=0.5, price_t0=10, price_t1=12.5),
        _make_case("NZ-2", spot_t0=100, spot_t1=105, delta=0.5, price_t0=10, price_t1=12.5),
        _make_case("NZ-3", spot_t0=100, spot_t1=105, delta=0.5, price_t0=10, price_t1=12.5),
    ]

    report = backtest_attribution(cases)

    assert report.overall.n_total == 5
    assert report.overall.n_zero_pnl == 2