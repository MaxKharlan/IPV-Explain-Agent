"""Taylor-разложение PnL по грекам.

Математическая модель (per-unit price):

    ΔV ≈ Δ·ΔS + ½·Γ·(ΔS)² + ν·Δσ + θ·Δt + ρ·Δr + ε

где
    ΔS = S₁ - S₀          — изменение спота
    Δσ = σ₁ - σ₀          — изменение implied vol
    Δt = (t₁ - t₀) / 365  — календарное время в годах (ACT/365)
    Δr = r₁ - r₀          — изменение ставки (опционально)
    ε                     — unexplained residual (вкл. cross-greeks: vanna,
                            vomma, charm, color и higher-order terms)

Греки берутся на t0 — это first-order аппроксимация вокруг состояния t0.
Для контрольной точки на ½(t0 + t1) (predictor-corrector) — отдельный
метод, в скоупе demo не нужен.
"""

from __future__ import annotations

from datetime import date

import numpy as np
from loguru import logger

from src.attribution.schemas import (
    AttributionComponents,
    AttributionResult,
    Greeks,
    Position,
    PricingResultMin,
    RiskFactorSnapshot,
)

# Дефолтный порог: |ε| / |ΔPnL| < 5% — стандартная планка для healthy attribution
DEFAULT_RESIDUAL_THRESHOLD: float = 0.05

# Численная защита: |ΔPnL| ниже — считаем «approximately zero»
ZERO_PNL_TOLERANCE: float = 1e-12


def _years_between(d0: date, d1: date) -> float:
    """Календарное время между двумя датами в годах (ACT/365).

    Args:
        d0: Начальная дата.
        d1: Конечная дата.

    Returns:
        Δt в годах. Может быть нулём, если даты совпадают.
    """
    return float((d1 - d0).days) / 365.0


def _compute_components(
    greeks: Greeks,
    rf_t0: RiskFactorSnapshot,
    rf_t1: RiskFactorSnapshot,
) -> tuple[AttributionComponents, list[str]]:
    """Численное ядро Taylor-разложения.

    Отделено от run_attribution, чтобы (а) тестировать чистую математику,
    (б) переиспользовать в backtesting/stress tests без Pydantic-обёртки.

    Args:
        greeks: Сырые греки на t0.
        rf_t0: Снапшот риск-факторов на t0.
        rf_t1: Снапшот риск-факторов на t1.

    Returns:
        Кортеж (компоненты, список диагностических сообщений). Residual
        здесь ещё не считается — для него нужны цены, см. run_attribution.
    """
    # float64 явно — чтобы исключить случайный float32 при операциях с numpy
    d_spot = np.float64(rf_t1.spot - rf_t0.spot)
    d_vol = np.float64(rf_t1.vol - rf_t0.vol)
    d_time = np.float64(_years_between(rf_t0.snapshot_date, rf_t1.snapshot_date))

    delta_effect = float(np.float64(greeks.delta) * d_spot)
    gamma_effect = float(0.5 * np.float64(greeks.gamma) * d_spot * d_spot)
    vega_effect = float(np.float64(greeks.vega) * d_vol)
    theta_effect = float(np.float64(greeks.theta) * d_time)

    notes: list[str] = []

    # Rho — опциональный. Применяем только если есть и грек, и обе ставки
    rho_effect = 0.0
    has_rho = greeks.rho is not None
    has_both_rates = rf_t0.rate is not None and rf_t1.rate is not None

    if has_rho and has_both_rates:
        d_rate = np.float64(rf_t1.rate - rf_t0.rate)
        rho_effect = float(np.float64(greeks.rho) * d_rate)
    elif has_rho or rf_t0.rate is not None or rf_t1.rate is not None:
        notes.append(
            "Partial rate data: rho contribution skipped, residual will absorb rate move."
        )

    # residual считаем в run_attribution — здесь возвращаем 0.0 как placeholder
    components = AttributionComponents(
        delta_effect=delta_effect,
        gamma_effect=gamma_effect,
        vega_effect=vega_effect,
        theta_effect=theta_effect,
        rho_effect=rho_effect,
        residual=0.0,
    )
    return components, notes


def run_attribution(
    position: Position,
    pricing: PricingResultMin,
    rf_t0: RiskFactorSnapshot,
    rf_t1: RiskFactorSnapshot,
    *,
    residual_threshold: float = DEFAULT_RESIDUAL_THRESHOLD,
) -> AttributionResult:
    """Запускает PnL attribution между двумя снапшотами риск-факторов.

    Args:
        position: Метаданные позиции.
        pricing: Результат Pricing Agent — цены на t0/t1 и греки на t0.
        rf_t0: Снапшот риск-факторов на t0.
        rf_t1: Снапшот риск-факторов на t1.
        residual_threshold: Допустимое отношение |ε| / |ΔPnL|. По умолчанию 5%.

    Returns:
        AttributionResult с покомпонентным разложением и флагом валидации.

    Raises:
        ValueError: при несогласованности position_id или инверсии дат.
    """
    if pricing.position_id != position.position_id:
        raise ValueError(
            f"position_id mismatch: position={position.position_id}, "
            f"pricing={pricing.position_id}"
        )
    if rf_t1.snapshot_date < rf_t0.snapshot_date:
        raise ValueError(
            f"Inverted snapshot order: t0={rf_t0.snapshot_date}, t1={rf_t1.snapshot_date}"
        )
    if not (0.0 < residual_threshold < 1.0):
        raise ValueError(
            f"residual_threshold must be in (0, 1), got {residual_threshold}"
        )

    logger.debug(
        "Running attribution for {} ({}, {} {})",
        position.position_id,
        position.instrument_type,
        position.quantity,
        position.currency,
    )

    raw_components, notes = _compute_components(pricing.greeks_t0, rf_t0, rf_t1)

    total_pnl = float(np.float64(pricing.price_t1) - np.float64(pricing.price_t0))
    explained_pnl = (
        raw_components.delta_effect
        + raw_components.gamma_effect
        + raw_components.vega_effect
        + raw_components.theta_effect
        + raw_components.rho_effect
    )
    residual = total_pnl - explained_pnl

    # explained_ratio безопасно при near-zero total_pnl
    if abs(total_pnl) < ZERO_PNL_TOLERANCE:
        explained_ratio = 1.0 if abs(residual) < ZERO_PNL_TOLERANCE else 0.0
        residual_passed = abs(residual) < ZERO_PNL_TOLERANCE
        notes.append(
            "Total PnL ≈ 0; relative residual undefined, fell back to absolute test."
        )
    else:
        explained_ratio = explained_pnl / total_pnl
        residual_passed = (abs(residual) / abs(total_pnl)) < residual_threshold

    if not residual_passed:
        logger.warning(
            "Residual breach for {}: residual={:.6f}, total_pnl={:.6f}, "
            "ratio={:.4f}, threshold={:.4f}",
            position.position_id,
            residual,
            total_pnl,
            (abs(residual) / abs(total_pnl)) if abs(total_pnl) >= ZERO_PNL_TOLERANCE else float("inf"),
            residual_threshold,
        )

    components = AttributionComponents(
        delta_effect=raw_components.delta_effect,
        gamma_effect=raw_components.gamma_effect,
        vega_effect=raw_components.vega_effect,
        theta_effect=raw_components.theta_effect,
        rho_effect=raw_components.rho_effect,
        residual=residual,
    )

    return AttributionResult(
        position_id=position.position_id,
        currency=position.currency,
        price_t0=float(pricing.price_t0),
        price_t1=float(pricing.price_t1),
        total_pnl=total_pnl,
        components=components,
        explained_pnl=explained_pnl,
        explained_ratio=explained_ratio,
        residual_threshold_passed=residual_passed,
        residual_threshold=residual_threshold,
        notes=notes,
    )