"""
pricing_agent.py
================
Pricing Agent — оборачивает quant-модели и возвращает PricingResultMin.

PricingResultMin импортируется из src.attribution.schemas — это общий
контракт между Pricing Agent и Attribution Engine.

Конвенция греков: ВСЕ греки на выходе — raw (per 1.0 unit, без масштабирования
на 1bp/1%). Для bond/swap внутренний dv01 (per 1bp) конвертируется
в raw rho на границе модуля.
"""

from src.attribution.schemas import Greeks, PricingResultMin
from src.models.black_scholes import black_scholes, BSResult
from src.models.bond_pricing import bond_price, BondResult
from src.models.swap_pricing import swap_price, SwapResult


# 1bp в абсолютных rate units (для конверсии dv01 → raw rho)
BPS_TO_RATE: float = 1e-4


def price_option(
    position_id: str,
    # параметры на t0
    S0: float, K: float, r0: float, sigma0: float, T0: float,
    # параметры на t1
    S1: float,           r1: float, sigma1: float, T1: float,
    option_type: str = "call",
) -> PricingResultMin:
    """Pricing европейского опциона через Black-Scholes на двух датах.

    Возвращает PricingResultMin с raw greeks (delta, gamma, vega, theta, rho).
    Все греки уже raw по конвенции BSResult.
    """
    result_t0: BSResult = black_scholes(S0, K, r0, sigma0, T0, option_type)
    result_t1: BSResult = black_scholes(S1, K, r1, sigma1, T1, option_type)

    return PricingResultMin(
        position_id=position_id,
        price_t0=result_t0.price,
        price_t1=result_t1.price,
        greeks_t0=Greeks(
            delta=result_t0.delta,
            gamma=result_t0.gamma,
            vega=result_t0.vega,
            theta=result_t0.theta,
            rho=result_t0.rho,
        ),
    )


def price_bond(
    position_id: str,
    cashflows: list[float],
    # параметры на t0
    times_t0: list[float], rates_t0: list[float],
    # параметры на t1
    times_t1: list[float], rates_t1: list[float],
) -> PricingResultMin:
    """Pricing облигации через дисконтирование cashflows на двух датах.

    Внутренний dv01 (per 1bp) конвертируется в raw rho (per 1.0 rate unit)
    для совместимости с Attribution Engine.
    """
    result_t0: BondResult = bond_price(cashflows, times_t0, rates_t0)
    result_t1: BondResult = bond_price(cashflows, times_t1, rates_t1)

    # dv01 (per 1bp) → raw rho (per 1.0 rate unit)
    rho_raw = result_t0.dv01 / BPS_TO_RATE

    return PricingResultMin(
        position_id=position_id,
        price_t0=result_t0.price,
        price_t1=result_t1.price,
        greeks_t0=Greeks(
            delta=0.0,
            gamma=0.0,
            vega=0.0,
            theta=0.0,
            rho=rho_raw,
        ),
    )


def price_swap(
    position_id: str,
    notional: float,
    fixed_rate: float,
    # параметры на t0
    times_t0: list[float],
    discount_rates_t0: list[float],
    forward_rates_t0: list[float],
    # параметры на t1
    times_t1: list[float],
    discount_rates_t1: list[float],
    forward_rates_t1: list[float],
) -> PricingResultMin:
    """Pricing процентного свопа через NPV двух ног на двух датах.

    Внутренний dv01 (per 1bp) конвертируется в raw rho (per 1.0 rate unit).
    """
    result_t0: SwapResult = swap_price(
        notional, fixed_rate, times_t0, discount_rates_t0, forward_rates_t0
    )
    result_t1: SwapResult = swap_price(
        notional, fixed_rate, times_t1, discount_rates_t1, forward_rates_t1
    )

    # dv01 (per 1bp) → raw rho (per 1.0 rate unit)
    rho_raw = result_t0.dv01 / BPS_TO_RATE

    return PricingResultMin(
        position_id=position_id,
        price_t0=result_t0.npv,
        price_t1=result_t1.npv,
        greeks_t0=Greeks(
            delta=0.0,
            gamma=0.0,
            vega=0.0,
            theta=0.0,
            rho=rho_raw,
        ),
    )