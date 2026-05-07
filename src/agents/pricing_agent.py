"""
pricing_agent.py
================
Pricing Agent — оборачивает quant-модели и возвращает PricingResultMin.

PricingResultMin — минимальный контракт между Pricing Agent и Attribution Engine.
Содержит: цены на двух датах + raw Greeks на t0.

"""

from dataclasses import dataclass
from enum import Enum

from src.models.black_scholes import black_scholes, BSResult
from src.models.bond_pricing import bond_price, BondResult
from src.models.swap_pricing import swap_price, SwapResult




class InstrumentType(str, Enum):
    OPTION = "option"
    BOND   = "bond"
    SWAP   = "swap"




@dataclass
class Greeks:

    delta: float = 0.0
    gamma: float = 0.0
    vega:  float = 0.0
    theta: float = 0.0
    rho:   float = 0.0
    dv01:  float = 0.0


@dataclass
class PricingResultMin:

    price_t0:        float
    price_t1:        float
    greeks_t0:       Greeks
    instrument_type: InstrumentType



def price_option(
    # параметры на t0
    S0: float, K: float, r0: float, sigma0: float, T0: float,
    # параметры на t1
    S1: float,           r1: float, sigma1: float, T1: float,
    option_type: str = "call",
) -> PricingResultMin:

    result_t0: BSResult = black_scholes(S0, K, r0, sigma0, T0, option_type)
    result_t1: BSResult = black_scholes(S1, K, r1, sigma1, T1, option_type)

    greeks = Greeks(
        delta=result_t0.delta,
        gamma=result_t0.gamma,
        vega=result_t0.vega,
        theta=result_t0.theta,
        rho=result_t0.rho,
    )

    return PricingResultMin(
        price_t0=result_t0.price,
        price_t1=result_t1.price,
        greeks_t0=greeks,
        instrument_type=InstrumentType.OPTION,
    )


def price_bond(
    # параметры на t0
    cashflows: list[float],
    times_t0: list[float], rates_t0: list[float],
    # параметры на t1
    times_t1: list[float], rates_t1: list[float],
) -> PricingResultMin:

    result_t0: BondResult = bond_price(cashflows, times_t0, rates_t0)
    result_t1: BondResult = bond_price(cashflows, times_t1, rates_t1)


    greeks = Greeks(dv01=result_t0.dv01)

    return PricingResultMin(
        price_t0=result_t0.price,
        price_t1=result_t1.price,
        greeks_t0=greeks,
        instrument_type=InstrumentType.BOND,
    )


def price_swap(
    notional: float,
    fixed_rate: float,

    times_t0: list[float],
    discount_rates_t0: list[float],
    forward_rates_t0: list[float],

    times_t1: list[float],
    discount_rates_t1: list[float],
    forward_rates_t1: list[float],
) -> PricingResultMin:

    result_t0: SwapResult = swap_price(
        notional, fixed_rate, times_t0, discount_rates_t0, forward_rates_t0
    )
    result_t1: SwapResult = swap_price(
        notional, fixed_rate, times_t1, discount_rates_t1, forward_rates_t1
    )

    greeks = Greeks(dv01=result_t0.dv01)

    return PricingResultMin(
        price_t0=result_t0.npv,
        price_t1=result_t1.npv,
        greeks_t0=greeks,
        instrument_type=InstrumentType.SWAP,
    )