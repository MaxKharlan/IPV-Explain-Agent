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

from __future__ import annotations

from datetime import date
from typing import Any

from src.agents.state import IPVState
from src.attribution.schemas import Greeks, PricingResultMin
from src.models.black_scholes import black_scholes, BSResult
from src.models.bond_pricing import bond_price, BondResult
from src.models.swap_pricing import swap_price, SwapResult


# 1bp в абсолютных rate units (для конверсии dv01 → raw rho)
BPS_TO_RATE: float = 1e-4


def _parse_iso_date(value: str) -> date:
    """Преобразует YYYY-MM-DD строку в date."""
    return date.fromisoformat(value)


def _normalize_rate(rate: float) -> float:
    """Приводит ставку к decimal rate format."""
    if abs(rate) > 1.0:
        return rate / 100.0
    return rate


def _extract_curve_inputs(snapshot: dict[str, Any]) -> tuple[list[float], list[float]]:
    """Достаёт times/rates из yield_curve snapshot."""
    points = snapshot.get("yield_curve", {}).get("points", [])
    times: list[float] = []
    rates: list[float] = []
    for point in points:
        tenor = str(point.get("tenor", "")).upper()
        if tenor.endswith("Y"):
            times.append(float(tenor[:-1]))
        elif tenor.endswith("M"):
            times.append(float(tenor[:-1]) / 12.0)
        else:
            continue
        rates.append(_normalize_rate(float(point.get("rate", 0.0))))
    if not times:
        raise ValueError("Yield curve points are required for pricing.")
    return times, rates


def _extract_reference_rate(snapshot: dict[str, Any]) -> float:
    """Берёт опорную ставку из первой точки кривой."""
    _, rates = _extract_curve_inputs(snapshot)
    return rates[0]


def resolve_option_sigmas(position: dict[str, Any]) -> tuple[float, float]:
    """Определяет sigma_t0 и sigma_t1 для option pricing.

    Пока Quant Core ещё не строит vol surface в orchestration-layer,
    используем явные волатильности из position.instrument. Если они не заданы,
    применяем безопасный demo fallback.
    """
    instrument = position.get("instrument", {})
    sigma0 = instrument.get("vol_t0", instrument.get("sigma_t0", instrument.get("vol", 0.25)))
    sigma1 = instrument.get("vol_t1", instrument.get("sigma_t1", instrument.get("vol", sigma0)))
    return float(sigma0), float(sigma1)


def _extract_option_spot(snapshot: dict[str, Any], underlier: str) -> float:
    """Достаёт spot по underlier из market snapshot."""
    spot_prices = snapshot.get("spot_prices", {})
    if underlier not in spot_prices:
        raise KeyError(f"Spot price for underlier={underlier} not found in snapshot.")
    return float(spot_prices[underlier])


def _years_to_maturity(snapshot_date: str, maturity_date: str) -> float:
    """Возвращает время до погашения/экспирации в годах."""
    start = _parse_iso_date(snapshot_date)
    end = _parse_iso_date(maturity_date)
    return max((end - start).days / 365.0, 1e-6)


def build_pricing_result(
    position: dict[str, Any],
    market_t0: dict[str, Any],
    market_t1: dict[str, Any],
) -> PricingResultMin:
    """Строит PricingResultMin из position и пары market snapshot.

    Сейчас end-to-end wiring полностью поддерживает option flow.
    Bond/swap wiring можно нарастить следующим шагом поверх уже готовых
    quant-функций price_bond/price_swap.
    """
    instrument_type = position.get("instrument_type")
    instrument = position.get("instrument", {})
    position_id = position["position_id"]

    if instrument_type != "option":
        raise NotImplementedError(
            "Current pipeline wiring supports option positions first. "
            "Bond and swap agent wiring should be added next."
        )

    underlier = instrument.get("underlier")
    strike = float(instrument["strike"])
    maturity_date = instrument["maturity_date"]
    option_type = instrument.get("option_type", "call")
    sigma0, sigma1 = resolve_option_sigmas(position)
    spot_t0 = _extract_option_spot(market_t0, underlier)
    spot_t1 = _extract_option_spot(market_t1, underlier)
    rate_t0 = _extract_reference_rate(market_t0)
    rate_t1 = _extract_reference_rate(market_t1)
    time_t0 = _years_to_maturity(market_t0["snapshot_date"], maturity_date)
    time_t1 = _years_to_maturity(market_t1["snapshot_date"], maturity_date)

    return price_option(
        position_id=position_id,
        S0=spot_t0,
        K=strike,
        r0=rate_t0,
        sigma0=sigma0,
        T0=time_t0,
        S1=spot_t1,
        r1=rate_t1,
        sigma1=sigma1,
        T1=time_t1,
        option_type=option_type,
    )


def run_pricing_agent(state: IPVState) -> IPVState:
    """Заполняет state полем pricing_result."""
    position = state.get("position")
    market_t0 = state.get("market_snapshot_t0")
    market_t1 = state.get("market_snapshot_t1")
    if position is None or market_t0 is None or market_t1 is None:
        raise ValueError("Pricing agent requires position and both market snapshots.")

    pricing_result = build_pricing_result(position, market_t0, market_t1)
    state["pricing_result"] = pricing_result.model_dump(mode="json")
    return state


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
