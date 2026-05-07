

import numpy as np
from dataclasses import dataclass


@dataclass
class SwapResult:
    """
    Результат расчёта свопа.

    npv           : итоговая стоимость свопа (float_leg - fixed_leg)
    fixed_leg_npv : NPV фиксированной ноги (то что платим)
    float_leg_npv : NPV плавающей ноги (то что получаем)
    dv01          : чувствительность NPV к сдвигу ставок на 1bp
    """
    npv:           float
    fixed_leg_npv: float
    float_leg_npv: float
    dv01:          float


def swap_price(
    notional: float,
    fixed_rate: float,
    times: list[float],
    discount_rates: list[float],
    forward_rates: list[float],
) -> SwapResult:

    times          = np.array(times)
    discount_rates = np.array(discount_rates)
    forward_rates  = np.array(forward_rates)


    periods = np.diff(np.concatenate([[0.0], times]))


    discount_factors = 1.0 / (1.0 + discount_rates) ** times


    fixed_cashflows   = fixed_rate * periods * notional
    fixed_leg_npv     = np.sum(fixed_cashflows * discount_factors)


    # PV = Σ forward_rate_t · period · notional · df_t
    float_cashflows   = forward_rates * periods * notional
    float_leg_npv     = np.sum(float_cashflows * discount_factors)


    npv = float_leg_npv - fixed_leg_npv


    bump = 0.0001

    df_up   = 1.0 / (1.0 + discount_rates + bump) ** times
    df_down = 1.0 / (1.0 + discount_rates - bump) ** times

    fr_up   = forward_rates + bump
    fr_down = forward_rates - bump

    npv_up   = np.sum((fr_up   * periods * notional) * df_up)   - np.sum(fixed_cashflows * df_up)
    npv_down = np.sum((fr_down * periods * notional) * df_down) - np.sum(fixed_cashflows * df_down)

    dv01 = (npv_up - npv_down) / 2.0

    return SwapResult(
        npv=npv,
        fixed_leg_npv=fixed_leg_npv,
        float_leg_npv=float_leg_npv,
        dv01=dv01,
    )