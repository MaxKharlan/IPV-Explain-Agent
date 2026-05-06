
import numpy as np
from dataclasses import dataclass, field


@dataclass
class BondResult:
    """

    dv01     : чувствительность цены к сдвигу ставки на 1bp
               (аналог Delta для облигаций)
    """
    price:    float
    duration: float
    dv01:     float


def bond_price(
    cashflows: list[float],
    times: list[float],
    rates: list[float],
) -> BondResult:

    cashflows = np.array(cashflows)
    times     = np.array(times)
    rates     = np.array(rates)


    discount_factors = 1.0 / (1.0 + rates) ** times
    pv_cashflows     = cashflows * discount_factors   # PV каждого потока
    price            = np.sum(pv_cashflows)


    duration = np.sum(times * pv_cashflows) / price

    # ── DV01 ──────────────────────────────────────────────────────────────
    # Изменение цены при сдвиге всех ставок на 1bp (0.0001)
    # Численная производная: (P(r + 1bp) - P(r - 1bp)) / 2
    bump = 0.0001
    pv_up   = np.sum(cashflows / (1.0 + rates + bump) ** times)
    pv_down = np.sum(cashflows / (1.0 + rates - bump) ** times)
    dv01    = (pv_up - pv_down) / 2.0

    return BondResult(price=price, duration=duration, dv01=dv01)