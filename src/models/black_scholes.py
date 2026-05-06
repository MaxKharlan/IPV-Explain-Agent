"""
Масштабирование (vega per 1%, theta per day, rho per 1bp) — НЕ ЗДЕСЬ.
Это делает Pricing Agent. Смешивать нельзя — residual съест ошибку.
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass


@dataclass
class BSResult:
    """
    delta : dV/dS
    gamma : d²V/dS²
    vega  : dV/dσ        (на единицу σ, не на 1%)
    theta : dV/dt        (на единицу t в годах, не на день)
    rho   : dV/dr        (на единицу r, не на 1bp)
  """
    price: float
    delta: float
    gamma: float
    vega:  float
    theta: float
    rho:   float


def _d1_d2(S: float, K: float, r: float, sigma: float, T: float):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def black_scholes(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
) -> BSResult:

    d1, d2 = _d1_d2(S, K, r, sigma, T)

    Nd1  = norm.cdf(d1)
    Nd2  = norm.cdf(d2)
    N_d1 = norm.cdf(-d1)
    N_d2 = norm.cdf(-d2)
    npd1 = norm.pdf(d1)

    disc = np.exp(-r * T)

    if option_type == "call":
        price = S * Nd1 - K * disc * Nd2
    else:
        price = K * disc * N_d2 - S * N_d1

    if option_type == "call":
        delta = Nd1           # от 0 до +1
    else:
        delta = Nd1 - 1       # от -1 до 0

    gamma = npd1 / (S * sigma * np.sqrt(T))

    vega = S * np.sqrt(T) * npd1
    if option_type == "call":
        theta = (
            -S * npd1 * sigma / (2 * np.sqrt(T))
            - r * K * disc * Nd2
        )
    else:
        theta = (
            -S * npd1 * sigma / (2 * np.sqrt(T))
            + r * K * disc * N_d2
        )

    if option_type == "call":
        rho = K * T * disc * Nd2
    else:
        rho = -K * T * disc * N_d2

    return BSResult(
        price=price,
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        rho=rho,
    )

"""
✅ Цена call и put
✅ Delta, Gamma, Vega, Theta, Rho
✅ Raw greeks конвенция
"""