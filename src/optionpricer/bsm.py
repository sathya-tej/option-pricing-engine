"""Black-Scholes-Merton pricing for European options."""

from __future__ import annotations
import scipy.stats
import math
from scipy.stats import norm


def _validate(S: float, K: float, T: float, sigma: float) -> None:
    if S <= 0:
        raise ValueError("Spot price S must be positive.")
    if K <= 0:
        raise ValueError("Strike K must be positive.")
    if T < 0:
        raise ValueError("Time to expiry T cannot be negative.")
    if sigma < 0:
        raise ValueError("Volatility sigma cannot be negative.")


def d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0):
    """Return (d1, d2). Requires T > 0 and sigma > 0."""
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def bsm_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
) -> float:
    """Price a European option under Black-Scholes-Merton.

    Parameters
    ----------
    S : spot price of the underlying
    K : strike price
    T : time to expiry in years
    r : continuously compounded risk-free rate
    sigma : annualised volatility
    option_type : "call" or "put"
    q : continuous dividend yield
    """
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")
    _validate(S, K, T, sigma)
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    
    # Edge cases: at expiry or zero volatility, the payoff is deterministic.
    if T == 0 or sigma == 0:
        forward_value = S * disc_q - K * disc_r
        if option_type == "call":
            return max(forward_value, 0.0)
        return max(-forward_value, 0.0)

    d1, d2 = d1_d2(S, K, T, r, sigma, q)

    if option_type == "call":
        return S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    return K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)


