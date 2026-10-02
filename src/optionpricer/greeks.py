from __future__ import annotations
import math

from .bsm import d1_d2, norm_cdf

def norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
) -> dict[str, float]:
    """Return delta, gamma, vega, theta, rho for a European option.

    Conventions (raw, per unit change):
      delta : dV/dS
      gamma : d2V/dS2
      vega  : dV/dsigma   (divide by 100 for "per 1 vol point")
      theta : dV/dt       (calendar time, per year; divide by 365 for per day)
      rho   : dV/dr       (divide by 100 for "per 1% rate move")
    """
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive.")
    if T <= 0 or sigma <= 0:
        raise ValueError("Greeks require T > 0 and sigma > 0.")

    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    sqrt_T = math.sqrt(T)
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    pdf_d1 = norm_pdf(d1)

    gamma = disc_q * pdf_d1 / (S * sigma * sqrt_T)
    vega = S * disc_q * pdf_d1 * sqrt_T
    decay = -S * disc_q * pdf_d1 * sigma / (2.0 * sqrt_T)

    if option_type == "call":
        delta = disc_q * norm_cdf(d1)
        theta = decay - r * K * disc_r * norm_cdf(d2) + q * S * disc_q * norm_cdf(d1)
        rho = K * T * disc_r * norm_cdf(d2)
    else:
        delta = -disc_q * norm_cdf(-d1)
        theta = decay + r * K * disc_r * norm_cdf(-d2) - q * S * disc_q * norm_cdf(-d1)
        rho = -K * T * disc_r * norm_cdf(-d2)

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


