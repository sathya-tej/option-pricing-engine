from __future__ import annotations

import math
from .bsm import bsm_price
from .greeks import greeks

def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    q: float = 0.0,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> float:
    """Find sigma such that bsm_price(...) == price.

    Raises ValueError if the price violates no-arbitrage bounds
    (no volatility can produce it).
    """
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")
    if S <= 0 or K <= 0 or T <= 0:
        raise ValueError("S, K and T must be positive.")

   
    disc_S = S * math.exp(-q * T)
    disc_K = K * math.exp(-r * T)
    if option_type == "call":
        lower, upper = max(disc_S - disc_K, 0.0), disc_S
    else:
        lower, upper = max(disc_K - disc_S, 0.0), disc_K
    if not (lower < price < upper):
        raise ValueError(
            f"Price {price} is outside the no-arbitrage range ({lower:.6f}, {upper:.6f})."
        )

    
    lo, hi = 1e-8, 1.0
    while bsm_price(S, K, T, r, hi, option_type, q) < price:
        hi *= 2.0
        if hi > 100.0:
            raise ValueError("Could not bracket the implied volatility.")

    
    sigma = 0.2
    for _ in range(max_iter):
        diff = bsm_price(S, K, T, r, sigma, option_type, q) - price
        if abs(diff) < tol:
            return sigma

        
        if diff > 0:
            hi = sigma
        else:
            lo = sigma

        vega = greeks(S, K, T, r, sigma, option_type, q)["vega"]
        new_sigma = sigma - diff / vega if vega > 1e-12 else None

       
        if new_sigma is None or not (lo < new_sigma < hi):
            new_sigma = 0.5 * (lo + hi)
        sigma = new_sigma

    raise RuntimeError("Implied volatility did not converge.")