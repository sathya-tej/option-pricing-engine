import pytest
import math

from optionpricer import bsm_price, implied_vol

CASES = [
    (100, 100, 1.0, 0.05, 0.00),
    (100, 90, 0.5, 0.03, 0.02),
    (100, 150, 0.5, 0.05, 0.00),  
    (100, 60, 1.0, 0.02, 0.01),    
    (100, 100, 0.02, 0.05, 0.00),
]

@pytest.mark.parametrize("opt", ["call", "put"])
@pytest.mark.parametrize("true_sigma", [0.05, 0.2, 0.5, 1.5, 3.0])
@pytest.mark.parametrize("S, K, T, r, q", CASES)
def test_round_trip(S, K, T, r, q, true_sigma, opt):
    market_price = bsm_price(S, K, T, r, true_sigma, opt, q)

    disc_S = S * math.exp(-q * T)
    disc_K = K * math.exp(-r * T)
    if opt == "call":
        lower, upper = max(disc_S - disc_K, 0.0), disc_S
    else:
        lower, upper = max(disc_K - disc_S, 0.0), disc_K

    # sigma is not identifiable when the price sits on a no-arbitrage bound
    if market_price - lower < 1e-6 or upper - market_price < 1e-6:
        pytest.skip("price too close to a no-arbitrage bound")

    recovered = implied_vol(market_price, S, K, T, r, opt, q)
    assert bsm_price(S, K, T, r, recovered, opt, q) == pytest.approx(market_price, abs=1e-8)

def test_simple_known_case():
    price = bsm_price(100, 100, 1.0, 0.05, 0.25, "call")
    assert implied_vol(price, 100, 100, 1.0, 0.05, "call") == pytest.approx(0.25, abs=1e-6)


def test_price_below_intrinsic_raises():
    # call intrinsic is about 100 - 90*exp(-0.05) = 14.4; a price of 5 is impossible
    with pytest.raises(ValueError):
        implied_vol(5.0, 100, 90, 1.0, 0.05, "call")


def test_price_above_upper_bound_raises():
    with pytest.raises(ValueError):
        implied_vol(101.0, 100, 100, 1.0, 0.05, "call")


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        implied_vol(10.0, 100, 100, 0.0, 0.05, "call")