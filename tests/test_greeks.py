import math

import pytest

from optionpricer import bsm_price, greeks

CASES = [
    (100, 100, 1.0, 0.05, 0.20, 0.00),
    (100, 90, 0.5, 0.03, 0.30, 0.02),
    (80, 100, 2.0, 0.01, 0.40, 0.00),
    (120, 100, 0.25, 0.07, 0.15, 0.04),
]


def price(S, K, T, r, sigma, opt, q):
    return bsm_price(S, K, T, r, sigma, opt, q)


@pytest.mark.parametrize("opt", ["call", "put"])
@pytest.mark.parametrize("S, K, T, r, sigma, q", CASES)
def test_greeks_match_finite_differences(S, K, T, r, sigma, q, opt):
    g = greeks(S, K, T, r, sigma, opt, q)
    p0 = price(S, K, T, r, sigma, opt, q)

    h = 0.01
    delta = (price(S + h, K, T, r, sigma, opt, q) - price(S - h, K, T, r, sigma, opt, q)) / (2 * h)
    gamma = (price(S + h, K, T, r, sigma, opt, q) - 2 * p0 + price(S - h, K, T, r, sigma, opt, q)) / h**2

    h = 1e-5
    vega = (price(S, K, T, r, sigma + h, opt, q) - price(S, K, T, r, sigma - h, opt, q)) / (2 * h)
    rho = (price(S, K, T, r + h, sigma, opt, q) - price(S, K, T, r - h, sigma, opt, q)) / (2 * h)
    # theta is the derivative w.r.t. calendar time, so theta = -dV/dT
    theta = -(price(S, K, T + h, r, sigma, opt, q) - price(S, K, T - h, r, sigma, opt, q)) / (2 * h)

    assert g["delta"] == pytest.approx(delta, rel=1e-5, abs=1e-7)
    assert g["gamma"] == pytest.approx(gamma, rel=1e-4, abs=1e-7)
    assert g["vega"] == pytest.approx(vega, rel=1e-5, abs=1e-6)
    assert g["rho"] == pytest.approx(rho, rel=1e-5, abs=1e-6)
    assert g["theta"] == pytest.approx(theta, rel=1e-5, abs=1e-6)


@pytest.mark.parametrize("S, K, T, r, sigma, q", CASES)
def test_call_put_relationships(S, K, T, r, sigma, q):
    c = greeks(S, K, T, r, sigma, "call", q)
    p = greeks(S, K, T, r, sigma, "put", q)
    assert c["gamma"] == pytest.approx(p["gamma"])
    assert c["vega"] == pytest.approx(p["vega"])
    assert c["delta"] - p["delta"] == pytest.approx(math.exp(-q * T))


def test_delta_ranges():
    c = greeks(100, 100, 1.0, 0.05, 0.2, "call")
    p = greeks(100, 100, 1.0, 0.05, 0.2, "put")
    assert 0 < c["delta"] < 1
    assert -1 < p["delta"] < 0
    assert c["gamma"] > 0 and c["vega"] > 0


def test_greeks_reject_expired_option():
    with pytest.raises(ValueError):
        greeks(100, 100, 0.0, 0.05, 0.2, "call")