import math

import pytest

from optionpricer import bsm_price


def test_hull_textbook_example():
    # Hull: S=42, K=40, r=10%, sigma=20%, T=0.5 -> call 4.76, put 0.81
    call = bsm_price(42, 40, 0.5, 0.10, 0.20, "call")
    put = bsm_price(42, 40, 0.5, 0.10, 0.20, "put")
    assert call == pytest.approx(4.759, abs=1e-3)
    assert put == pytest.approx(0.809, abs=1e-3)


def test_atm_one_year_reference():
    # S=K=100, T=1, r=5%, sigma=20%
    call = bsm_price(100, 100, 1.0, 0.05, 0.20, "call")
    put = bsm_price(100, 100, 1.0, 0.05, 0.20, "put")
    assert call == pytest.approx(10.4506, abs=1e-4)
    assert put == pytest.approx(5.5735, abs=1e-4)


# Put-call parity

@pytest.mark.parametrize(
    "S, K, T, r, sigma, q",
    [
        (100, 100, 1.0, 0.05, 0.20, 0.00),
        (100, 90, 0.5, 0.03, 0.30, 0.02),
        (50, 80, 2.0, 0.01, 0.40, 0.00),
        (120, 100, 0.1, 0.07, 0.15, 0.04),
    ],
)
def test_put_call_parity(S, K, T, r, sigma, q):
    call = bsm_price(S, K, T, r, sigma, "call", q)
    put = bsm_price(S, K, T, r, sigma, "put", q)
    lhs = call - put
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert lhs == pytest.approx(rhs, abs=1e-10)


# No-arbitrage bounds

@pytest.mark.parametrize(
    "S, K, T, r, sigma, q",
    [
        (100, 100, 1.0, 0.05, 0.20, 0.00),
        (100, 70, 0.5, 0.03, 0.30, 0.02),
        (60, 100, 2.0, 0.01, 0.40, 0.00),
    ],
)
def test_price_bounds(S, K, T, r, sigma, q):
    disc_S = S * math.exp(-q * T)
    disc_K = K * math.exp(-r * T)
    call = bsm_price(S, K, T, r, sigma, "call", q)
    put = bsm_price(S, K, T, r, sigma, "put", q)

    assert max(disc_S - disc_K, 0) <= call <= disc_S
    assert max(disc_K - disc_S, 0) <= put <= disc_K


# Monotonicity

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_price_increases_with_volatility(option_type):
    vols = [0.05, 0.10, 0.20, 0.40, 0.80]
    prices = [bsm_price(100, 100, 1.0, 0.05, v, option_type) for v in vols]
    assert prices == sorted(prices)


def test_call_increases_and_put_decreases_with_spot():
    spots = [80, 90, 100, 110, 120]
    calls = [bsm_price(s, 100, 1.0, 0.05, 0.2, "call") for s in spots]
    puts = [bsm_price(s, 100, 1.0, 0.05, 0.2, "put") for s in spots]
    assert calls == sorted(calls)
    assert puts == sorted(puts, reverse=True)


# ---------- Edge cases ----------

def test_expiry_returns_intrinsic_value():
    assert bsm_price(110, 100, 0.0, 0.05, 0.2, "call") == pytest.approx(10.0)
    assert bsm_price(90, 100, 0.0, 0.05, 0.2, "call") == 0.0
    assert bsm_price(90, 100, 0.0, 0.05, 0.2, "put") == pytest.approx(10.0)


def test_zero_volatility_is_discounted_forward_payoff():
    S, K, T, r = 100, 90, 1.0, 0.05
    expected = S - K * math.exp(-r * T)
    assert bsm_price(S, K, T, r, 0.0, "call") == pytest.approx(expected)


def test_tiny_volatility_matches_zero_volatility():
    zero = bsm_price(100, 90, 1.0, 0.05, 0.0, "call")
    tiny = bsm_price(100, 90, 1.0, 0.05, 1e-8, "call")
    assert tiny == pytest.approx(zero, abs=1e-6)


# ---------- Input validation ----------

@pytest.mark.parametrize(
    "kwargs",
    [
        dict(S=-1, K=100, T=1, r=0.05, sigma=0.2),
        dict(S=100, K=0, T=1, r=0.05, sigma=0.2),
        dict(S=100, K=100, T=-1, r=0.05, sigma=0.2),
        dict(S=100, K=100, T=1, r=0.05, sigma=-0.2),
    ],
)
def test_invalid_inputs_raise(kwargs):
    with pytest.raises(ValueError):
        bsm_price(**kwargs)


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        bsm_price(100, 100, 1, 0.05, 0.2, "straddle")