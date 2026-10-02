import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np

from optionpricer import bsm_price, greeks, implied_vol

OUT = ROOT / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)

K, r, sigma, q = 100.0, 0.05, 0.20, 0.0
spots = np.linspace(60, 140, 200)


# Figure 1: option value vs spot, compared with the payoff at expiry
def plot_prices():
    T = 1.0
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, opt in zip(axes, ["call", "put"]):
        value = [bsm_price(s, K, T, r, sigma, opt, q) for s in spots]
        payoff = np.maximum(spots - K, 0) if opt == "call" else np.maximum(K - spots, 0)
        ax.plot(spots, value, label=f"BSM value (T = {T}y)")
        ax.plot(spots, payoff, "--", color="gray", label="Payoff at expiry")
        ax.axvline(K, color="lightgray", lw=0.8)
        ax.set_title(f"European {opt}")
        ax.set_xlabel("Spot price")
        ax.set_ylabel("Option value")
        ax.legend()
    fig.suptitle(f"Price vs spot  (K={K:.0f}, r={r:.0%}, sigma={sigma:.0%})")
    fig.tight_layout()
    fig.savefig(OUT / "price_vs_spot.png", dpi=150)
    plt.close(fig)


# Figure 2: how the Greeks change with spot, for three expiries
def plot_greeks():
    expiries = [0.05, 0.25, 1.0]
    panels = [
        ("delta", "Delta", 1.0),
        ("gamma", "Gamma", 1.0),
        ("vega", "Vega (per 1 vol point)", 100.0),
        ("theta", "Theta (per day)", 365.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (key, title, scale) in zip(axes.ravel(), panels):
        for T in expiries:
            vals = [greeks(s, K, T, r, sigma, "call", q)[key] / scale for s in spots]
            ax.plot(spots, vals, label=f"T = {T}y")
        ax.set_title(title)
        ax.set_xlabel("Spot price")
        ax.legend()
    fig.suptitle("Call Greeks vs spot: note how gamma sharpens near expiry")
    fig.tight_layout()
    fig.savefig(OUT / "greeks_vs_spot.png", dpi=150)
    plt.close(fig)


# Figure 3: implied vol solver on a synthetic smile
def plot_smile():
    T = 0.5
    strikes = np.linspace(70, 130, 25)
    true_vol = 0.20 + 0.50 * (strikes / 100 - 1) ** 2  # made-up smile shape
    recovered = []
    for k, v in zip(strikes, true_vol):
        opt = "put" if k < 100 else "call"  # out-of-the-money option is better conditioned
        price = bsm_price(100.0, float(k), T, r, float(v), opt, q)
        recovered.append(implied_vol(price, 100.0, float(k), T, r, opt, q))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(strikes, true_vol, label="Volatility used to make the prices")
    ax.plot(strikes, recovered, "o", ms=5, label="Recovered by the solver")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied volatility")
    ax.set_title("Implied vol round trip on a synthetic smile")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "implied_vol_smile.png", dpi=150)
    plt.close(fig)

    worst = max(abs(a - b) for a, b in zip(true_vol, recovered))
    print(f"Largest gap between true and recovered vol: {worst:.2e}")


if __name__ == "__main__":
    plot_prices()
    plot_greeks()
    plot_smile()
    print(f"Saved figures to {OUT}")