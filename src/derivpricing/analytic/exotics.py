"""Closed forms for path-dependent options under Black-Scholes (continuous monitoring
unless stated). They serve as benchmarks and control variates for Monte Carlo."""

from __future__ import annotations

import math

import numpy as np

from derivpricing.analytic.black_scholes import bs_price
from derivpricing.analytic.normal import norm_cdf, norm_pdf

# Broadie-Glasserman-Kou constant: -zeta(1/2) / sqrt(2 pi)
BGK_BETA = 0.5825971579390106


def down_and_out_put(S, K, r, sigma, T, barrier):
    """Continuously monitored down-and-out put (requires barrier < min(S, K)).

    P_do = K e^{-rT}[N(d4) - N(d2) - a(N(d7) - N(d5))] - S[N(d3) - N(d1) - b(N(d8) - N(d6))],
    a = (H/S)^{2r/sigma^2 - 1}, b = (H/S)^{2r/sigma^2 + 1}.
    """
    if not barrier < min(S, K):
        raise ValueError("formula requires barrier < min(S, K)")
    H, st = barrier, sigma * math.sqrt(T)
    lam_m, lam_p = (r - 0.5 * sigma**2) * T, (r + 0.5 * sigma**2) * T
    d1 = (math.log(S / K) + lam_p) / st
    d2 = (math.log(S / K) + lam_m) / st
    d3 = (math.log(S / H) + lam_p) / st
    d4 = (math.log(S / H) + lam_m) / st
    d5 = (math.log(S / H) - lam_m) / st
    d6 = (math.log(S / H) - lam_p) / st
    d7 = (math.log(S * K / H**2) - lam_m) / st
    d8 = (math.log(S * K / H**2) - lam_p) / st
    a = (H / S) ** (2 * r / sigma**2 - 1)
    b = (H / S) ** (2 * r / sigma**2 + 1)
    N = norm_cdf
    return float(K * math.exp(-r * T) * (N(d4) - N(d2) - a * (N(d7) - N(d5)))
                 - S * (N(d3) - N(d1) - b * (N(d8) - N(d6))))


def bgk_shift(barrier: float, sigma: float, T: float, n_monitor: int, direction: str = "down") -> float:
    """Continuity correction: a discrete barrier H behaves like a continuous one at
    H exp(-/+ beta sigma sqrt(T/m)) (Broadie, Glasserman & Kou 1997)."""
    shift = BGK_BETA * sigma * math.sqrt(T / n_monitor)
    return barrier * math.exp(-shift if direction == "down" else shift)


def geometric_asian(S, K, r, sigma, T, n_monitor: int, kind: str = "call", q: float = 0.0) -> float:
    """Discretely monitored fixed-strike geometric Asian (average over t_1..t_m, t_j = jT/m).

    ln G is normal with mean ln S + (r - q - sigma^2/2) mean(t_j) and variance
    sigma^2/m^2 sum_{i,j} min(t_i, t_j); price = e^{-rT} E[(G - K)^+] in closed form.
    """
    t = np.arange(1, n_monitor + 1) * T / n_monitor
    mean = math.log(S) + (r - q - 0.5 * sigma**2) * t.mean()
    var = sigma**2 * np.minimum.outer(t, t).sum() / n_monitor**2
    sd = math.sqrt(var)
    forward = math.exp(mean + 0.5 * var)  # E[G]
    d1 = (math.log(forward / K) + 0.5 * var) / sd
    d2 = d1 - sd
    if kind == "call":
        return float(math.exp(-r * T) * (forward * norm_cdf(d1) - K * norm_cdf(d2)))
    return float(math.exp(-r * T) * (K * norm_cdf(-d2) - forward * norm_cdf(-d1)))


def floating_lookback_call(S, r, sigma, T, s_min: float | None = None, q: float = 0.0) -> float:
    """Continuously monitored floating-strike lookback call, payoff S_T - min S (Goldman-Sosin-Gatto)."""
    m = S if s_min is None else s_min
    st = sigma * math.sqrt(T)
    a1 = (math.log(S / m) + (r - q + 0.5 * sigma**2) * T) / st
    a2 = a1 - st
    if abs(r - q) < 1e-12:
        return float(S * math.exp(-r * T) * norm_cdf(a1) - m * math.exp(-r * T) * norm_cdf(a2)
                     + S * math.exp(-r * T) * st * (norm_pdf(a1) + a1 * (norm_cdf(a1) - 1.0)))
    a3 = (math.log(S / m) + (-r + q + 0.5 * sigma**2) * T) / st
    k = sigma**2 / (2 * (r - q))
    return float(S * math.exp(-q * T) * norm_cdf(a1) - m * math.exp(-r * T) * norm_cdf(a2)
                 - S * math.exp(-q * T) * k * norm_cdf(-a1)
                 + S * math.exp(-r * T) * k * (S / m) ** (-2 * (r - q) / sigma**2) * norm_cdf(-a3))


def down_and_in_put(S, K, r, sigma, T, barrier) -> float:
    """In-out parity: down-and-in = vanilla - down-and-out."""
    return float(bs_price(S, K, r, sigma, T, "put") - down_and_out_put(S, K, r, sigma, T, barrier))
