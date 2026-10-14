"""Black-Scholes-Merton closed forms: prices, Greeks and implied volatility.

Underlying dS = (r - q) S dt + sigma S dW under the risk-neutral measure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from derivpricing.analytic.normal import norm_cdf, norm_pdf


def _d1_d2(S, K, r, sigma, T, q=0.0):
    sqrt_t = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_t)
    return d1, d1 - sigma * sqrt_t


def bs_price(S, K, r, sigma, T, kind: str = "call", q: float = 0.0, cdf=norm_cdf):
    """European call/put price. Vectorised over any argument; T <= 0 returns intrinsic value."""
    S, K, T = np.asarray(S, float), np.asarray(K, float), np.asarray(T, float)
    intrinsic = np.maximum(S - K, 0.0) if kind == "call" else np.maximum(K - S, 0.0)
    T_safe = np.where(T > 0, T, 1.0)
    d1, d2 = _d1_d2(S, K, r, sigma, T_safe, q)
    df_r, df_q = np.exp(-r * T_safe), np.exp(-q * T_safe)
    if kind == "call":
        price = S * df_q * cdf(d1) - K * df_r * cdf(d2)
    elif kind == "put":
        price = K * df_r * cdf(-d2) - S * df_q * cdf(-d1)
    else:
        raise ValueError("kind must be 'call' or 'put'")
    out = np.where(T > 0, price, intrinsic)
    return float(out) if out.ndim == 0 else out


@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    vega: float
    theta: float  # dV/dt (calendar time): negative for a long call
    rho: float


def bs_greeks(S, K, r, sigma, T, kind: str = "call", q: float = 0.0) -> Greeks:
    """Analytic Black-Scholes Greeks (vega per unit sigma, theta per year)."""
    d1, d2 = _d1_d2(S, K, r, sigma, T, q)
    df_r, df_q, sqrt_t = math.exp(-r * T), math.exp(-q * T), math.sqrt(T)
    gamma = df_q * norm_pdf(d1) / (S * sigma * sqrt_t)
    vega = S * df_q * norm_pdf(d1) * sqrt_t
    decay = -S * df_q * norm_pdf(d1) * sigma / (2 * sqrt_t)
    if kind == "call":
        delta = df_q * norm_cdf(d1)
        theta = decay - r * K * df_r * norm_cdf(d2) + q * S * df_q * norm_cdf(d1)
        rho = K * T * df_r * norm_cdf(d2)
    else:
        delta = df_q * (norm_cdf(d1) - 1.0)
        theta = decay + r * K * df_r * norm_cdf(-d2) - q * S * df_q * norm_cdf(-d1)
        rho = -K * T * df_r * norm_cdf(-d2)
    return Greeks(float(delta), float(gamma), float(vega), float(theta), float(rho))


def implied_vol(price, S, K, r, T, kind: str = "call", q: float = 0.0, lo: float = 1e-6, hi: float = 5.0) -> float:
    """Implied volatility by Brent's method (price is monotone increasing in sigma)."""
    return brentq(lambda s: bs_price(S, K, r, s, T, kind, q) - price, lo, hi, xtol=1e-12)


def digital_price(S, K, r, sigma, T, kind: str = "call", q: float = 0.0) -> float:
    """Cash-or-nothing digital paying 1: e^{-rT} N(+/- d2)."""
    _, d2 = _d1_d2(S, K, r, sigma, T, q)
    return float(math.exp(-r * T) * norm_cdf(d2 if kind == "call" else -d2))
