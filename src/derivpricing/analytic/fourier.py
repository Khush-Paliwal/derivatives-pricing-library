"""Characteristic-function pricing for European options.

A model enters only through phi(u) = E[exp(i u ln S_T)], the characteristic
function of the log terminal price under the risk-neutral measure. Two
inversion methods are provided:

* :func:`cf_price` - Gil-Pelaez / Heston (1993) probabilities P1, P2 by
  numerical quadrature (accurate, one strike at a time).
* :func:`carr_madan_fft` - Carr & Madan (1999) damped call transform on an FFT
  grid (a whole strike surface in O(N log N)).

Characteristic functions for Black-Scholes, Merton jump-diffusion and Heston
(the numerically stable "little Heston trap" form of Albrecher et al. 2007)
are included.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from scipy import integrate

CharFn = Callable[[np.ndarray], np.ndarray]


# --------------------------------------------------------------------------- #
# Characteristic functions of ln S_T
# --------------------------------------------------------------------------- #
def bs_cf(S, r, sigma, T, q=0.0) -> CharFn:
    mu = math.log(S) + (r - q - 0.5 * sigma**2) * T
    return lambda u: np.exp(1j * u * mu - 0.5 * sigma**2 * u**2 * T)


def merton_cf(S, r, sigma, T, lam, mu_j, sigma_j, q=0.0) -> CharFn:
    """Merton (1976): lognormal jumps ln Y ~ N(mu_j, sigma_j^2) at Poisson rate lam.

    The drift carries the compensator -lam * kbar, kbar = E[Y - 1], so that the
    discounted price is a martingale.
    """
    kbar = math.exp(mu_j + 0.5 * sigma_j**2) - 1.0
    mu = math.log(S) + (r - q - lam * kbar - 0.5 * sigma**2) * T
    return lambda u: np.exp(1j * u * mu - 0.5 * sigma**2 * u**2 * T
                            + lam * T * (np.exp(1j * u * mu_j - 0.5 * sigma_j**2 * u**2) - 1.0))


def heston_cf(S, r, T, v0, kappa, theta, xi, rho, q=0.0) -> CharFn:
    """Heston (1993) with dv = kappa (theta - v) dt + xi sqrt(v) dW_v, d<W_S, W_v> = rho dt.

    Uses the "little trap" branch (g = (b - d) / (b + d) with e^{-dT}) that avoids
    the complex-log discontinuity of the original formulation.
    """
    x0 = math.log(S) + (r - q) * T

    def phi(u):
        u = np.asarray(u, dtype=complex)
        b = kappa - rho * xi * 1j * u
        d = np.sqrt(b**2 + xi**2 * (1j * u + u**2))
        g = (b - d) / (b + d)
        e = np.exp(-d * T)
        C = kappa * theta / xi**2 * ((b - d) * T - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
        D = (b - d) / xi**2 * (1.0 - e) / (1.0 - g * e)
        return np.exp(1j * u * x0 + C + D * v0)

    return phi


# --------------------------------------------------------------------------- #
# Inversion
# --------------------------------------------------------------------------- #
def cf_price(cf: CharFn, S, K, r, T, kind: str = "call", q: float = 0.0, upper: float = 500.0) -> float:
    """European price from the characteristic function via Gil-Pelaez inversion.

    C = S e^{-qT} P1 - K e^{-rT} P2 with
    P2 = 1/2 + 1/pi int_0^inf Re[e^{-iu ln K} phi(u) / (iu)] du       (exercise probability)
    P1 = 1/2 + 1/pi int_0^inf Re[e^{-iu ln K} phi(u - i) / (iu phi(-i))] du  (stock-measure probability)
    """
    lnK = math.log(K)
    fwd = cf(np.array(-1j))  # phi(-i) = E[S_T] = S e^{(r-q)T}

    def integrand2(u):
        return (np.exp(-1j * u * lnK) * cf(np.array(u)) / (1j * u)).real

    def integrand1(u):
        return (np.exp(-1j * u * lnK) * cf(np.array(u - 1j)) / (1j * u * fwd)).real

    p1 = 0.5 + integrate.quad(integrand1, 1e-10, upper, limit=500)[0] / math.pi
    p2 = 0.5 + integrate.quad(integrand2, 1e-10, upper, limit=500)[0] / math.pi
    call = S * math.exp(-q * T) * p1 - K * math.exp(-r * T) * p2
    if kind == "call":
        return float(call)
    return float(call - S * math.exp(-q * T) + K * math.exp(-r * T))  # put-call parity


def carr_madan_fft(cf: CharFn, S, r, T, alpha: float = 1.5, n: int = 4096, eta: float = 0.25):
    """Call prices on a log-strike grid via the Carr-Madan FFT.

    The damped call c(k) e^{alpha k} is square-integrable; its Fourier transform is
    psi(v) = e^{-rT} phi(v - (alpha + 1) i) / (alpha^2 + alpha - v^2 + i(2 alpha + 1) v).
    Simpson weights are used for the frequency integral.

    Returns (strikes, call_prices).
    """
    v = np.arange(n) * eta
    lam = 2 * math.pi / (n * eta)
    b = n * lam / 2
    k = -b + lam * np.arange(n) + math.log(S)  # centre the strike grid on S
    psi = math.exp(-r * T) * cf(v - (alpha + 1) * 1j) / (alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v)
    simpson = (3 + (-1) ** (np.arange(n) + 1)) / 3.0
    simpson[0] = 1 / 3.0
    x = np.exp(1j * v * (b - math.log(S))) * psi * eta * simpson
    calls = (np.exp(-alpha * k) / math.pi) * np.fft.fft(x).real
    return np.exp(k), calls


def fft_price(cf: CharFn, S, K, r, T, **kw) -> float:
    """Single-strike price by cubic-spline interpolation of the Carr-Madan FFT grid in log-strike.

    (Linear interpolation between grid strikes ~0.6% apart costs ~5e-4 in price; a local
    cubic spline brings the error down to the FFT's own ~1e-7.)
    """
    from scipy.interpolate import CubicSpline

    strikes, calls = carr_madan_fft(cf, S, r, T, **kw)
    k = np.log(strikes)
    i = int(np.searchsorted(k, math.log(K)))
    window = slice(max(i - 8, 0), min(i + 8, k.size))
    return float(CubicSpline(k[window], calls[window])(math.log(K)))


def merton_series_price(S, K, r, sigma, T, lam, mu_j, sigma_j, kind="call", q=0.0, n_terms: int = 80) -> float:
    """Merton's Poisson-weighted Black-Scholes series.

    price = sum_k e^{-lam' T}(lam' T)^k / k! * BS(sigma_k, r_k),
    lam' = lam (1 + kbar), sigma_k^2 = sigma^2 + k sigma_j^2 / T,
    r_k = r - lam kbar + k ln(1 + kbar) / T.
    """
    from derivpricing.analytic.black_scholes import bs_price

    if lam == 0:
        return float(bs_price(S, K, r, sigma, T, kind, q))
    kbar = math.exp(mu_j + 0.5 * sigma_j**2) - 1.0
    lam_p = lam * (1 + kbar)
    total = 0.0
    log_w = -lam_p * T
    for k in range(n_terms):
        if k > 0:
            log_w += math.log(lam_p * T) - math.log(k)
        sig_k = math.sqrt(sigma**2 + k * sigma_j**2 / T)
        r_k = r - lam * kbar + k * math.log(1 + kbar) / T
        total += math.exp(log_w) * bs_price(S, K, r_k, sig_k, T, kind, q)
    return float(total)
