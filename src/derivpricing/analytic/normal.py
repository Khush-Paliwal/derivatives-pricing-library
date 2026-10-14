"""Standard normal CDF: exact (scipy) and two classic rational approximations.

The approximations are what one writes when no statistics library is
available; they are kept here (and tested against scipy) because the
course and many interview exercises ask for them explicitly.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

SQRT_2PI = math.sqrt(2 * math.pi)


def norm_pdf(x):
    return np.exp(-0.5 * np.asarray(x, float) ** 2) / SQRT_2PI


def norm_cdf(x):
    """Exact standard normal CDF (scipy)."""
    return stats.norm.cdf(x)


_AS_COEFS = (0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429)


def norm_cdf_as(x):
    """Abramowitz & Stegun 26.2.17 (as in Hull): |error| < 7.5e-8.

    N(x) ~ 1 - n(x) (a1 k + a2 k^2 + ... + a5 k^5),  k = 1 / (1 + 0.2316419 x),  x >= 0.
    """
    x = np.asarray(x, float)
    k = 1.0 / (1.0 + 0.2316419 * np.abs(x))
    poly = sum(a * k ** (i + 1) for i, a in enumerate(_AS_COEFS))
    upper = 1.0 - norm_pdf(np.abs(x)) * poly
    return np.where(x >= 0, upper, 1.0 - upper)


_D_COEFS = (0.0498673470, 0.0211410061, 0.0032776263, 0.0000380036, 0.0000488906, 0.0000053830)


def norm_cdf_poly16(x):
    """Abramowitz & Stegun 26.2.19: N(x) ~ 1 - 0.5 (1 + d1 x + ... + d6 x^6)^-16, |error| < 1.5e-7."""
    x = np.asarray(x, float)
    ax = np.abs(x)
    poly = 1.0 + sum(d * ax ** (i + 1) for i, d in enumerate(_D_COEFS))
    upper = 1.0 - 0.5 * poly**-16
    return np.where(x >= 0, upper, 1.0 - upper)
