"""Low-discrepancy (quasi-random) sequences.

Quasi-Monte Carlo replaces pseudo-random points by a deterministic, evenly
spread sequence. For smooth integrands in d dimensions the integration error
is O((log N)^d / N) instead of the O(N^{-1/2}) of plain Monte Carlo.

* van der Corput (base b): the radical inverse phi_b(k) = sum_j a_j b^{-j-1}
  of k = sum_j a_j b^j.
* Halton: one van der Corput sequence per dimension with pairwise *coprime*
  bases (2, 3, 5, ...); non-coprime bases produce correlated, badly spread points.
* Sobol (via scipy): digital net in base 2 with Owen scrambling, which makes the
  estimator unbiased and lets independent scrambles give a standard error.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.stats import qmc as _qmc

PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)


def van_der_corput(n: int, base: int = 2, start: int = 1) -> np.ndarray:
    """First n radical-inverse points in [0, 1) (start=1 skips the point 0)."""
    k = np.arange(start, start + n)
    out = np.zeros(n)
    f = 1.0 / base
    while np.any(k > 0):
        out += f * (k % base)
        k //= base
        f /= base
    return out


def halton(n: int, dim: int = 2, bases=None, start: int = 1) -> np.ndarray:
    bases = bases or PRIMES[:dim]
    return np.column_stack([van_der_corput(n, b, start) for b in bases])


def sobol_normals(n: int, dim: int, seed=None) -> np.ndarray:
    """n scrambled-Sobol points in [0,1)^dim mapped to N(0, 1) by the inverse CDF."""
    m = int(np.ceil(np.log2(max(n, 2))))
    u = _qmc.Sobol(d=dim, scramble=True, seed=seed).random_base2(m)[:n]
    return stats.norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))


def qmc_integrate(f, n: int, dim: int, method: str = "halton", bases=None) -> float:
    """Estimate int_{[0,1]^dim} f(x) dx by an equal-weight average over a QMC point set."""
    pts = halton(n, dim, bases) if method == "halton" else _qmc.Sobol(dim, scramble=False).random(n)
    return float(np.mean(f(pts)))
