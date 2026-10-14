"""Monte Carlo estimation statistics shared by every pricer.

All Monte Carlo prices in the library are returned as an :class:`MCResult`, so
a price always travels with its standard error and confidence interval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class MCResult:
    """A Monte Carlo estimate.

    Attributes
    ----------
    price : sample mean of the (discounted) payoff samples
    stderr : standard error, s / sqrt(n) with s the ddof=1 sample std
    n : number of *independent* samples (antithetic pairs count once)
    """

    price: float
    stderr: float
    n: int

    def ci(self, level: float = 0.95) -> tuple[float, float]:
        z = stats.norm.ppf(0.5 + level / 2)
        return self.price - z * self.stderr, self.price + z * self.stderr

    def z_score(self, reference: float) -> float:
        """(price - reference) / stderr: how many standard errors from a benchmark."""
        return (self.price - reference) / self.stderr if self.stderr > 0 else math.inf

    def __float__(self) -> float:
        return self.price

    def __repr__(self) -> str:
        lo, hi = self.ci()
        return f"MCResult(price={self.price:.6f}, stderr={self.stderr:.6f}, n={self.n:,}, 95% CI=[{lo:.6f}, {hi:.6f}])"


def mc_estimate(samples: np.ndarray) -> MCResult:
    """Sample mean and standard error of i.i.d. samples."""
    x = np.asarray(samples, dtype=float).ravel()
    return MCResult(float(x.mean()), float(x.std(ddof=1) / math.sqrt(x.size)), int(x.size))


def required_paths(sample_std: float, tolerance: float, level: float = 0.95) -> int:
    """Smallest n with z * s / sqrt(n) <= tolerance  (n = ceil((z s / eps)^2))."""
    z = stats.norm.ppf(0.5 + level / 2)
    return int(math.ceil((z * sample_std / tolerance) ** 2))


def control_variate(target: np.ndarray, control: np.ndarray, control_mean: float) -> tuple[MCResult, float]:
    """Control-variate estimator T = X - b (Y - E[Y]) with the variance-optimal b.

    b* = Cov(X, Y) / Var(Y) is estimated from the same sample (a negligible
    O(1/n) bias). Returns the estimate and the variance-reduction factor
    Var(T) / Var(X) = 1 - rho^2.
    """
    x = np.asarray(target, float).ravel()
    y = np.asarray(control, float).ravel()
    cov = np.cov(x, y, ddof=1)
    b = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0
    adjusted = x - b * (y - control_mean)
    return mc_estimate(adjusted), float(adjusted.var(ddof=1) / x.var(ddof=1)) if x.var() > 0 else 1.0
