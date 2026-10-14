"""Merton (1976) jump-diffusion.

    dS / S_- = (r - q - lam kbar) dt + sigma dW + (Y - 1) dN,   N ~ Poisson(lam), ln Y ~ N(mu_j, sigma_j^2)

so that S_t = S_0 exp((r - q - lam kbar - sigma^2/2) t + sigma W_t) prod_{j <= N_t} Y_j and
kbar = E[Y - 1] = exp(mu_j + sigma_j^2 / 2) - 1 (the compensator that makes the discounted
price a martingale).

Simulation uses the exact "timeline" method: in each step the number of jumps
n ~ Poisson(lam dt) and, given n, the summed log-jump ~ N(n mu_j, n sigma_j^2).
The log-price increment is therefore sampled exactly, for any step size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from derivpricing.analytic.fourier import cf_price, merton_cf, merton_series_price
from derivpricing.models.base import Model, Paths


@dataclass
class MertonJD(Model):
    S0: float
    r: float
    sigma: float
    lam: float
    mu_j: float
    sigma_j: float
    q: float = 0.0

    n_factors = 1

    @property
    def kbar(self) -> float:
        return math.exp(self.mu_j + 0.5 * self.sigma_j**2) - 1.0

    def simulate(self, T: float, Z: np.ndarray, rng: np.random.Generator | None = None) -> Paths:
        rng = rng or np.random.default_rng()
        z = Z[..., 0] if Z.ndim == 3 else Z
        n_paths, n_steps = z.shape
        dt = T / n_steps
        n_jumps = rng.poisson(self.lam * dt, size=(n_paths, n_steps))
        jump = n_jumps * self.mu_j + np.sqrt(n_jumps) * self.sigma_j * rng.standard_normal((n_paths, n_steps))
        drift = (self.r - self.q - self.lam * self.kbar - 0.5 * self.sigma**2) * dt
        incr = drift + self.sigma * np.sqrt(dt) * z + jump
        logS = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(incr, axis=1)], axis=1)
        return Paths(self._grid(T, n_steps), self.S0 * np.exp(logS), {"n_jumps": n_jumps})

    def european_price(self, K, T, kind="call", method: str = "series") -> float:
        if method == "series":
            return merton_series_price(self.S0, K, self.r, self.sigma, T, self.lam, self.mu_j, self.sigma_j, kind, self.q)
        cf = merton_cf(self.S0, self.r, self.sigma, T, self.lam, self.mu_j, self.sigma_j, self.q)
        return cf_price(cf, self.S0, K, self.r, T, kind, self.q)
