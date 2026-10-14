"""Geometric Brownian motion (Black-Scholes dynamics).

dS = (r - q) S dt + sigma S dW. Three discretisations are offered so that the
effect of discretisation error can be studied against the exact solution:

* ``exact``    S_{k+1} = S_k exp((r - q - sigma^2/2) dt + sigma sqrt(dt) Z)   (no bias)
* ``euler``    S_{k+1} = S_k (1 + (r - q) dt + sigma sqrt(dt) Z)               (strong order 1/2)
* ``milstein`` euler + 1/2 sigma^2 S_k (dW^2 - dt)                              (strong order 1)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from derivpricing.analytic.black_scholes import bs_price
from derivpricing.models.base import Model, Paths


@dataclass
class GBM(Model):
    S0: float
    r: float
    sigma: float
    q: float = 0.0
    scheme: str = "exact"

    n_factors = 1

    def simulate(self, T: float, Z: np.ndarray, rng=None) -> Paths:
        n_paths, n_steps = Z.shape[:2]
        z = Z[..., 0] if Z.ndim == 3 else Z
        dt = T / n_steps
        t = self._grid(T, n_steps)
        if self.scheme == "exact":
            incr = (self.r - self.q - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * z
            logS = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(incr, axis=1)], axis=1)
            return Paths(t, self.S0 * np.exp(logS))
        S = np.empty((n_paths, n_steps + 1))
        S[:, 0] = self.S0
        for k in range(n_steps):
            dW = np.sqrt(dt) * z[:, k]
            S[:, k + 1] = S[:, k] * (1 + (self.r - self.q) * dt + self.sigma * dW)
            if self.scheme == "milstein":
                S[:, k + 1] += 0.5 * self.sigma**2 * S[:, k] * (dW**2 - dt)
            elif self.scheme != "euler":
                raise ValueError("scheme must be 'exact', 'euler' or 'milstein'")
        return Paths(t, S)

    def european_price(self, K, T, kind="call") -> float:
        return bs_price(self.S0, K, self.r, self.sigma, T, kind, self.q)
