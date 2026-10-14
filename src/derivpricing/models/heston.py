"""Heston (1993) stochastic-volatility model.

    dS = (r - q) S dt + sqrt(v) S dW_1
    dv = kappa (theta - v) dt + xi sqrt(v) dW_2,      d<W_1, W_2> = rho dt

The variance is a CIR process: it stays positive in continuous time iff the
Feller condition 2 kappa theta >= xi^2 holds, but an Euler step can still
produce v < 0. Three fixes (Lord, Koekkoek & van Dijk 2010) are implemented,
written as v_{k+1} = f1(v_k) + kappa (theta - f2(v_k)) dt + xi sqrt(f3(v_k)) dW:

=================  ======  ======  ======
scheme             f1      f2      f3
=================  ======  ======  ======
reflection         |v|     |v|     |v|
partial_truncation v       v       v^+
full_truncation    v       v^+     v^+      <- least biased, default
=================  ======  ======  ======

The stock is advanced with the log-Euler step
S_{k+1} = S_k exp((r - q - f3(v_k)/2) dt + sqrt(f3(v_k) dt) Z_1), which keeps S > 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from derivpricing.analytic.fourier import cf_price, heston_cf
from derivpricing.models.base import Model, Paths

_SCHEMES = {
    "full_truncation": (lambda v: v, lambda v: np.maximum(v, 0.0), lambda v: np.maximum(v, 0.0)),
    "partial_truncation": (lambda v: v, lambda v: v, lambda v: np.maximum(v, 0.0)),
    "reflection": (np.abs, np.abs, np.abs),
}


@dataclass
class Heston(Model):
    S0: float
    v0: float
    r: float
    kappa: float
    theta: float
    xi: float
    rho: float
    q: float = 0.0
    scheme: str = "full_truncation"

    n_factors = 2

    @property
    def feller(self) -> bool:
        """True if 2 kappa theta >= xi^2 (variance never touches zero in continuous time)."""
        return 2 * self.kappa * self.theta >= self.xi**2

    def simulate(self, T: float, Z: np.ndarray, rng=None) -> Paths:
        f1, f2, f3 = _SCHEMES[self.scheme]
        n_paths, n_steps, _ = Z.shape
        dt = T / n_steps
        sq = np.sqrt(dt)
        z1 = Z[..., 0]
        z2 = self.rho * Z[..., 0] + np.sqrt(1 - self.rho**2) * Z[..., 1]  # Cholesky correlation
        S = np.empty((n_paths, n_steps + 1))
        v = np.empty((n_paths, n_steps + 1))
        S[:, 0], v[:, 0] = self.S0, self.v0
        for k in range(n_steps):
            vk = v[:, k]
            v_diff = f3(vk)
            S[:, k + 1] = S[:, k] * np.exp((self.r - self.q - 0.5 * v_diff) * dt + np.sqrt(v_diff) * sq * z1[:, k])
            v[:, k + 1] = f1(vk) + self.kappa * (self.theta - f2(vk)) * dt + self.xi * np.sqrt(v_diff) * sq * z2[:, k]
        return Paths(self._grid(T, n_steps), S, {"v": v})

    def characteristic_function(self, T: float):
        return heston_cf(self.S0, self.r, T, self.v0, self.kappa, self.theta, self.xi, self.rho, self.q)

    def european_price(self, K, T, kind="call") -> float:
        """Semi-closed-form price by Fourier inversion of the Heston characteristic function."""
        return cf_price(self.characteristic_function(T), self.S0, K, self.r, T, kind, self.q)
