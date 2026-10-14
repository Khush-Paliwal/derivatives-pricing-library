"""Common interface for path-simulating models.

A model never draws its own Gaussian shocks: the Monte Carlo engine hands it a
standard-normal array ``Z`` of shape (n_paths, n_steps, n_factors). Antithetic
variates, quasi-random (Sobol) sampling and common random numbers for Greeks
are therefore implemented once, in the engine, for every model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Paths:
    """Simulated paths on a uniform grid t_0 = 0 < t_1 < ... < t_m = T.

    S has shape (n_paths, m + 1) with S[:, 0] = S_0. ``state`` holds any extra
    state variables (e.g. Heston variance ``v``) with the same shape.
    """

    t: np.ndarray
    S: np.ndarray
    state: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0])

    @property
    def T(self) -> float:
        return float(self.t[-1])

    @property
    def terminal(self) -> np.ndarray:
        return self.S[:, -1]


class Model:
    """Base class: subclasses define ``n_factors``, ``r``, ``q``, ``S0`` and ``simulate``."""

    n_factors: int = 1
    S0: float
    r: float
    q: float = 0.0

    def simulate(self, T: float, Z: np.ndarray, rng: np.random.Generator | None = None) -> Paths:
        raise NotImplementedError

    def discount(self, T: float) -> float:
        return float(np.exp(-self.r * T))

    @staticmethod
    def _grid(T: float, n_steps: int) -> np.ndarray:
        return np.linspace(0.0, T, n_steps + 1)
