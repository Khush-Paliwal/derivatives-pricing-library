"""Monte Carlo pricing engine with variance reduction.

    price = e^{-rT} E^Q[payoff(paths)] ~ e^{-rT} (1/n) sum_i payoff_i,   stderr = s / sqrt(n)

Supported techniques (combinable):

* **Antithetic variates** - simulate Z and -Z and average each pair. For a
  payoff monotone in Z the pair members are negatively correlated, so
  Var((X + X')/2) = (Var X + Cov(X, X'))/2 < Var(X)/2.
* **Control variates** - subtract b (Y - E[Y]) for a correlated Y with known
  mean: the discounted terminal price (E = S_0 e^{-qT}) or any payoff with a
  closed form (e.g. a geometric Asian for an arithmetic Asian).
* **Randomised quasi-Monte Carlo** - scrambled Sobol normals; the standard error
  comes from independent scrambles (n_reps replications).

The same normals can be reused across calls (``Z=``) to get common random
numbers for finite-difference Greeks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from derivpricing.models.base import Model, Paths
from derivpricing.qmc import sobol_normals
from derivpricing.stats import MCResult, control_variate, mc_estimate


def draw_normals(n_paths: int, n_steps: int, n_factors: int, rng: np.random.Generator,
                 antithetic: bool = False, qmc: bool = False) -> np.ndarray:
    """Standard normals of shape (n_paths, n_steps, n_factors); antithetic stacks [Z; -Z]."""
    half = n_paths // 2 if antithetic else n_paths
    if qmc:
        z = sobol_normals(half, n_steps * n_factors, seed=rng).reshape(half, n_steps, n_factors)
    else:
        z = rng.standard_normal((half, n_steps, n_factors))
    return np.concatenate([z, -z]) if antithetic else z


@dataclass
class ControlVariate:
    """A control: a function of the paths (already discounted) and its known expectation."""

    fn: Callable[[Paths], np.ndarray]
    mean: float

    @classmethod
    def underlying(cls, model: Model, T: float) -> ControlVariate:
        """Y = e^{-rT} S_T with E[Y] = S_0 e^{-qT} (a martingale under Q)."""
        disc = model.discount(T)
        return cls(lambda p: disc * p.terminal, model.S0 * np.exp(-model.q * T))


def _pair_average(x: np.ndarray, antithetic: bool) -> np.ndarray:
    if not antithetic:
        return x
    h = x.size // 2
    return 0.5 * (x[:h] + x[h:])


def price(model: Model, payoff: Callable[[Paths], np.ndarray], T: float, n_paths: int = 100_000,
          n_steps: int = 1, seed: int | None = 0, antithetic: bool = True,
          control: ControlVariate | None = None, qmc: bool = False, n_reps: int = 16,
          Z: np.ndarray | None = None) -> MCResult:
    """Discounted Monte Carlo price of ``payoff`` under ``model``.

    Parameters
    ----------
    n_paths : total paths (antithetic pairs count as two paths but one sample).
    n_steps : time steps (1 is exact for European payoffs under GBM/Merton).
    Z : optional pre-drawn normals (common random numbers); overrides n_paths/n_steps.
    """
    rng = np.random.default_rng(seed)
    disc = model.discount(T)

    if qmc:  # randomised QMC: independent scrambles -> standard error across replications
        estimates = [
            price(model, payoff, T, n_paths // n_reps, n_steps, seed=int(rng.integers(2**31)),
                  antithetic=antithetic, control=control, qmc=False,
                  Z=draw_normals(n_paths // n_reps, n_steps, model.n_factors, rng, antithetic, qmc=True)).price
            for _ in range(n_reps)
        ]
        est = mc_estimate(np.array(estimates))
        return MCResult(est.price, est.stderr, n_paths)

    if Z is None:
        Z = draw_normals(n_paths, n_steps, model.n_factors, rng, antithetic)
    paths = model.simulate(T, Z, rng)
    x = _pair_average(disc * payoff(paths), antithetic)
    if control is None:
        return mc_estimate(x)
    y = _pair_average(control.fn(paths), antithetic)
    return control_variate(x, y, control.mean)[0]


def simulate(model: Model, T: float, n_paths: int, n_steps: int, seed: int | None = 0,
             antithetic: bool = False) -> Paths:
    """Convenience wrapper: draw normals and simulate paths."""
    rng = np.random.default_rng(seed)
    return model.simulate(T, draw_normals(n_paths, n_steps, model.n_factors, rng, antithetic), rng)
