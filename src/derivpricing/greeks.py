"""Monte Carlo Greeks.

Three estimators for dV/dtheta, V = E[e^{-rT} f(S_T)]:

* **Bump-and-revalue with common random numbers (CRN)** - central differences
  (V(theta + h) - V(theta - h)) / 2h using the *same* normals for both prices,
  so the noise cancels; works for any model/payoff but is biased O(h^2).
* **Pathwise** - differentiate inside the expectation:
  delta = e^{-rT} E[1{S_T > K} S_T / S_0] for a call under GBM. Unbiased,
  low variance, needs a payoff that is Lipschitz in theta (fails for digitals).
* **Likelihood ratio (score function)** - differentiate the density instead:
  delta = e^{-rT} E[f(S_T) Z / (S_0 sigma sqrt(T))]. Works for discontinuous
  payoffs, higher variance.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from derivpricing import monte_carlo as mc
from derivpricing.models.base import Model
from derivpricing.stats import MCResult, mc_estimate


def bump_greeks(model: Model, payoff, T: float, n_paths: int = 200_000, n_steps: int = 1,
                seed: int = 0, rel_bump: float = 0.01) -> dict[str, float]:
    """Delta, gamma, vega (if the model has ``sigma``) and theta by central differences with CRN."""
    rng = np.random.default_rng(seed)
    Z = mc.draw_normals(n_paths, n_steps, model.n_factors, rng, antithetic=True)

    def value(m: Model, t: float) -> float:
        return mc.price(m, payoff, t, Z=Z, seed=seed).price

    h = rel_bump * model.S0
    v0 = value(model, T)
    up, dn = value(replace(model, S0=model.S0 + h), T), value(replace(model, S0=model.S0 - h), T)
    out = {"price": v0, "delta": (up - dn) / (2 * h), "gamma": (up - 2 * v0 + dn) / h**2}
    if hasattr(model, "sigma"):
        hs = rel_bump * model.sigma
        out["vega"] = (value(replace(model, sigma=model.sigma + hs), T)
                       - value(replace(model, sigma=model.sigma - hs), T)) / (2 * hs)
    ht = rel_bump * T
    out["theta"] = -(value(model, T + ht) - value(model, T - ht)) / (2 * ht)  # dV/dt = -dV/dT
    return out


def pathwise_delta_call(S0, K, r, sigma, T, n_paths: int = 200_000, seed: int = 0) -> MCResult:
    z = np.random.default_rng(seed).standard_normal(n_paths)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * z)
    return mc_estimate(math.exp(-r * T) * (ST > K) * ST / S0)


def likelihood_ratio_delta(payoff_terminal, S0, r, sigma, T, n_paths: int = 200_000, seed: int = 0) -> MCResult:
    """LR delta for any terminal payoff f(S_T) under GBM (e.g. a digital)."""
    z = np.random.default_rng(seed).standard_normal(n_paths)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * z)
    return mc_estimate(math.exp(-r * T) * payoff_terminal(ST) * z / (S0 * sigma * math.sqrt(T)))
