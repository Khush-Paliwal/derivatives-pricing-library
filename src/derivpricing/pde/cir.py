"""CIR term-structure PDEs.

By Feynman-Kac, V(t, r) = E[exp(-int_t^T r ds) g(r_T) | r_t = r] solves

    V_t + 1/2 sigma^2 r V_rr + kappa (theta - r) V_r - r V = 0,   V(T, r) = g(r).

At r = 0 the diffusion vanishes and the drift kappa theta > 0 points into the
domain, so no boundary condition is needed there: the PDE itself is applied with
an upwind (forward) difference. At r_max a linear (V_rr = 0) condition is used.

* g = 1 gives the zero-coupon bond - checked against A(tau) e^{-B(tau) r}.
* g = (face P(T, S; r) - K)^+ gives the bond call of the CIR 1985 formula.
"""

from __future__ import annotations

import numpy as np

from derivpricing.pde.theta_scheme import Boundary, solve
from derivpricing.rates.cir import CIR

SCHEMES = {"explicit": 0.0, "implicit": 1.0, "cn": 0.5}


def _grid(model: CIR, r_max: float | None, n_space: int) -> np.ndarray:
    r_max = r_max or max(0.5, 6 * max(model.r0, model.theta))
    return np.linspace(0.0, r_max, n_space + 1)


def _coefficients(model: CIR, r: np.ndarray):
    return 0.5 * model.sigma**2 * r, model.kappa * (model.theta - r), -r


def zcb_pde(model: CIR, T: float, n_space: int = 400, n_time: int = 400, scheme: str = "cn",
            r_max: float | None = None) -> float:
    r = _grid(model, r_max, n_space)
    V = solve(r, np.ones_like(r), T, n_time, *_coefficients(model, r), SCHEMES[scheme],
              lower=Boundary("pde"), upper=Boundary("linear"))
    return float(np.interp(model.r0, r, V))


def zcb_option_pde(model: CIR, T: float, S: float, K: float, kind: str = "call", face: float = 1.0,
                   n_space: int = 400, n_time: int = 400, scheme: str = "implicit",
                   r_max: float | None = None) -> float:
    """European option on a CIR zero-coupon bond by finite differences."""
    r = _grid(model, r_max, n_space)
    bond_T = face * model.zcb(S, r, t=T)
    payoff = np.maximum(bond_T - K, 0.0) if kind == "call" else np.maximum(K - bond_T, 0.0)
    V = solve(r, payoff, T, n_time, *_coefficients(model, r), SCHEMES[scheme],
              lower=Boundary("pde"), upper=Boundary("linear"), rannacher=2 if scheme == "cn" else 0)
    return float(np.interp(model.r0, r, V))
