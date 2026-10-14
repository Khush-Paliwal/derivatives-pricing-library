"""Black-Scholes PDE by finite differences (European and American).

In S:        V_t + 1/2 sigma^2 S^2 V_SS + (r - q) S V_S - r V = 0       (variable coefficients)
In x = ln S: V_t + 1/2 sigma^2 V_xx + (r - q - sigma^2/2) V_x - r V = 0 (constant coefficients)

Boundary conditions (the "delta" conditions):
  call: V_S -> 0 as S -> 0,   V_S -> 1 as S -> inf
  put:  V_S -> -1 as S -> 0,  V_S -> 0 as S -> inf
American options: V <- max(V, intrinsic) after every time step.
"""

from __future__ import annotations

import math

import numpy as np

from derivpricing.pde.theta_scheme import Boundary, solve

SCHEMES = {"explicit": 0.0, "implicit": 1.0, "cn": 0.5}


def bs_fd(S0, K, r, sigma, T, kind: str = "put", american: bool = False, scheme: str = "cn",
          grid: str = "log", n_space: int = 400, n_time: int = 400, width: float = 5.0, q: float = 0.0,
          rannacher: int | None = None, return_grid: bool = False):
    """Finite-difference price of a vanilla option.

    grid="log": x = ln S on [ln S0 - width*sigma*sqrt(T), ln S0 + width*sigma*sqrt(T)], S0 on a node.
    grid="S":   S on [0, S0 * exp(width*sigma*sqrt(T))].
    Returns the price at S0 (or (S_grid, V) if return_grid).
    """
    theta = SCHEMES[scheme]
    if rannacher is None:
        rannacher = 2 if scheme == "cn" else 0
    half = width * sigma * math.sqrt(T)
    if grid == "log":
        n_side = n_space // 2
        x = math.log(S0) + np.linspace(-half, half, 2 * n_side + 1)
        S = np.exp(x)
        y, a, b, c = x, 0.5 * sigma**2, r - q - 0.5 * sigma**2, -r
    elif grid == "S":
        S = np.linspace(0.0, S0 * math.exp(half), n_space + 1)
        y, a, b, c = S, 0.5 * sigma**2 * S**2, (r - q) * S, -r
    else:
        raise ValueError("grid must be 'log' or 'S'")

    payoff = np.maximum(S - K, 0.0) if kind == "call" else np.maximum(K - S, 0.0)
    dS_lo, dS_hi = S[1] - S[0], S[-1] - S[-2]
    if kind == "call":
        lower = Boundary("pde") if grid == "S" else Boundary("difference", 0.0)
        upper = Boundary("difference", -dS_hi)  # V_{N-1} - V_N = -(S_N - S_{N-1})
    else:
        lower = Boundary("pde") if grid == "S" else Boundary("difference", -dS_lo)  # V_1 - V_0 = -(S_1 - S_0)
        upper = Boundary("difference", 0.0)
    V = solve(y, payoff, T, n_time, a, b, c, theta, lower, upper,
              exercise=payoff if american else None, rannacher=rannacher)
    if return_grid:
        return S, V
    return float(np.interp(S0, S, V))


def explicit_stability_dx(sigma: float, dt: float) -> float:
    """Smallest dx for which the explicit log-grid scheme is a positive (stable) recursion:
    dx >= sigma sqrt(3 dt) keeps p_u, p_m, p_d > 0 (the trinomial-tree equivalence)."""
    return sigma * math.sqrt(3 * dt)
