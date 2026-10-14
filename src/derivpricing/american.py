"""American options: Longstaff-Schwartz least-squares Monte Carlo and a CRR binomial benchmark.

LSMC (Longstaff & Schwartz 2001)
--------------------------------
V_0 = sup_tau E[e^{-r tau} h(S_tau)]. Working backwards over exercise dates
t_{m-1}, ..., t_1, on *in-the-money* paths only, regress the realised discounted
future cash flow on basis functions of the state to estimate the continuation
value C(t_j, x) ~ sum_l a_l L_l(x). Exercise where h(S_{t_j}) >= C; a path
exercises at most once, so its cash flow is overwritten (not added) when an
earlier exercise is chosen. The estimator is low-biased (sub-optimal exercise
rule) up to regression noise.

Basis families (course table, state scaled by the strike for conditioning):
weighted Laguerre e^{-x/2} L_n(x), physicists' Hermite H_n(x), monomials x^n.
"""

from __future__ import annotations

import math

import numpy as np

from derivpricing.models.base import Paths


def laguerre_weighted(x: np.ndarray, k: int) -> np.ndarray:
    """e^{-x/2} L_n(x), n = 0..k-1, with (n+1) L_{n+1} = (2n + 1 - x) L_n - n L_{n-1}."""
    L = np.empty((x.size, k))
    L[:, 0] = 1.0
    if k > 1:
        L[:, 1] = 1.0 - x
    for n in range(1, k - 1):
        L[:, n + 1] = ((2 * n + 1 - x) * L[:, n] - n * L[:, n - 1]) / (n + 1)
    return np.exp(-x / 2)[:, None] * L


def hermite(x: np.ndarray, k: int) -> np.ndarray:
    """Physicists' Hermite H_0..H_{k-1}: H_{n+1} = 2x H_n - 2n H_{n-1}."""
    H = np.empty((x.size, k))
    H[:, 0] = 1.0
    if k > 1:
        H[:, 1] = 2 * x
    for n in range(1, k - 1):
        H[:, n + 1] = 2 * x * H[:, n] - 2 * n * H[:, n - 1]
    return H


def monomial(x: np.ndarray, k: int) -> np.ndarray:
    return np.vander(x, N=k, increasing=True)


BASES = {"laguerre": laguerre_weighted, "hermite": hermite, "monomial": monomial}


def lsmc_american(paths: Paths, K: float, r: float, kind: str = "put", basis: str = "laguerre",
                  k: int = 3, use_variance: bool = False) -> float:
    """American option price by LSMC on pre-simulated paths.

    use_variance=True adds the Heston variance as a second regressor with the six
    bivariate polynomials 1, x, v, x v, x^2, v^2 (x = S / K).
    """
    S = paths.S
    n_steps = S.shape[1] - 1
    disc = math.exp(-r * paths.dt)

    def intrinsic(s):
        return np.maximum(K - s, 0.0) if kind == "put" else np.maximum(s - K, 0.0)

    cash = intrinsic(S[:, -1])
    for j in range(n_steps - 1, 0, -1):
        cash *= disc  # value at t_j of each path's future cash flow
        ev = intrinsic(S[:, j])
        itm = ev > 0
        if itm.sum() < k + 1:
            continue
        x = S[itm, j] / K
        if use_variance:
            v = paths.state["v"][itm, j]
            X = np.column_stack([np.ones_like(x), x, v, x * v, x**2, v**2])
        else:
            X = BASES[basis](x, k)
        coef, *_ = np.linalg.lstsq(X, cash[itm], rcond=None)
        exercise = ev[itm] >= X @ coef
        idx = np.flatnonzero(itm)[exercise]
        cash[idx] = ev[idx]
    return float(disc * cash.mean())


def binomial_crr(S, K, r, sigma, T, n_steps: int = 1000, kind: str = "put", american: bool = True,
                 q: float = 0.0) -> float:
    """Cox-Ross-Rubinstein tree: u = e^{sigma sqrt(dt)}, d = 1/u, p = (e^{(r-q)dt} - d)/(u - d)."""
    dt = T / n_steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    disc = math.exp(-r * dt)
    j = np.arange(n_steps + 1)
    s = S * u ** (n_steps - j) * d**j
    payoff = (lambda x: np.maximum(K - x, 0.0)) if kind == "put" else (lambda x: np.maximum(x - K, 0.0))
    V = payoff(s)
    for step in range(n_steps - 1, -1, -1):
        j = np.arange(step + 1)
        V = disc * (p * V[:-1] + (1 - p) * V[1:])
        if american:
            V = np.maximum(V, payoff(S * u ** (step - j) * d**j))
    return float(V[0])
