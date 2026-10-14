"""Generic one-dimensional SDE schemes dX = a(X) dt + b(X) dW and convergence studies.

Euler-Maruyama:  X_{k+1} = X_k + a dt + b dW                       strong order 1/2, weak order 1
Milstein:        X_{k+1} = Euler + 1/2 b b' ((dW)^2 - dt)          strong order 1,   weak order 1

The extra Milstein term comes from the Ito-Taylor expansion:
int int dW_u dW_s = ((W_t - W_s)^2 - (t - s)) / 2.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Fn = Callable[[np.ndarray], np.ndarray]


def euler(a: Fn, b: Fn, x0: float, T: float, Z: np.ndarray) -> np.ndarray:
    n_paths, n_steps = Z.shape
    dt = T / n_steps
    X = np.empty((n_paths, n_steps + 1))
    X[:, 0] = x0
    for k in range(n_steps):
        x = X[:, k]
        X[:, k + 1] = x + a(x) * dt + b(x) * np.sqrt(dt) * Z[:, k]
    return X


def milstein(a: Fn, b: Fn, db: Fn, x0: float, T: float, Z: np.ndarray) -> np.ndarray:
    n_paths, n_steps = Z.shape
    dt = T / n_steps
    X = np.empty((n_paths, n_steps + 1))
    X[:, 0] = x0
    for k in range(n_steps):
        x = X[:, k]
        dW = np.sqrt(dt) * Z[:, k]
        X[:, k + 1] = x + a(x) * dt + b(x) * dW + 0.5 * b(x) * db(x) * (dW**2 - dt)
    return X


def gbm_strong_error(scheme: str, S0: float, r: float, sigma: float, T: float, n_steps: int,
                     n_paths: int = 20_000, seed: int = 0) -> float:
    """E|X_T^scheme - S_T^exact| using the *same* Brownian increments (pathwise error)."""
    Z = np.random.default_rng(seed).standard_normal((n_paths, n_steps))
    W_T = np.sqrt(T / n_steps) * Z.sum(axis=1)
    exact = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * W_T)
    a, b, db = (lambda x: r * x), (lambda x: sigma * x), (lambda x: sigma + 0 * x)
    X = euler(a, b, S0, T, Z) if scheme == "euler" else milstein(a, b, db, S0, T, Z)
    return float(np.mean(np.abs(X[:, -1] - exact)))


def convergence_order(steps, errors) -> float:
    """Slope of log(error) against log(dt) (the empirical order)."""
    dt = 1.0 / np.asarray(steps, float)
    return float(np.polyfit(np.log(dt), np.log(errors), 1)[0])
