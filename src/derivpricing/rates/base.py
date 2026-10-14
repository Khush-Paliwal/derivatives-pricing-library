"""Shared helpers for short-rate models."""

from __future__ import annotations

import numpy as np


def discount_factors(r: np.ndarray, dt: float, rule: str = "trapezoid") -> np.ndarray:
    """exp(-int_0^T r ds) per path from a rate path matrix of shape (n_paths, n_steps + 1).

    ``trapezoid`` is O(dt^2); ``right`` (dt * sum_{j>=1} r_j) is the O(dt) Euler sum
    often used in course notes. Both converge to the same limit.
    """
    if rule == "trapezoid":
        integral = dt * (0.5 * r[:, 0] + r[:, 1:-1].sum(axis=1) + 0.5 * r[:, -1])
    elif rule == "right":
        integral = dt * r[:, 1:].sum(axis=1)
    else:
        raise ValueError("rule must be 'trapezoid' or 'right'")
    return np.exp(-integral)


def cumulative_discount(r: np.ndarray, dt: float) -> np.ndarray:
    """Trapezoid cumulative discount factor at every grid date, shape (n_paths, n_steps + 1)."""
    incr = 0.5 * dt * (r[:, 1:] + r[:, :-1])
    return np.exp(-np.concatenate([np.zeros((r.shape[0], 1)), np.cumsum(incr, axis=1)], axis=1))


def yield_from_price(P: float, tau: float) -> float:
    """Continuously compounded yield R = -ln P / tau."""
    return float(-np.log(P) / tau)
