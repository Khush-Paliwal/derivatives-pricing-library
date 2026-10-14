"""Payoffs as functions of simulated paths.

Each payoff maps a :class:`~derivpricing.models.base.Paths` object to an array of
*undiscounted* cash flows at maturity. Path-dependent payoffs monitor the grid
points t_1, ..., t_m (S_0 is excluded from averages but included in running
extrema, the usual convention for lookbacks).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from derivpricing.models.base import Paths

PayoffFn = Callable[[Paths], np.ndarray]


def _vanilla(x: np.ndarray, K: float, kind: str) -> np.ndarray:
    if kind == "call":
        return np.maximum(x - K, 0.0)
    if kind == "put":
        return np.maximum(K - x, 0.0)
    raise ValueError("kind must be 'call' or 'put'")


@dataclass(frozen=True)
class European:
    K: float
    kind: str = "call"

    def __call__(self, p: Paths) -> np.ndarray:
        return _vanilla(p.terminal, self.K, self.kind)


@dataclass(frozen=True)
class Digital:
    """Cash-or-nothing digital paying 1 if S_T > K (call) or S_T < K (put)."""

    K: float
    kind: str = "call"

    def __call__(self, p: Paths) -> np.ndarray:
        return (p.terminal > self.K if self.kind == "call" else p.terminal < self.K).astype(float)


@dataclass(frozen=True)
class Asian:
    """Average-price (fixed strike) or average-strike (floating) option.

    fixed:    call (A - K)^+,   put (K - A)^+
    floating: call (S_T - A)^+, put (A - S_T)^+
    with A the arithmetic or geometric mean of S(t_1), ..., S(t_m).
    """

    K: float = 0.0
    kind: str = "call"
    average: str = "arithmetic"
    strike: str = "fixed"

    def mean(self, p: Paths) -> np.ndarray:
        mon = p.S[:, 1:]
        if self.average == "arithmetic":
            return mon.mean(axis=1)
        if self.average == "geometric":
            return np.exp(np.log(mon).mean(axis=1))
        raise ValueError("average must be 'arithmetic' or 'geometric'")

    def __call__(self, p: Paths) -> np.ndarray:
        A = self.mean(p)
        if self.strike == "fixed":
            return _vanilla(A, self.K, self.kind)
        return np.maximum(p.terminal - A, 0.0) if self.kind == "call" else np.maximum(A - p.terminal, 0.0)


@dataclass(frozen=True)
class Barrier:
    """Knock-in / knock-out vanilla with a discretely monitored barrier.

    ``barrier`` may be a constant or a function of time b(t) (time-dependent barrier).
    down-out alive iff min S(t_j) > b(t_j); up-out iff max S(t_j) < b(t_j); "in" is the complement.
    """

    K: float
    barrier: float | Callable[[np.ndarray], np.ndarray]
    kind: str = "call"
    direction: str = "down"
    knock: str = "out"

    def __call__(self, p: Paths) -> np.ndarray:
        b = self.barrier(p.t[1:]) if callable(self.barrier) else np.full(p.t.size - 1, self.barrier)
        mon = p.S[:, 1:]
        if self.direction == "down":
            hit = (mon <= b).any(axis=1)
        elif self.direction == "up":
            hit = (mon >= b).any(axis=1)
        else:
            raise ValueError("direction must be 'down' or 'up'")
        alive = ~hit if self.knock == "out" else hit
        return _vanilla(p.terminal, self.K, self.kind) * alive


@dataclass(frozen=True)
class Lookback:
    """floating call S_T - min S, floating put max S - S_T, fixed call (max S - K)^+, fixed put (K - min S)^+."""

    K: float = 0.0
    kind: str = "call"
    strike: str = "floating"

    def __call__(self, p: Paths) -> np.ndarray:
        s_max, s_min, s_T = p.S.max(axis=1), p.S.min(axis=1), p.terminal
        if self.strike == "floating":
            return s_T - s_min if self.kind == "call" else s_max - s_T
        return np.maximum(s_max - self.K, 0.0) if self.kind == "call" else np.maximum(self.K - s_min, 0.0)
