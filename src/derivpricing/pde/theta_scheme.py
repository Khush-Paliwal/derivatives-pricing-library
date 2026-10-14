"""A general theta-scheme for one-dimensional linear parabolic PDEs.

Written in time-to-maturity tau = T - t, every pricing PDE in this library has the form

    V_tau = a(y) V_yy + b(y) V_y + c(y) V,     V(0, y) = payoff(y).

Central differences on a uniform grid y_j give (L V)_j = l_j V_{j-1} + m_j V_j + u_j V_{j+1} with

    l_j = a_j / dy^2 - b_j / (2 dy),   m_j = -2 a_j / dy^2 + c_j,   u_j = a_j / dy^2 + b_j / (2 dy),

and the theta-scheme  (I - theta dt L) V^{n+1} = (I + (1 - theta) dt L) V^n :

* theta = 0   explicit (EFD): O(dt + dy^2), stable only if dt <= dy^2 / (2 max a)
* theta = 1   fully implicit (IFD): O(dt + dy^2), unconditionally stable
* theta = 1/2 Crank-Nicolson (CN): O(dt^2 + dy^2), unconditionally stable but can ring
  on non-smooth payoffs - ``rannacher`` replaces the first steps by implicit ones.

Each step is a banded linear solve (O(n)). Boundary rows replace the first/last
equations; American exercise is imposed by projection V <- max(V, exercise) after
every step.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded


@dataclass(frozen=True)
class Boundary:
    """Boundary condition at one end of the grid.

    kind:
      ``dirichlet``  V = value(tau)
      ``difference`` V_inner - V_edge = value(tau)  (a slope condition written on grid values,
                     e.g. dV/dS = -1 for a put at S -> 0 becomes V_1 - V_0 = -(S_1 - S_0))
      ``linear``     V_yy = 0 (linear extrapolation)
      ``pde``        the PDE itself with a one-sided first derivative and no diffusion,
                     valid where a(y) = 0 (degenerate boundary, e.g. r = 0 in CIR or S = 0)
    """

    kind: str = "linear"
    value: Callable[[float], float] | float = 0.0

    def at(self, tau: float) -> float:
        return self.value(tau) if callable(self.value) else float(self.value)


LINEAR = Boundary("linear")


def thomas(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Thomas algorithm for a tridiagonal system (lower[0] and upper[-1] unused).

    Forward elimination then back substitution, O(n), no pivoting - safe for the
    diagonally dominant matrices produced by implicit/CN schemes.
    """
    n = diag.size
    c, d = np.empty(n), np.empty(n)
    c[0], d[0] = upper[0] / diag[0], rhs[0] / diag[0]
    for i in range(1, n):
        m = diag[i] - lower[i] * c[i - 1]
        c[i] = upper[i] / m if i < n - 1 else 0.0
        d[i] = (rhs[i] - lower[i] * d[i - 1]) / m
    x = np.empty(n)
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


def solve(y: np.ndarray, payoff: np.ndarray, T: float, n_time: int, a, b, c, theta: float = 0.5,
          lower: Boundary = LINEAR, upper: Boundary = LINEAR,
          exercise: np.ndarray | None = None, rannacher: int = 0) -> np.ndarray:
    """March V from tau = 0 to tau = T; returns V on the grid ``y`` at tau = T (i.e. t = 0)."""
    y = np.asarray(y, float)
    n = y.size
    dy, dt = y[1] - y[0], T / n_time
    a, b, c = (np.broadcast_to(np.asarray(v, float), (n,)) for v in (a, b, c))
    if theta < 0.5 and dt > dy**2 / (2 * a.max() + 1e-300):
        warnings.warn(f"explicit scheme unstable: dt={dt:.2e} > dy^2/(2 max a)={dy**2 / (2 * a.max()):.2e}",
                      RuntimeWarning, stacklevel=2)

    l = a / dy**2 - b / (2 * dy)
    m = -2 * a / dy**2 + c
    u = a / dy**2 + b / (2 * dy)

    def apply_L(V):
        out = m * V
        out[1:-1] += l[1:-1] * V[:-2] + u[1:-1] * V[2:]
        return out

    def matrix(th):
        """Banded (2,2) storage of I - th dt L with boundary rows."""
        ab = np.zeros((5, n))  # ab[2 + i - j, j] = A[i, j]
        ab[2, :] = 1 - th * dt * m
        ab[1, 1:] = -th * dt * u[:-1]  # A[i, i+1]
        ab[3, :-1] = -th * dt * l[1:]  # A[i+1, i]
        for edge, bc in ((0, lower), (n - 1, upper)):
            inner = 1 if edge == 0 else n - 2
            far = 2 if edge == 0 else n - 3
            for j in (edge, inner, far):
                ab[2 + edge - j, j] = 0.0
            if bc.kind == "dirichlet":
                ab[2, edge] = 1.0
            elif bc.kind == "difference":
                ab[2, edge], ab[2 + edge - inner, inner] = -1.0, 1.0
            elif bc.kind == "linear":
                ab[2, edge], ab[2 + edge - inner, inner], ab[2 + edge - far, far] = 1.0, -2.0, 1.0
            elif bc.kind == "pde":
                sgn = 1 if edge == 0 else -1  # one-sided derivative pointing into the grid
                coef = sgn * b[edge] / dy
                ab[2, edge] = 1 - th * dt * (c[edge] - coef)
                ab[2 + edge - inner, inner] = -th * dt * coef
            else:
                raise ValueError(f"unknown boundary kind {bc.kind!r}")
        return ab

    def rhs(V, th, tau_new):
        out = V + (1 - th) * dt * apply_L(V)
        for edge, bc in ((0, lower), (n - 1, upper)):
            inner = 1 if edge == 0 else n - 2
            if bc.kind in ("dirichlet", "difference"):
                out[edge] = bc.at(tau_new)
            elif bc.kind == "linear":
                out[edge] = 0.0
            else:  # pde
                sgn = 1 if edge == 0 else -1
                coef = sgn * b[edge] / dy
                out[edge] = V[edge] + (1 - th) * dt * ((c[edge] - coef) * V[edge] + coef * V[inner])
        return out

    mats = {th: matrix(th) for th in {theta, 1.0}}
    V = np.asarray(payoff, float).copy()
    for step in range(n_time):
        th = 1.0 if step < rannacher else theta
        V = solve_banded((2, 2), mats[th], rhs(V, th, (step + 1) * dt))
        if exercise is not None:
            V = np.maximum(V, exercise)
    return V
