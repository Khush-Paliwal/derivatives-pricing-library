"""G2++ two-factor Gaussian short-rate model (Brigo & Mercurio 2006, ch. 4).

    r_t = x_t + y_t + phi,   dx = -a x dt + sigma dW_1,   dy = -b y dt + eta dW_2,   d<W_1, W_2> = rho dt

with x_0 = y_0 = 0 and a constant shift phi (equal to r_0). Two correlated
Ornstein-Uhlenbeck factors give a richer correlation structure across
maturities than any one-factor model, while everything stays Gaussian:

* int_t^T r ds is normal, so P(t, T) = exp(-phi tau - B_a x_t - B_b y_t + V(t, T)/2)
  with B_z = (1 - e^{-z tau}) / z and V the variance of the integral;
* ln P(T, S) at T is normal, so options on ZCBs have Black-type closed forms with
  volatility Sigma(t, T, S).
Rates can become negative (as in all Gaussian models).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from derivpricing.rates.base import discount_factors
from derivpricing.stats import MCResult, mc_estimate


@dataclass(frozen=True)
class G2pp:
    a: float
    b: float
    sigma: float
    eta: float
    rho: float
    phi: float

    @staticmethod
    def _B(z: float, tau: float) -> float:
        return (1 - math.exp(-z * tau)) / z

    def V(self, tau: float) -> float:
        """Var(int_t^T (x + y) ds | F_t), tau = T - t."""
        a, b, s, e, rho = self.a, self.b, self.sigma, self.eta, self.rho
        va = s**2 / a**2 * (tau + 2 / a * math.exp(-a * tau) - 1 / (2 * a) * math.exp(-2 * a * tau) - 3 / (2 * a))
        vb = e**2 / b**2 * (tau + 2 / b * math.exp(-b * tau) - 1 / (2 * b) * math.exp(-2 * b * tau) - 3 / (2 * b))
        vab = 2 * rho * s * e / (a * b) * (tau + (math.exp(-a * tau) - 1) / a + (math.exp(-b * tau) - 1) / b
                                            - (math.exp(-(a + b) * tau) - 1) / (a + b))
        return va + vb + vab

    def zcb(self, T: float, x=0.0, y=0.0, t: float = 0.0):
        tau = T - t
        return np.exp(-self.phi * tau - self._B(self.a, tau) * np.asarray(x) - self._B(self.b, tau) * np.asarray(y)
                      + 0.5 * self.V(tau))

    def Sigma(self, T: float, S: float, t: float = 0.0) -> float:
        """Volatility of ln P(T, S) seen from t (the option's 'total vol')."""
        a, b, s, e, rho = self.a, self.b, self.sigma, self.eta, self.rho
        aS, bS = 1 - math.exp(-a * (S - T)), 1 - math.exp(-b * (S - T))
        v = (s**2 / (2 * a**3) * aS**2 * (1 - math.exp(-2 * a * (T - t)))
             + e**2 / (2 * b**3) * bS**2 * (1 - math.exp(-2 * b * (T - t)))
             + 2 * rho * s * e / (a * b * (a + b)) * aS * bS * (1 - math.exp(-(a + b) * (T - t))))
        return math.sqrt(v)

    def zcb_option(self, T: float, S: float, K: float, kind: str = "put", face: float = 1.0) -> float:
        """Black-type formula: call = L P(0,S) N(d+) - K P(0,T) N(d-), d+/- = ln(L P(0,S) / (K P(0,T)))/Sigma +/- Sigma/2."""
        PS, PT = face * float(self.zcb(S)), float(self.zcb(T))
        sig = self.Sigma(T, S)
        d1 = math.log(PS / (K * PT)) / sig + 0.5 * sig
        d2 = d1 - sig
        if kind == "call":
            return float(PS * norm.cdf(d1) - K * PT * norm.cdf(d2))
        return float(K * PT * norm.cdf(-d2) - PS * norm.cdf(-d1))

    # ------------------------------------------------------------------ simulation
    def simulate(self, T: float, n_steps: int, n_paths: int, rng: np.random.Generator | None = None,
                 exact: bool = True, antithetic: bool = True):
        """Factor paths (x, y), each (n_paths, n_steps + 1).

        exact=True samples the bivariate OU transition exactly:
        x' = x e^{-a dt} + e_x, y' = y e^{-b dt} + e_y with Var e_x = sigma^2 (1 - e^{-2a dt})/(2a),
        Var e_y = eta^2 (1 - e^{-2b dt})/(2b), Cov = rho sigma eta (1 - e^{-(a+b) dt})/(a + b).
        """
        rng = rng or np.random.default_rng()
        dt = T / n_steps
        half = n_paths // 2 if antithetic else n_paths
        Z = rng.standard_normal((half, n_steps, 2))
        Z = np.concatenate([Z, -Z]) if antithetic else Z
        if exact:
            ea, eb = math.exp(-self.a * dt), math.exp(-self.b * dt)
            vx = self.sigma**2 * (1 - ea**2) / (2 * self.a)
            vy = self.eta**2 * (1 - eb**2) / (2 * self.b)
            cxy = self.rho * self.sigma * self.eta * (1 - math.exp(-(self.a + self.b) * dt)) / (self.a + self.b)
            L = np.linalg.cholesky([[vx, cxy], [cxy, vy]])
        x = np.zeros((Z.shape[0], n_steps + 1))
        y = np.zeros_like(x)
        for k in range(n_steps):
            if exact:
                shock = Z[:, k, :] @ L.T
                x[:, k + 1] = x[:, k] * ea + shock[:, 0]
                y[:, k + 1] = y[:, k] * eb + shock[:, 1]
            else:
                w1 = Z[:, k, 0]
                w2 = self.rho * w1 + math.sqrt(1 - self.rho**2) * Z[:, k, 1]
                x[:, k + 1] = x[:, k] - self.a * x[:, k] * dt + self.sigma * math.sqrt(dt) * w1
                y[:, k + 1] = y[:, k] - self.b * y[:, k] * dt + self.eta * math.sqrt(dt) * w2
        return x, y

    def mc_zcb(self, T, n_steps=250, n_paths=50_000, seed=0) -> MCResult:
        x, y = self.simulate(T, n_steps, n_paths, np.random.default_rng(seed))
        return mc_estimate(self._pair(discount_factors(x + y + self.phi, T / n_steps)))

    def mc_zcb_option(self, T, S, K, kind="put", face=1.0, n_steps=250, n_paths=100_000, seed=0,
                      exact=True) -> MCResult:
        """Simulate the factors to T; the bond at T is the closed form P(T, S; x_T, y_T)."""
        x, y = self.simulate(T, n_steps, n_paths, np.random.default_rng(seed), exact=exact)
        bond_T = face * self.zcb(S, x[:, -1], y[:, -1], t=T)
        payoff = np.maximum(bond_T - K, 0.0) if kind == "call" else np.maximum(K - bond_T, 0.0)
        return mc_estimate(self._pair(discount_factors(x + y + self.phi, T / n_steps) * payoff))

    @staticmethod
    def _pair(v: np.ndarray) -> np.ndarray:
        h = v.size // 2
        return 0.5 * (v[:h] + v[h:])
