"""Cox-Ingersoll-Ross (1985) short-rate model.

    dr = kappa (theta - r) dt + sigma sqrt(r) dW

Mean-reverting, non-negative (strictly positive if the Feller condition
2 kappa theta >= sigma^2 holds), affine: P(t, T) = A(tau) exp(-B(tau) r_t) with

    h = sqrt(kappa^2 + 2 sigma^2),
    B(tau) = 2 (e^{h tau} - 1) / ((h + kappa)(e^{h tau} - 1) + 2h),
    A(tau) = [2h e^{(kappa + h) tau / 2} / ((h + kappa)(e^{h tau} - 1) + 2h)]^{2 kappa theta / sigma^2}.

The transition law is a scaled noncentral chi-square, which gives both an exact
simulation scheme and the closed-form bond option (CIR 1985).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import ncx2

from derivpricing.rates.base import cumulative_discount, discount_factors
from derivpricing.stats import MCResult, mc_estimate


@dataclass(frozen=True)
class CIR:
    r0: float
    kappa: float
    theta: float
    sigma: float

    # ------------------------------------------------------------------ closed forms
    @property
    def feller(self) -> bool:
        return 2 * self.kappa * self.theta >= self.sigma**2

    def AB(self, tau: float) -> tuple[float, float]:
        h = math.sqrt(self.kappa**2 + 2 * self.sigma**2)
        e = math.exp(h * tau) - 1.0
        den = (h + self.kappa) * e + 2 * h
        B = 2 * e / den
        A = (2 * h * math.exp((self.kappa + h) * tau / 2) / den) ** (2 * self.kappa * self.theta / self.sigma**2)
        return A, B

    def zcb(self, T: float, r=None, t: float = 0.0):
        """P(t, T) at short rate r (default r0); vectorised over r."""
        A, B = self.AB(T - t)
        r = self.r0 if r is None else np.asarray(r, float)
        return A * np.exp(-B * r)

    def yield_curve(self, maturities, r=None) -> np.ndarray:
        return np.array([-math.log(self.zcb(T, r)) / T for T in maturities])

    def coupon_bond(self, coupons, times, r=None, t: float = 0.0) -> float:
        return float(sum(c * self.zcb(Ti, r, t) for c, Ti in zip(coupons, times, strict=True)))

    def zcb_option(self, T: float, S: float, K: float, kind: str = "call", face: float = 1.0) -> float:
        """European option (expiry T) on a ZCB maturing at S > T, strike K in the same units as ``face``.

        c = face P(0,S) chi2(2 r*(phi + psi + B); d, 2 phi^2 r0 e^{theta' T} / (phi + psi + B))
            - K P(0,T) chi2(2 r*(phi + psi); d, 2 phi^2 r0 e^{theta' T} / (phi + psi))
        with theta' = sqrt(kappa^2 + 2 sigma^2), phi = 2 theta' / (sigma^2 (e^{theta' T} - 1)),
        psi = (kappa + theta') / sigma^2, d = 4 kappa theta / sigma^2 and r* the rate at which
        the bond at T is worth exactly the strike: r* = ln(A(S-T) / (K/face)) / B(S-T).
        """
        g = math.sqrt(self.kappa**2 + 2 * self.sigma**2)
        phi = 2 * g / (self.sigma**2 * (math.exp(g * T) - 1))
        psi = (self.kappa + g) / self.sigma**2
        A, B = self.AB(S - T)
        r_star = math.log(A / (K / face)) / B
        d = 4 * self.kappa * self.theta / self.sigma**2
        nc = 2 * phi**2 * self.r0 * math.exp(g * T)
        P_S, P_T = face * self.zcb(S), self.zcb(T)
        call = (P_S * ncx2.cdf(2 * r_star * (phi + psi + B), d, nc / (phi + psi + B))
                - K * P_T * ncx2.cdf(2 * r_star * (phi + psi), d, nc / (phi + psi)))
        return float(call if kind == "call" else call - P_S + K * P_T)

    def coupon_bond_option(self, T, coupons, times, K, kind="call") -> float:
        """Jamshidian (1989): an option on a coupon bond is a portfolio of ZCB options.

        Because bond prices fall monotonically in r, find r* with sum_i c_i P(T, T_i; r*) = K,
        set K_i = P(T, T_i; r*), and sum c_i * ZCBoption(T, T_i, K_i).
        """
        cs, ts = np.asarray(coupons, float), np.asarray(times, float)
        live = ts > T
        cs, ts = cs[live], ts[live]

        def bond_at_T(r):
            return sum(c * self.AB(Ti - T)[0] * math.exp(-self.AB(Ti - T)[1] * r) for c, Ti in zip(cs, ts, strict=True))

        r_star = brentq(lambda r: bond_at_T(r) - K, -0.5, 5.0)
        total = 0.0
        for c, Ti in zip(cs, ts, strict=True):
            A, B = self.AB(Ti - T)
            total += c * self.zcb_option(T, Ti, A * math.exp(-B * r_star), kind)
        return float(total)

    # ------------------------------------------------------------------ simulation
    def simulate(self, T: float, n_steps: int, n_paths: int, rng: np.random.Generator | None = None,
                 scheme: str = "exact", antithetic: bool = False) -> np.ndarray:
        """Short-rate paths, shape (n_paths, n_steps + 1).

        ``exact``: r_{t+dt} = c * X, X ~ noncentral chi2(d, r_t e^{-kappa dt} / c),
                   c = sigma^2 (1 - e^{-kappa dt}) / (4 kappa), d = 4 kappa theta / sigma^2.
        ``full_truncation``: r_{k+1} = r_k + kappa (theta - r_k^+) dt + sigma sqrt(r_k^+ dt) Z.
        """
        rng = rng or np.random.default_rng()
        dt = T / n_steps
        r = np.empty((n_paths, n_steps + 1))
        r[:, 0] = self.r0
        if scheme == "exact":
            c = self.sigma**2 * (1 - math.exp(-self.kappa * dt)) / (4 * self.kappa)
            d = 4 * self.kappa * self.theta / self.sigma**2
            for k in range(n_steps):
                r[:, k + 1] = c * rng.noncentral_chisquare(d, r[:, k] * math.exp(-self.kappa * dt) / c)
            return r
        if scheme != "full_truncation":
            raise ValueError("scheme must be 'exact' or 'full_truncation'")
        half = n_paths // 2 if antithetic else n_paths
        Z = rng.standard_normal((half, n_steps))
        Z = np.concatenate([Z, -Z]) if antithetic else Z
        for k in range(n_steps):
            rp = np.maximum(r[:, k], 0.0)
            r[:, k + 1] = r[:, k] + self.kappa * (self.theta - rp) * dt + self.sigma * np.sqrt(rp * dt) * Z[:, k]
        return r

    def mc_zcb(self, T, n_steps=250, n_paths=50_000, seed=0, scheme="exact") -> MCResult:
        r = self.simulate(T, n_steps, n_paths, np.random.default_rng(seed), scheme)
        return mc_estimate(discount_factors(r, T / n_steps))

    def mc_coupon_bond(self, coupons, times, steps_per_year=250, n_paths=50_000, seed=0,
                       scheme="exact") -> MCResult:
        """sum_i c_i exp(-int_0^{T_i} r ds) along each path (one simulation to the last date)."""
        times = np.asarray(times, float)
        n_steps = int(round(steps_per_year * times.max()))
        r = self.simulate(float(times.max()), n_steps, n_paths, np.random.default_rng(seed), scheme)
        D = cumulative_discount(r, times.max() / n_steps)
        idx = np.rint(times / times.max() * n_steps).astype(int)
        return mc_estimate(D[:, idx] @ np.asarray(coupons, float))

    def mc_zcb_option(self, T, S, K, kind="call", face=1.0, n_steps=250, n_paths=100_000, seed=0,
                      scheme="exact") -> MCResult:
        """Simulate r to the option expiry T; the bond at T is the closed form P(T, S; r_T)."""
        r = self.simulate(T, n_steps, n_paths, np.random.default_rng(seed), scheme)
        bond_T = face * self.zcb(S, r[:, -1], t=T)
        payoff = np.maximum(bond_T - K, 0.0) if kind == "call" else np.maximum(K - bond_T, 0.0)
        return mc_estimate(discount_factors(r, T / n_steps) * payoff)
