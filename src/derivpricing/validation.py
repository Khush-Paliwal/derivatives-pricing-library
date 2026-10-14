"""Cross-validation report: every Monte Carlo price against a closed form or PDE.

``python -m derivpricing.validation`` (or the ``derivpricing-validate`` script)
regenerates ``reports/VALIDATION.md`` and the convergence figures.

A check passes when |MC - reference| <= 3 standard errors (MC) or the absolute
difference is below the stated tolerance (deterministic methods).
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from derivpricing import monte_carlo as mc
from derivpricing import payoffs as po
from derivpricing.american import binomial_crr, lsmc_american
from derivpricing.analytic import bs_price, fft_price, heston_cf
from derivpricing.analytic import exotics as ex
from derivpricing.models import GBM, Heston, MertonJD
from derivpricing.models.sde import convergence_order, gbm_strong_error
from derivpricing.pde import bs_fd, zcb_option_pde, zcb_pde
from derivpricing.qmc import qmc_integrate
from derivpricing.rates import CIR, G2pp
from derivpricing.stats import MCResult

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Check:
    group: str
    name: str
    method: str
    value: float
    reference: float
    ref_method: str
    stderr: float | None = None
    tol: float | None = None

    @property
    def passed(self) -> bool:
        if self.stderr is not None:
            return abs(self.value - self.reference) <= 3 * self.stderr
        return abs(self.value - self.reference) <= (self.tol or 1e-6)

    def row(self) -> str:
        diff = self.value - self.reference
        se = f"{self.stderr:.4f}" if self.stderr is not None else "-"
        z = f"{diff / self.stderr:+.2f}" if self.stderr else "-"
        mark = "pass" if self.passed else "**FAIL**"
        return (f"| {self.group} | {self.name} | {self.method} | {self.value:.5f} | {se} | "
                f"{self.reference:.5f} | {self.ref_method} | {diff:+.5f} | {z} | {mark} |")


def _mc(group, name, method, res: MCResult, ref, ref_method) -> Check:
    return Check(group, name, method, res.price, ref, ref_method, stderr=res.stderr)


def run_checks() -> list[Check]:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    checks: list[Check] = []
    S, K, r, sig, T = 100.0, 100.0, 0.05, 0.25, 1.0

    # --- Black-Scholes: MC with variance reduction, Fourier, PDE
    g = GBM(S, r, sig)
    bs = bs_price(S, K, r, sig, T)
    checks.append(_mc("GBM", "European call", "MC crude", mc.price(g, po.European(K), T, 200_000, antithetic=False), bs, "Black-Scholes"))
    checks.append(_mc("GBM", "European call", "MC antithetic + control", mc.price(
        g, po.European(K), T, 200_000, control=mc.ControlVariate.underlying(g, T)), bs, "Black-Scholes"))
    checks.append(_mc("GBM", "European call", "Sobol RQMC", mc.price(g, po.European(K), T, 131_072, qmc=True, antithetic=False), bs, "Black-Scholes"))
    checks.append(Check("GBM", "European put", "Crank-Nicolson PDE (ln S)", bs_fd(S, K, r, sig, T, "put"),
                        bs_price(S, K, r, sig, T, "put"), "Black-Scholes", tol=2e-3))
    checks.append(Check("GBM", "European put", "Implicit PDE (S grid)", bs_fd(S, K, r, sig, T, "put", scheme="implicit", grid="S"),
                        bs_price(S, K, r, sig, T, "put"), "Black-Scholes", tol=5e-3))
    checks.append(Check("GBM", "European put", "Explicit PDE (ln S)", bs_fd(S, K, r, sig, T, "put", scheme="explicit", n_time=4000),
                        bs_price(S, K, r, sig, T, "put"), "Black-Scholes", tol=2e-3))

    # --- path-dependent
    m = 50
    geo = ex.geometric_asian(S, K, r, sig, T, m)
    checks.append(_mc("Path-dependent", "Geometric Asian call (m=50)", "MC",
                      mc.price(g, po.Asian(K, average="geometric"), T, 200_000, n_steps=m), geo, "closed form"))
    hb = 80.0
    dop_disc = ex.down_and_out_put(S, K, r, sig, T, ex.bgk_shift(hb, sig, T, 250))
    checks.append(_mc("Path-dependent", "Down-and-out put (H=80, 250 dates)", "MC",
                      mc.price(g, po.Barrier(K, hb, "put", "down", "out"), T, 200_000, n_steps=250), dop_disc,
                      "closed form + BGK shift"))

    # --- American
    amer = binomial_crr(40, 40, 0.06, 0.2, 1, 2000)
    paths = mc.simulate(GBM(40, 0.06, 0.2), 1, 100_000, 50, seed=1, antithetic=True)
    checks.append(Check("American", "American put (S=K=40)", "LSMC Laguerre k=4, 50 dates",
                        lsmc_american(paths, 40, 0.06, "put", "laguerre", 4), amer, "CRR binomial 2000", tol=0.03))
    checks.append(Check("American", "American put (S=K=40)", "Crank-Nicolson PDE + projection",
                        bs_fd(40, 40, 0.06, 0.2, 1, "put", american=True), amer, "CRR binomial 2000", tol=2e-3))

    # --- Heston
    h = Heston(S, 0.04, r, 2.0, 0.04, 0.5, -0.7)
    hcf = h.european_price(K, T)
    checks.append(_mc("Heston", "European call", "MC full truncation, 200 steps",
                      mc.price(h, po.European(K), T, 200_000, n_steps=200), hcf, "characteristic function"))
    checks.append(Check("Heston", "European call", "Carr-Madan FFT",
                        fft_price(heston_cf(S, r, T, 0.04, 2.0, 0.04, 0.5, -0.7), S, K, r, T), hcf,
                        "Gil-Pelaez quadrature", tol=1e-3))

    # --- Merton
    mj = MertonJD(S, r, 0.2, 0.75, -0.1, 0.2)
    checks.append(_mc("Jump-diffusion", "Merton European call", "MC (exact timeline)",
                      mc.price(mj, po.European(K), T, 400_000), mj.european_price(K, T), "Poisson-BS series"))
    checks.append(Check("Jump-diffusion", "Merton European call", "Fourier (char. function)",
                        mj.european_price(K, T, method="cf"), mj.european_price(K, T), "Poisson-BS series", tol=1e-6))

    # --- CIR
    c = CIR(0.05, 0.92, 0.055, 0.12)
    checks.append(_mc("CIR", "ZCB P(0,1)", "MC exact noncentral chi2", c.mc_zcb(1.0), c.zcb(1.0), "affine closed form"))
    checks.append(Check("CIR", "ZCB P(0,1)", "Crank-Nicolson PDE", zcb_pde(c, 1.0), c.zcb(1.0), "affine closed form", tol=1e-5))
    cpn, tms = [30] * 7 + [1030], [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
    checks.append(_mc("CIR", "Coupon bond (8 x semiannual)", "MC full-truncation Euler",
                      c.mc_coupon_bond(cpn, tms, scheme="full_truncation"), c.coupon_bond(cpn, tms), "sum of ZCBs"))
    ref = c.zcb_option(0.5, 1.0, 980, "call", 1000)
    checks.append(_mc("CIR", "ZCB call (K=980, T=0.5, S=1)", "MC", c.mc_zcb_option(0.5, 1.0, 980, "call", 1000, n_paths=200_000),
                      ref, "noncentral chi2"))
    checks.append(Check("CIR", "ZCB call (K=980, T=0.5, S=1)", "Implicit PDE", zcb_option_pde(c, 0.5, 1.0, 980, "call", 1000),
                        ref, "noncentral chi2", tol=1e-3))

    # --- G2++
    for rho in (-0.7, 0.0, 0.7):
        g2 = G2pp(0.1, 0.3, 0.05, 0.09, rho, 0.055)
        checks.append(_mc("G2++", f"ZCB put (K=950, rho={rho:+.1f})", "MC exact OU",
                          g2.mc_zcb_option(0.5, 1.0, 950, "put", 1000, n_paths=400_000, n_steps=100),
                          g2.zcb_option(0.5, 1.0, 950, "put", 1000), "Brigo-Mercurio closed form"))
    return checks


def convergence_figures(out: Path) -> dict[str, float]:
    """Save convergence plots; return the empirical orders."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    orders = {}

    # 1. MC vs QMC error against N
    f = lambda x: np.exp(x.sum(axis=1))  # noqa: E731  int over [0,1]^2 = (e - 1)^2
    true = (math.e - 1) ** 2
    Ns = 2 ** np.arange(8, 17)
    rng = np.random.default_rng(0)
    mc_err = [np.mean([abs(f(rng.random((n, 2))).mean() - true) for _ in range(20)]) for n in Ns]
    qmc_err = [abs(qmc_integrate(f, n, 2) - true) for n in Ns]
    orders["MC error vs N"] = float(np.polyfit(np.log(Ns), np.log(mc_err), 1)[0])
    orders["Halton QMC error vs N"] = float(np.polyfit(np.log(Ns), np.log(qmc_err), 1)[0])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(Ns, mc_err, "o-", label=f"pseudo-random MC (slope {orders['MC error vs N']:.2f})")
    ax.loglog(Ns, qmc_err, "s-", label=f"Halton QMC (slope {orders['Halton QMC error vs N']:.2f})")
    ax.set(xlabel="N", ylabel="|error|", title=r"$\int_{[0,1]^2} e^{x+y}\,dx\,dy$")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "mc_vs_qmc.png", dpi=150); plt.close(fig)

    # 2. Euler vs Milstein strong error
    steps = [4, 8, 16, 32, 64, 128]
    e_err = [gbm_strong_error("euler", 100, 0.05, 0.4, 1, n) for n in steps]
    m_err = [gbm_strong_error("milstein", 100, 0.05, 0.4, 1, n) for n in steps]
    orders["Euler strong order"] = convergence_order(steps, e_err)
    orders["Milstein strong order"] = convergence_order(steps, m_err)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(steps, e_err, "o-", label=f"Euler (order {orders['Euler strong order']:.2f})")
    ax.loglog(steps, m_err, "s-", label=f"Milstein (order {orders['Milstein strong order']:.2f})")
    ax.set(xlabel="time steps", ylabel=r"$E|\hat S_T - S_T|$", title="Strong convergence, GBM")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "euler_vs_milstein.png", dpi=150); plt.close(fig)

    # 3. PDE scheme error vs grid
    ref = bs_price(100, 100, 0.05, 0.25, 1, "put")
    ns = [25, 50, 100, 200, 400]
    errs = {s: [abs(bs_fd(100, 100, 0.05, 0.25, 1, "put", scheme=s, n_space=n, n_time=n,
                          rannacher=0 if s != "cn" else 2) - ref) for n in ns] for s in ("implicit", "cn")}
    orders["Implicit FD order (dt ~ dx)"] = float(np.polyfit(np.log(ns), np.log(errs["implicit"]), 1)[0])
    orders["Crank-Nicolson order (dt ~ dx)"] = float(np.polyfit(np.log(ns), np.log(errs["cn"]), 1)[0])
    fig, ax = plt.subplots(figsize=(6, 4))
    for s, lab in (("implicit", "Implicit"), ("cn", "Crank-Nicolson")):
        ax.loglog(ns, errs[s], "o-", label=lab)
    ax.set(xlabel="grid points (space = time)", ylabel="|error|", title="Black-Scholes PDE convergence")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "pde_convergence.png", dpi=150); plt.close(fig)

    # 4. G2++ rho sweep and 5. variance-reduction comparison
    rhos = np.linspace(-0.7, 0.7, 8)
    cf, mcv, mse = [], [], []
    for i, rho in enumerate(rhos):
        g2 = G2pp(0.1, 0.3, 0.05, 0.09, rho, 0.055)
        cf.append(g2.zcb_option(0.5, 1, 950, "put", 1000))
        res = g2.mc_zcb_option(0.5, 1, 950, "put", 1000, n_paths=60_000, n_steps=100, seed=i)
        mcv.append(res.price); mse.append(res.stderr)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rhos, cf, "-", label="closed form")
    ax.errorbar(rhos, mcv, yerr=1.96 * np.array(mse), fmt="o", capsize=3, label="Monte Carlo (95% CI)")
    ax.set(xlabel=r"$\rho$", ylabel="put value ($)", title="G2++ ZCB put vs factor correlation")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "g2pp_rho_sweep.png", dpi=150); plt.close(fig)

    g = GBM(100, 0.05, 0.25)
    methods = {
        "crude": dict(antithetic=False),
        "antithetic": dict(antithetic=True),
        "antithetic + control": dict(antithetic=True, control=mc.ControlVariate.underlying(g, 1)),
        "Sobol RQMC": dict(qmc=True, antithetic=False),
    }
    se = {k: mc.price(g, po.European(100), 1, 65_536, **kw).stderr for k, kw in methods.items()}
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(list(se), list(se.values()), color=["#888", "#4c72b0", "#55a868", "#c44e52"])
    ax.set(ylabel="standard error (65,536 paths)", title="Variance reduction, ATM call")
    for i, v in enumerate(se.values()):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "variance_reduction.png", dpi=150); plt.close(fig)
    orders.update({f"stderr[{k}]": v for k, v in se.items()})
    return orders


def write_report(path: Path | None = None) -> Path:
    path = path or ROOT / "reports" / "VALIDATION.md"
    t0 = time.time()
    checks = run_checks()
    orders = convergence_figures(path.parent / "figures")
    n_pass = sum(c.passed for c in checks)
    lines = [
        "# Validation report",
        "",
        f"_Generated by `derivpricing-validate` in {time.time() - t0:.0f}s. "
        f"{n_pass}/{len(checks)} checks pass (MC within 3 standard errors; deterministic methods within tolerance)._",
        "",
        "| Model | Instrument | Method | Value | Std. err | Reference | Reference method | Diff | z | |",
        "|---|---|---|---|---|---|---|---|---|---|",
        *[c.row() for c in checks],
        "",
        "## Convergence",
        "",
        "| Quantity | Value |",
        "|---|---|",
        *[f"| {k} | {v:.4g} |" for k, v in orders.items()],
        "",
        "Theory: MC error ~ N^-0.5, QMC ~ N^-1 (up to log factors), Euler strong order 0.5, Milstein 1, "
        "implicit FD order 1 in dt, Crank-Nicolson order 2.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    p = write_report()
    print(p.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
