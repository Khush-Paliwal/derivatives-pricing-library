import numpy as np
import pytest

from derivpricing import monte_carlo as mc
from derivpricing.american import binomial_crr, lsmc_american
from derivpricing.analytic import bs_price
from derivpricing.models import GBM, Heston
from derivpricing.pde import bs_fd, thomas, zcb_option_pde, zcb_pde
from derivpricing.pde.theta_scheme import Boundary, solve
from derivpricing.rates import CIR, G2pp, cap, caplet, forward_swap_rate

# --------------------------------------------------------------------------- PDE


def test_thomas_matches_dense_solve():
    rng = np.random.default_rng(0)
    n = 50
    lo, up = rng.random(n), rng.random(n)
    di = 3 + rng.random(n)
    A = np.diag(di) + np.diag(lo[1:], -1) + np.diag(up[:-1], 1)
    b = rng.random(n)
    assert np.allclose(thomas(lo, di, up, b), np.linalg.solve(A, b))


@pytest.mark.parametrize("scheme,grid,n_time", [("explicit", "log", 4000), ("implicit", "log", 1000),
                                                ("cn", "log", 400), ("cn", "S", 400), ("implicit", "S", 1000)])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_black_scholes_pde_schemes(scheme, grid, n_time, kind):
    ref = bs_price(100, 105, 0.04, 0.3, 1.0, kind)
    assert bs_fd(100, 105, 0.04, 0.3, 1.0, kind, scheme=scheme, grid=grid, n_time=n_time) == pytest.approx(ref, abs=5e-3)


def test_explicit_scheme_warns_when_unstable():
    with pytest.warns(RuntimeWarning, match="unstable"):
        bs_fd(100, 100, 0.05, 0.2, 1.0, scheme="explicit", n_space=400, n_time=50)


def test_american_put_pde_matches_binomial_and_dominates_european():
    crr = binomial_crr(40, 40, 0.06, 0.2, 1.0, 2000)
    assert crr == pytest.approx(2.3190, abs=2e-3)  # course benchmark
    am = bs_fd(40, 40, 0.06, 0.2, 1.0, "put", american=True)
    assert am == pytest.approx(crr, abs=2e-3)
    assert am > bs_price(40, 40, 0.06, 0.2, 1.0, "put")
    # an American call on a non-dividend stock is never exercised early
    assert bs_fd(40, 40, 0.06, 0.2, 1.0, "call", american=True) == pytest.approx(bs_price(40, 40, 0.06, 0.2, 1.0), abs=3e-3)


def test_theta_scheme_heat_equation_against_exact_solution():
    # u_t = u_xx on [0, pi], u(0,x) = sin x  ->  u = e^{-t} sin x
    x = np.linspace(0, np.pi, 201)
    u = solve(x, np.sin(x), 0.5, 200, 1.0, 0.0, 0.0, 0.5, Boundary("dirichlet", 0.0), Boundary("dirichlet", 0.0))
    assert np.max(np.abs(u - np.exp(-0.5) * np.sin(x))) < 1e-4


# --------------------------------------------------------------------------- American MC


def test_lsmc_bases_converge_to_binomial():
    crr = binomial_crr(40, 40, 0.06, 0.2, 1.0, 2000)
    paths = mc.simulate(GBM(40, 0.06, 0.2), 1.0, 60_000, 50, seed=11, antithetic=True)
    prices = {b: lsmc_american(paths, 40, 0.06, "put", b, 4) for b in ("laguerre", "hermite", "monomial")}
    for p in prices.values():
        assert crr - 0.05 < p < crr + 0.02  # low-biased, close
    assert min(prices.values()) > bs_price(40, 40, 0.06, 0.2, 1.0, "put")


def test_lsmc_heston_two_factor():
    h = Heston(48, 0.05, 0.03, 5.8, 0.0625, 0.42, -0.6)
    p = mc.simulate(h, 1.0, 60_000, 50, seed=2, antithetic=True)
    am = lsmc_american(p, 48, 0.03, "put", use_variance=True)
    assert am == pytest.approx(4.03, abs=0.06)  # course benchmark
    assert am > h.european_price(48, 1.0, "put")


# --------------------------------------------------------------------------- CIR


@pytest.fixture
def cir():
    return CIR(0.05, 0.92, 0.055, 0.12)


def test_cir_closed_form_course_benchmarks(cir):
    assert cir.zcb(0.5) == pytest.approx(0.974837, abs=1e-6)
    assert cir.zcb(1.0) == pytest.approx(0.949646, abs=1e-6)
    assert cir.zcb(4.0) == pytest.approx(0.807658, abs=1e-6)
    assert cir.coupon_bond([30] * 7 + [1030], [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]) == pytest.approx(1021.1391, abs=1e-3)
    assert cir.zcb_option(0.5, 1.0, 980, "call", 1000) == pytest.approx(0.3941, abs=1e-4)


def test_cir_pde_matches_closed_forms(cir):
    assert zcb_pde(cir, 2.0) == pytest.approx(cir.zcb(2.0), abs=1e-6)
    assert zcb_option_pde(cir, 0.5, 1.0, 980, "call", 1000) == pytest.approx(cir.zcb_option(0.5, 1.0, 980, "call", 1000), abs=5e-4)
    put = cir.zcb_option(0.5, 1.0, 960, "put", 1000)
    assert zcb_option_pde(cir, 0.5, 1.0, 960, "put", 1000, scheme="cn") == pytest.approx(put, abs=5e-4)


@pytest.mark.parametrize("scheme", ["exact", "full_truncation"])
def test_cir_monte_carlo(cir, scheme):
    assert abs(cir.mc_zcb(1.0, scheme=scheme, n_paths=40_000).z_score(cir.zcb(1.0))) < 3.5
    opt = cir.mc_zcb_option(0.5, 1.0, 975, "call", 1000, n_paths=100_000, scheme=scheme)
    assert abs(opt.z_score(cir.zcb_option(0.5, 1.0, 975, "call", 1000))) < 3.5


def test_cir_exact_simulation_moments(cir):
    r = cir.simulate(2.0, 4, 200_000, np.random.default_rng(5))
    mean = cir.theta + (cir.r0 - cir.theta) * np.exp(-cir.kappa * 2.0)
    assert r[:, -1].mean() == pytest.approx(mean, abs=3e-4)
    assert r.min() >= 0


def test_jamshidian_equals_single_zcb_option(cir):
    assert cir.coupon_bond_option(0.5, [1000], [1.0], 980) == pytest.approx(cir.zcb_option(0.5, 1.0, 980, "call", 1000), rel=1e-8)


# --------------------------------------------------------------------------- G2++


def test_g2pp_closed_form_course_benchmarks():
    g = G2pp(0.1, 0.3, 0.05, 0.09, 0.7, 0.055)
    assert g.V(1.0) == pytest.approx(0.00475614, abs=1e-8)
    assert 1000 * g.zcb(1.0) == pytest.approx(948.738636, abs=1e-5)
    assert g.Sigma(0.5, 1.0) == pytest.approx(0.04103345, abs=1e-8)
    assert g.zcb_option(0.5, 1.0, 950, "put", 1000) == pytest.approx(6.17163, abs=1e-5)
    call = g.zcb_option(0.5, 1.0, 950, "call", 1000)
    assert call - g.zcb_option(0.5, 1.0, 950, "put", 1000) == pytest.approx(1000 * g.zcb(1.0) - 950 * g.zcb(0.5))


def test_g2pp_put_increases_with_rho_and_mc_agrees():
    puts = [G2pp(0.1, 0.3, 0.05, 0.09, rho, 0.055).zcb_option(0.5, 1, 950, "put", 1000) for rho in np.linspace(-0.7, 0.7, 8)]
    assert np.all(np.diff(puts) > 0)
    g = G2pp(0.1, 0.3, 0.05, 0.09, 0.3, 0.055)
    assert abs(g.mc_zcb_option(0.5, 1, 950, "put", 1000, n_paths=200_000, n_steps=50).z_score(
        g.zcb_option(0.5, 1, 950, "put", 1000))) < 3.5
    assert abs(g.mc_zcb_option(0.5, 1, 950, "put", 1000, n_paths=200_000, n_steps=100, exact=False).z_score(
        g.zcb_option(0.5, 1, 950, "put", 1000))) < 3.5


def test_g2pp_reduces_to_vasicek_when_second_factor_off():
    a, sig, r0 = 0.2, 0.01, 0.04
    g = G2pp(a, 1.0, sig, 1e-12, 0.0, r0)
    tau = 3.0
    B = (1 - np.exp(-a * tau)) / a
    vasicek = np.exp((r0 - sig**2 / (2 * a**2)) * (B - tau) - sig**2 * B**2 / (4 * a) - B * 0.0 - r0 * B)
    # Vasicek with long-run mean r0 started at r0 equals G2++ (x0 = 0, phi = r0) with a single factor
    assert float(g.zcb(tau)) == pytest.approx(float(vasicek), rel=1e-10)


def test_caps_floors_parity():
    g = G2pp(0.1, 0.3, 0.01, 0.015, -0.5, 0.03)
    resets, d = np.arange(0.25, 3.0, 0.25), 0.25
    K = forward_swap_rate(g, 0.25, 3.0, d)
    # at the par swap rate, cap = floor (payer = receiver swaption-free parity: cap - floor = swap = 0)
    assert cap(g, resets, d, K) == pytest.approx(cap(g, resets, d, K, floor=True), rel=1e-8)
    # zero-strike caplet = floating leg P(0,T) - P(0,T+d) when rates stay positive (CIR); in a
    # Gaussian model P(T,T+d) > 1 is possible, so the G2++ zero-strike caplet is slightly larger
    c = CIR(0.05, 0.92, 0.055, 0.12)
    assert caplet(c, 1.0, d, 0.0) == pytest.approx(float(c.zcb(1.0) - c.zcb(1.25)), rel=1e-8)
    assert caplet(g, 1.0, d, 0.0) > float(g.zcb(1.0) - g.zcb(1.25))
