import math

import numpy as np
import pytest

from derivpricing import monte_carlo as mc
from derivpricing import payoffs as po
from derivpricing.analytic import bs_price, digital_price
from derivpricing.analytic import exotics as ex
from derivpricing.analytic.black_scholes import bs_greeks
from derivpricing.greeks import bump_greeks, likelihood_ratio_delta, pathwise_delta_call
from derivpricing.models import GBM, Heston, MertonJD
from derivpricing.models.sde import convergence_order, gbm_strong_error
from derivpricing.qmc import halton, qmc_integrate, van_der_corput
from derivpricing.stats import mc_estimate, required_paths

Z_TOL = 3.5  # MC must be within 3.5 standard errors of the reference


def within(res, ref):
    return abs(res.z_score(ref)) < Z_TOL


def test_gbm_exact_euler_milstein_all_price_correctly():
    ref = bs_price(100, 100, 0.05, 0.25, 1.0)
    for scheme, steps in (("exact", 1), ("euler", 100), ("milstein", 100)):
        assert within(mc.price(GBM(100, 0.05, 0.25, scheme=scheme), po.European(100), 1.0, 100_000, n_steps=steps), ref)


def test_variance_reduction_ranks():
    g = GBM(100, 0.05, 0.25)
    se = {k: mc.price(g, po.European(100), 1.0, 40_000, **kw).stderr for k, kw in {
        "crude": dict(antithetic=False),
        "control": dict(control=mc.ControlVariate.underlying(g, 1.0)),
        "qmc": dict(qmc=True, antithetic=False),
    }.items()}
    assert se["control"] < se["crude"] / 3 and se["qmc"] < se["crude"] / 10


def test_heston_mc_matches_characteristic_function_for_every_scheme():
    h = Heston(100, 0.04, 0.03, 1.5, 0.04, 0.6, -0.7)  # violates Feller: 2*1.5*0.04 < 0.36
    assert not h.feller
    ref = h.european_price(100, 1.0)
    full = mc.price(h, po.European(100), 1.0, 100_000, n_steps=200)
    assert within(full, ref)
    # Lord, Koekkoek & van Dijk (2010): with Feller violated, reflection is visibly biased
    # while full truncation is not; partial truncation sits in between.
    err = {s: abs(mc.price(Heston(100, 0.04, 0.03, 1.5, 0.04, 0.6, -0.7, scheme=s), po.European(100), 1.0,
                           100_000, n_steps=100).price - ref) for s in ("full_truncation", "partial_truncation", "reflection")}
    assert err["reflection"] > 0.2 > err["full_truncation"]
    assert err["partial_truncation"] < err["reflection"]


def test_merton_mc_matches_series():
    m = MertonJD(100, 0.05, 0.2, 1.0, -0.1, 0.15)
    for K in (80, 100, 120):
        assert within(mc.price(m, po.European(K), 1.0, 200_000), m.european_price(K, 1.0))
    assert within(mc.price(m, po.European(100, "put"), 1.0, 200_000), m.european_price(100, 1.0, "put"))


def test_martingale_property_of_discounted_prices():
    for model in (GBM(100, 0.05, 0.3), Heston(100, 0.04, 0.05, 2, 0.04, 0.4, -0.5), MertonJD(100, 0.05, 0.2, 2, -0.2, 0.1)):
        p = mc.simulate(model, 1.0, 100_000, 50, seed=3)
        disc = np.exp(-0.05 * p.t)
        means = (p.S * disc).mean(axis=0)
        assert np.allclose(means, 100, rtol=4e-3)


def test_exotics_against_closed_forms():
    g = GBM(100, 0.05, 0.25)
    assert within(mc.price(g, po.Asian(100, average="geometric"), 1, 100_000, n_steps=12),
                  ex.geometric_asian(100, 100, 0.05, 0.25, 1, 12))
    shifted = ex.down_and_out_put(100, 100, 0.05, 0.25, 1, ex.bgk_shift(85, 0.25, 1, 100))
    assert within(mc.price(g, po.Barrier(100, 85, "put", "down", "out"), 1, 200_000, n_steps=100), shifted)


def test_barrier_in_out_parity_pathwise():
    g = GBM(100, 0.05, 0.3)
    Z = mc.draw_normals(20_000, 50, 1, np.random.default_rng(0))
    out = mc.price(g, po.Barrier(100, 120, "call", "up", "out"), 1, Z=Z, antithetic=False).price
    inn = mc.price(g, po.Barrier(100, 120, "call", "up", "in"), 1, Z=Z, antithetic=False).price
    van = mc.price(g, po.European(100), 1, Z=Z, antithetic=False).price
    assert out + inn == pytest.approx(van, rel=1e-12)


def test_arithmetic_asian_control_variate_and_lookback_ordering():
    g = GBM(100, 0.05, 0.25)
    geo = ex.geometric_asian(100, 100, 0.05, 0.25, 1, 50)
    cv = mc.ControlVariate(lambda p: math.exp(-0.05) * po.Asian(100, average="geometric")(p), geo)
    plain = mc.price(g, po.Asian(100), 1, 50_000, n_steps=50)
    with_cv = mc.price(g, po.Asian(100), 1, 50_000, n_steps=50, control=cv)
    assert with_cv.stderr < plain.stderr / 10 and abs(with_cv.price - plain.price) < 4 * plain.stderr
    assert with_cv.price > geo  # AM >= GM
    lb = mc.price(g, po.Lookback(), 1, 20_000, n_steps=250)
    assert lb.price < ex.floating_lookback_call(100, 0.05, 0.25, 1)  # discrete monitoring sees a higher minimum


def test_greek_estimators_agree_with_black_scholes():
    ref = bs_greeks(100, 100, 0.05, 0.2, 1.0)
    bump = bump_greeks(GBM(100, 0.05, 0.2), po.European(100), 1.0, n_paths=200_000)
    assert bump["delta"] == pytest.approx(ref.delta, abs=5e-3)
    assert bump["vega"] == pytest.approx(ref.vega, rel=0.03)
    assert bump["gamma"] == pytest.approx(ref.gamma, rel=0.1)
    assert within(pathwise_delta_call(100, 100, 0.05, 0.2, 1.0), ref.delta)
    h = 1e-3
    dig_delta = (digital_price(100 + h, 100, 0.05, 0.2, 1) - digital_price(100 - h, 100, 0.05, 0.2, 1)) / (2 * h)
    assert within(likelihood_ratio_delta(lambda s: (s > 100).astype(float), 100, 0.05, 0.2, 1.0, 400_000), dig_delta)


def test_sde_strong_orders():
    steps = [8, 16, 32, 64]
    assert convergence_order(steps, [gbm_strong_error("euler", 100, 0.05, 0.4, 1, n) for n in steps]) == pytest.approx(0.5, abs=0.15)
    assert convergence_order(steps, [gbm_strong_error("milstein", 100, 0.05, 0.4, 1, n) for n in steps]) == pytest.approx(1.0, abs=0.2)


def test_low_discrepancy_sequences():
    assert np.allclose(van_der_corput(7, 2), [0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875])
    assert np.allclose(van_der_corput(4, 3), [1 / 3, 2 / 3, 1 / 9, 4 / 9])
    pts = halton(1000, 2)
    assert pts.shape == (1000, 2) and pts.min() > 0 and pts.max() < 1
    f = lambda x: np.exp(-x[:, 0] * x[:, 1]) * (np.sin(6 * np.pi * x[:, 0]) + np.cbrt(np.cos(2 * np.pi * x[:, 1])))  # noqa: E731
    assert qmc_integrate(f, 10_000, 2) == pytest.approx(0.0263, abs=1e-3)  # course Project-1 integral


def test_stats_helpers():
    res = mc_estimate(np.arange(10.0))
    assert res.price == 4.5 and res.stderr == pytest.approx(np.std(np.arange(10.0), ddof=1) / math.sqrt(10))
    assert required_paths(5.0, 0.01) == math.ceil((1.959963984540054 * 5 / 0.01) ** 2)
