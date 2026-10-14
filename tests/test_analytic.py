import math

import numpy as np
import pytest

from derivpricing.analytic import (
    bs_cf,
    bs_greeks,
    bs_price,
    carr_madan_fft,
    cf_price,
    digital_price,
    fft_price,
    heston_cf,
    implied_vol,
    merton_cf,
    merton_series_price,
)
from derivpricing.analytic import exotics as ex
from derivpricing.analytic.normal import norm_cdf, norm_cdf_as, norm_cdf_poly16


def test_black_scholes_course_benchmark_and_parity():
    assert bs_price(100, 100, 0.055, 0.2, 5) == pytest.approx(30.373, abs=1e-3)
    c, p = bs_price(100, 95, 0.04, 0.3, 0.7), bs_price(100, 95, 0.04, 0.3, 0.7, "put")
    assert c - p == pytest.approx(100 - 95 * math.exp(-0.04 * 0.7), abs=1e-10)  # put-call parity


def test_normal_cdf_approximations():
    x = np.linspace(-6, 6, 2001)
    assert np.max(np.abs(norm_cdf_as(x) - norm_cdf(x))) < 7.5e-8
    assert np.max(np.abs(norm_cdf_poly16(x) - norm_cdf(x))) < 1.5e-7


def test_greeks_match_finite_differences():
    g = bs_greeks(100, 105, 0.03, 0.25, 0.8, "call")
    h = 1e-4
    assert g.delta == pytest.approx((bs_price(100 + h, 105, 0.03, 0.25, 0.8) - bs_price(100 - h, 105, 0.03, 0.25, 0.8)) / (2 * h), rel=1e-6)
    assert g.vega == pytest.approx((bs_price(100, 105, 0.03, 0.25 + h, 0.8) - bs_price(100, 105, 0.03, 0.25 - h, 0.8)) / (2 * h), rel=1e-6)
    assert g.theta == pytest.approx(-(bs_price(100, 105, 0.03, 0.25, 0.8 + h) - bs_price(100, 105, 0.03, 0.25, 0.8 - h)) / (2 * h), rel=1e-5)


def test_implied_vol_round_trip_and_digital():
    assert implied_vol(bs_price(100, 120, 0.02, 0.37, 1.5), 100, 120, 0.02, 1.5) == pytest.approx(0.37, abs=1e-9)
    h = 1e-3  # digital = -dC/dK
    assert digital_price(100, 100, 0.05, 0.2, 1) == pytest.approx(
        -(bs_price(100, 100 + h, 0.05, 0.2, 1) - bs_price(100, 100 - h, 0.05, 0.2, 1)) / (2 * h), abs=1e-6)


def test_fourier_reproduces_black_scholes():
    cf = bs_cf(100, 0.05, 0.25, 1.0)
    ref = bs_price(100, 110, 0.05, 0.25, 1.0)
    assert cf_price(cf, 100, 110, 0.05, 1.0) == pytest.approx(ref, abs=1e-6)
    assert fft_price(cf, 100, 110, 0.05, 1.0) == pytest.approx(ref, abs=1e-4)
    strikes, calls = carr_madan_fft(cf, 100, 0.05, 1.0)
    mask = (strikes > 60) & (strikes < 160)
    assert np.allclose(calls[mask], bs_price(100, strikes[mask], 0.05, 0.25, 1.0), atol=1e-3)


def test_heston_reduces_to_black_scholes_when_vol_of_vol_vanishes():
    cf = heston_cf(100, 0.05, 1.0, v0=0.04, kappa=1.5, theta=0.04, xi=1e-4, rho=0.0)
    assert cf_price(cf, 100, 100, 0.05, 1.0) == pytest.approx(bs_price(100, 100, 0.05, 0.2, 1.0), abs=1e-4)


def test_merton_series_equals_characteristic_function_and_bs_limit():
    args = (100, 105, 0.05, 0.2, 1.0, 0.8, -0.05, 0.15)
    cf = merton_cf(100, 0.05, 0.2, 1.0, 0.8, -0.05, 0.15)
    assert merton_series_price(*args) == pytest.approx(cf_price(cf, 100, 105, 0.05, 1.0), abs=1e-6)
    assert merton_series_price(100, 105, 0.05, 0.2, 1.0, 0.0, 0.0, 0.1) == pytest.approx(bs_price(100, 105, 0.05, 0.2, 1.0), abs=1e-10)


def test_barrier_closed_form_course_benchmark_and_parity():
    assert ex.down_and_out_put(50, 50, 0.1, 0.4, 5 / 12, 30) == pytest.approx(3.228, abs=1e-3)
    assert ex.down_and_out_put(50, 50, 0.1, 0.3, 5 / 12, 30) == pytest.approx(2.729, abs=1e-3)
    vanilla = bs_price(50, 50, 0.1, 0.4, 5 / 12, "put")
    assert ex.down_and_in_put(50, 50, 0.1, 0.4, 5 / 12, 30) + ex.down_and_out_put(50, 50, 0.1, 0.4, 5 / 12, 30) == pytest.approx(vanilla)
    assert ex.down_and_out_put(50, 50, 0.1, 0.4, 5 / 12, 1e-6) == pytest.approx(vanilla, abs=1e-8)  # unreachable barrier


def test_geometric_asian_limits():
    # one monitoring date at T: the geometric average is S_T
    assert ex.geometric_asian(100, 100, 0.05, 0.2, 1.0, 1) == pytest.approx(bs_price(100, 100, 0.05, 0.2, 1.0), abs=1e-10)
    assert ex.geometric_asian(100, 100, 0.05, 0.2, 1.0, 50) < bs_price(100, 100, 0.05, 0.2, 1.0)  # averaging lowers vol
