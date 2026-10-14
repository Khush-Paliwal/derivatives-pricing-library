"""Interest-rate products built from zero-coupon bond options.

A caplet on the simple rate L(T, T + d) with strike K pays d (L - K)^+ at T + d.
Its value at T is (1 + d K) (1/(1 + d K) - P(T, T + d))^+, so

    caplet = (1 + d K) * ZCBput(expiry T, bond maturity T + d, strike 1 / (1 + d K)),
    floorlet = (1 + d K) * ZCBcall(same),

and a cap/floor is the sum over the reset dates. Any model with a
``zcb_option(T, S, K, kind)`` method (CIR, G2++) can price them.
"""

from __future__ import annotations

import numpy as np


def caplet(model, T: float, delta: float, K: float, notional: float = 1.0, floor: bool = False) -> float:
    strike = 1.0 / (1.0 + delta * K)
    kind = "call" if floor else "put"
    return float(notional * (1.0 + delta * K) * model.zcb_option(T, T + delta, strike, kind))


def cap(model, reset_times, delta: float, K: float, notional: float = 1.0, floor: bool = False) -> float:
    """Cap (or floor) with resets at ``reset_times`` (first caplet usually starts after the first reset)."""
    return float(sum(caplet(model, T, delta, K, notional, floor) for T in np.asarray(reset_times, float) if T > 0))


def forward_swap_rate(model, start: float, end: float, delta: float) -> float:
    """Par rate of a swap paying fixed every ``delta`` from start to end."""
    pay = np.arange(start + delta, end + 1e-12, delta)
    annuity = delta * sum(float(model.zcb(t)) for t in pay)
    return float((float(model.zcb(start)) - float(model.zcb(end))) / annuity)
