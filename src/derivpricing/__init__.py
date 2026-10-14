"""derivpricing - Monte Carlo, Fourier and finite-difference pricing for equity
derivatives and short-rate term-structure models.

Quick start
-----------
>>> from derivpricing.models import Heston
>>> from derivpricing import monte_carlo as mc, payoffs as po
>>> h = Heston(S0=100, v0=0.04, r=0.05, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
>>> mc.price(h, po.European(100), T=1.0, n_paths=100_000, n_steps=200)   # doctest: +SKIP
>>> h.european_price(100, 1.0)                                            # characteristic function

Packages
--------
analytic   Black-Scholes, Greeks, implied vol, exotic closed forms, Fourier (Heston/Merton)
models     GBM, Heston, Merton jump-diffusion, generic Euler/Milstein
monte_carlo, payoffs, qmc, greeks, american
           MC engine with antithetic / control variates / Sobol RQMC, exotic payoffs,
           pathwise and likelihood-ratio Greeks, Longstaff-Schwartz LSMC
pde        theta-scheme (explicit / implicit / Crank-Nicolson) for Black-Scholes and CIR
rates      CIR and G2++ short-rate models, caps/floors, Jamshidian coupon-bond options
"""

__version__ = "1.0.0"
