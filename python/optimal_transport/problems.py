"""Marginal-density problem definitions and analytical reference solutions.

Port of matlab/shared/1d/problems/prob_gaussian.m and analytical_sb_gaussian.m.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class ProblemDef:
    name: str
    rho0_func: Callable[[np.ndarray], np.ndarray]
    rho1_func: Callable[[np.ndarray], np.ndarray]
    mu0: float
    mu1: float
    sigma: float


def normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)


def prob_gaussian(mu0: float = 1 / 3, mu1: float = 2 / 3, sigma: float = 0.05) -> ProblemDef:
    """Transport a Gaussian from x=mu0 to x=mu1 on [0,1]."""
    return ProblemDef(
        name="gaussian",
        rho0_func=lambda xx: normal_pdf(xx, mu0, sigma),
        rho1_func=lambda xx: normal_pdf(xx, mu1, sigma),
        mu0=mu0,
        mu1=mu1,
        sigma=sigma,
    )


def prob_bimodal(centers=(0.25, 0.75), mu1: float = 0.5, sigma: float = 0.05) -> ProblemDef:
    """Bimodal -> unimodal: an equal-weight two-hump rho0 merging into one hump.

    No closed-form Schrodinger bridge exists for these marginals, so
    analytical_sb_gaussian must NOT be called on this problem -- its mu0/mu1/sigma
    fields are filled only because ProblemDef requires them. Above
    study_utils.SINKHORN_VAREPS_THRESHOLD a Sinkhorn reference is available as
    usual (Sinkhorn needs no structure in the marginals); below it there is no
    reference at all, and a grid-refinement study against the finest run is the
    available substitute.
    """
    c0, c1 = centers

    def rho0_func(xx):
        return 0.5 * normal_pdf(xx, c0, sigma) + 0.5 * normal_pdf(xx, c1, sigma)

    return ProblemDef(
        name="bimodal",
        rho0_func=rho0_func,
        rho1_func=lambda xx: normal_pdf(xx, mu1, sigma),
        mu0=0.5 * (c0 + c1),   # unused; see docstring
        mu1=mu1,
        sigma=sigma,
    )


def analytical_sb_gaussian(problem, vareps: float):
    """Exact Schrodinger bridge solution for equal-variance Gaussian marginals.

    Problem: N(mu0, sigma^2) -> N(mu1, sigma^2).
    FP constraint: d_t rho + d_x m = vareps * d_xx rho.

    Returns (rho_ana, mx_ana) on the same staggered grid convention as the
    numerical solver:
      rho_ana  (ntm, nx)   density   at times k*dt,       positions (i-0.5)*dx
      mx_ana   (nt,  nxm)  momentum  at times (k-0.5)*dt,  positions j*dx
    """
    mu0, mu1, sigma = problem.mu0, problem.mu1, problem.sigma
    nt, dt = problem.nt, problem.dt
    nx, dx = problem.nx, problem.dx
    ntm, nxm = nt - 1, nx - 1

    # Excess variance amplitude due to diffusion.
    alpha = np.sqrt(sigma**4 + vareps**2) - sigma**2

    # --- rho: times k*dt (k=1..ntm), positions (i-0.5)*dx ---
    t_rho = (np.arange(1, ntm + 1) * dt)[:, None]           # (ntm, 1)
    x_rho = ((np.arange(1, nx + 1) - 0.5) * dx)[None, :]    # (1, nx)

    mu_t = (1 - t_rho) * mu0 + t_rho * mu1
    sig2_t = sigma**2 + 2 * alpha * t_rho * (1 - t_rho)

    rho_ana = normal_pdf(x_rho, mu_t, np.sqrt(sig2_t))
    rho_ana = rho_ana / (rho_ana.sum(axis=1, keepdims=True) * dx)

    # --- mx: times (k-0.5)*dt (k=1..nt), positions j*dx ---
    t_mx = ((np.arange(1, nt + 1) - 0.5) * dt)[:, None]     # (nt, 1)
    x_mx = (np.arange(1, nxm + 1) * dx)[None, :]            # (1, nxm)

    mu_t_mx = (1 - t_mx) * mu0 + t_mx * mu1
    sig2_t_mx = sigma**2 + 2 * alpha * t_mx * (1 - t_mx)

    row = normal_pdf(x_mx, mu_t_mx, np.sqrt(sig2_t_mx))
    row = row / (row.sum(axis=1, keepdims=True) * dx)

    # Velocity: translational part + diffusion-driven spreading/squeezing.
    v = (mu1 - mu0) + (alpha * (1 - 2 * t_mx) - vareps) / sig2_t_mx * (x_mx - mu_t_mx)
    mx_ana = row * v

    return rho_ana, mx_ana
