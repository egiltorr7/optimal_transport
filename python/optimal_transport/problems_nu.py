"""Analytical Schrodinger-bridge reference, sampled at a non-uniform time
grid. Sibling of problems.py's analytical_sb_gaussian.

The closed-form solution is a function of continuous t -- grid resolution
never entered its derivation, only *where* it gets sampled did. So this is
a one-line generalization: evaluate the same formula at problem.t_edges /
problem.t_centers (the actual, possibly-clustered grid points) instead of
the uniform k*dt / (k-0.5)*dt problems.py assumes.
"""
from __future__ import annotations

import numpy as np

from .problems import normal_pdf


def analytical_sb_gaussian_nu(problem, vareps: float):
    """Exact SB solution for equal-variance Gaussian marginals, on the
    problem's actual (possibly non-uniform) time grid.

    Returns (rho_ana, mx_ana) matching pipeline_nu.py's staggered-grid
    convention:
      rho_ana  (ntm, nx)   density   at problem.t_edges[1:-1], positions (i-0.5)*dx
      mx_ana   (nt,  nxm)  momentum  at problem.t_centers,      positions j*dx
    """
    mu0, mu1, sigma = problem.mu0, problem.mu1, problem.sigma
    nx, dx = problem.nx, problem.dx
    nxm = nx - 1

    alpha = np.sqrt(sigma**4 + vareps**2) - sigma**2

    t_rho = problem.t_edges[1:-1][:, None]              # (ntm, 1)
    x_rho = ((np.arange(1, nx + 1) - 0.5) * dx)[None, :]  # (1, nx)

    mu_t = (1 - t_rho) * mu0 + t_rho * mu1
    sig2_t = sigma**2 + 2 * alpha * t_rho * (1 - t_rho)

    rho_ana = normal_pdf(x_rho, mu_t, np.sqrt(sig2_t))
    rho_ana = rho_ana / (rho_ana.sum(axis=1, keepdims=True) * dx)

    t_mx = problem.t_centers[:, None]                    # (nt, 1)
    x_mx = (np.arange(1, nxm + 1) * dx)[None, :]          # (1, nxm)

    mu_t_mx = (1 - t_mx) * mu0 + t_mx * mu1
    sig2_t_mx = sigma**2 + 2 * alpha * t_mx * (1 - t_mx)

    row = normal_pdf(x_mx, mu_t_mx, np.sqrt(sig2_t_mx))
    row = row / (row.sum(axis=1, keepdims=True) * dx)

    v = (mu1 - mu0) + (alpha * (1 - 2 * t_mx) - vareps) / sig2_t_mx * (x_mx - mu_t_mx)
    mx_ana = row * v

    return rho_ana, mx_ana
