"""Analytical 2D Schrodinger-bridge reference, sampled at a non-uniform time
grid. Cross of problems_2d.py's analytical_sb_gaussian_2d and problems_nu.py.

Same one-line generalization as the 1D problems_nu.py: the closed-form
solution is a function of continuous t -- grid resolution never entered its
derivation, only *where* it gets sampled did -- so this evaluates
problems_2d.py's formula at problem.t_edges[1:-1] / problem.t_centers (the
actual, possibly-clustered grid points) instead of at the uniform k*dt and
(k-0.5)*dt problems_2d.py assumes.

Space is still uniform, so the x/y sampling positions are unchanged, as are
the per-time-slice normalizations.
"""
from __future__ import annotations

import jax.numpy as jnp


def analytical_sb_gaussian_nu_2d(problem, vareps: float):
    """Exact SB solution for equal-covariance isotropic 2D Gaussians, on the
    problem's actual (possibly non-uniform) time grid.

    x and y decouple (see problems_2d.analytical_sb_gaussian_2d for why, and
    for why that still exercises projection's lambda_xy = lambda_x+lambda_y
    coupling rather than dodging it).

    Returns (rho_ana, mx_ana, my_ana) on pipeline_nu_2d.py's staggered-grid
    convention:
      rho_ana  (ntm, nx, ny)   density    at t_edges[1:-1], cell-centre in x,y
      mx_ana   (nt,  nxm, ny)  x-momentum at t_centers,     x-staggered
      my_ana   (nt,  nx, nym)  y-momentum at t_centers,     y-staggered
    """
    mu0x, mu0y = problem.mu0
    mu1x, mu1y = problem.mu1
    sigma = problem.sigma
    nx, dx = problem.nx, problem.dx
    ny, dy = problem.ny, problem.dy
    nxm, nym = nx - 1, ny - 1

    alpha = jnp.sqrt(sigma**4 + vareps**2) - sigma**2

    # --- rho: interior time edges, cell-centre positions ---
    t_r = problem.t_edges[1:-1][:, None, None]              # (ntm, 1, 1)
    x_r = ((jnp.arange(1, nx + 1) - 0.5) * dx)[None, :, None]
    y_r = ((jnp.arange(1, ny + 1) - 0.5) * dy)[None, None, :]

    mu_tx = (1 - t_r) * mu0x + t_r * mu1x
    mu_ty = (1 - t_r) * mu0y + t_r * mu1y
    sig2_t = sigma**2 + 2 * alpha * t_r * (1 - t_r)

    rho_ana = jnp.exp(-((x_r - mu_tx) ** 2 + (y_r - mu_ty) ** 2) / (2 * sig2_t))
    rho_ana = rho_ana / (rho_ana.sum(axis=(1, 2), keepdims=True) * dx * dy)

    # --- mx: time centres, x-staggered positions ---
    t_m = problem.t_centers[:, None, None]                   # (nt, 1, 1)
    x_mx = (jnp.arange(1, nxm + 1) * dx)[None, :, None]
    y_mx = ((jnp.arange(1, ny + 1) - 0.5) * dy)[None, None, :]

    mu_tx_m = (1 - t_m) * mu0x + t_m * mu1x
    mu_ty_m = (1 - t_m) * mu0y + t_m * mu1y
    sig2_t_m = sigma**2 + 2 * alpha * t_m * (1 - t_m)

    rho_mx = jnp.exp(-((x_mx - mu_tx_m) ** 2 + (y_mx - mu_ty_m) ** 2) / (2 * sig2_t_m))
    rho_mx = rho_mx / (rho_mx.sum(axis=(1, 2), keepdims=True) * dx * dy)
    v_x = (mu1x - mu0x) + (alpha * (1 - 2 * t_m) - vareps) / sig2_t_m * (x_mx - mu_tx_m)
    mx_ana = rho_mx * v_x

    # --- my: time centres, y-staggered positions ---
    x_my = ((jnp.arange(1, nx + 1) - 0.5) * dx)[None, :, None]
    y_my = (jnp.arange(1, nym + 1) * dy)[None, None, :]

    rho_my = jnp.exp(-((x_my - mu_tx_m) ** 2 + (y_my - mu_ty_m) ** 2) / (2 * sig2_t_m))
    rho_my = rho_my / (rho_my.sum(axis=(1, 2), keepdims=True) * dx * dy)
    v_y = (mu1y - mu0y) + (alpha * (1 - 2 * t_m) - vareps) / sig2_t_m * (y_my - mu_ty_m)
    my_ana = rho_my * v_y

    return rho_ana, mx_ana, my_ana
