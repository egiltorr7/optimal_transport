"""2D marginal-density problem definitions and analytical reference solutions.

2D analogue of problems.py. Port of matlab/shared/2d/problems/prob_gaussian.m
and matlab/shared/2d/problems/analytical_sb_gaussian.m.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class ProblemDef2D:
    name: str
    rho0_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    rho1_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    mu0: tuple[float, float]
    mu1: tuple[float, float]
    sigma: float


def normal_pdf_2d(xx: jnp.ndarray, yy: jnp.ndarray, mu: tuple[float, float], sigma: float) -> jnp.ndarray:
    mux, muy = mu
    return jnp.exp(-((xx - mux) ** 2 + (yy - muy) ** 2) / (2 * sigma**2)) / (2 * jnp.pi * sigma**2)


def prob_gaussian_2d(
    mu0: tuple[float, float] = (0.35, 0.35),
    mu1: tuple[float, float] = (0.65, 0.65),
    sigma: float = 0.1,
) -> ProblemDef2D:
    """Transport an isotropic 2D Gaussian from mu0 to mu1 on [0,1]^2."""
    return ProblemDef2D(
        name="gaussian",
        rho0_func=lambda xx, yy: normal_pdf_2d(xx, yy, mu0, sigma),
        rho1_func=lambda xx, yy: normal_pdf_2d(xx, yy, mu1, sigma),
        mu0=mu0,
        mu1=mu1,
        sigma=sigma,
    )


def analytical_sb_gaussian_2d(problem, vareps: float):
    """Exact Schrodinger bridge solution for equal-covariance isotropic 2D Gaussians.

    Problem: N(mu0, sigma^2*I) -> N(mu1, sigma^2*I). Because the covariance
    is isotropic and equal at both ends, x and y decouple and the solution
    is the product of two independent 1D SB solutions (one per axis) --
    same alpha/sigma_t^2 formula as problems.py's 1D analytical_sb_gaussian,
    applied along each axis and multiplied. This also exercises exactly the
    x/y coupling that projection_2d's combined DCT eigenvalue
    lambda_xy = lambda_x + lambda_y needs to get right, so it doubles as a
    correctness check for that coupling, not just for isotropic problems.

    Returns (rho_ana, mx_ana, my_ana) on the same staggered grid convention
    as the numerical solver:
      rho_ana  (ntm, nx, ny)   density    at times k*dt,        cell-centre
      mx_ana   (nt,  nxm, ny)  x-momentum at times (k-0.5)*dt,  x-staggered
      my_ana   (nt,  nx, nym)  y-momentum at times (k-0.5)*dt,  y-staggered
    """
    mu0x, mu0y = problem.mu0
    mu1x, mu1y = problem.mu1
    sigma = problem.sigma
    nt, dt = problem.nt, problem.dt
    nx, dx = problem.nx, problem.dx
    ny, dy = problem.ny, problem.dy
    ntm, nxm, nym = nt - 1, nx - 1, ny - 1

    alpha = jnp.sqrt(sigma**4 + vareps**2) - sigma**2

    # --- rho: times k*dt (k=1..ntm), cell-centre positions ---
    t_r = (jnp.arange(1, ntm + 1) * dt)[:, None, None]
    x_r = ((jnp.arange(1, nx + 1) - 0.5) * dx)[None, :, None]
    y_r = ((jnp.arange(1, ny + 1) - 0.5) * dy)[None, None, :]

    mu_tx = (1 - t_r) * mu0x + t_r * mu1x
    mu_ty = (1 - t_r) * mu0y + t_r * mu1y
    sig2_t = sigma**2 + 2 * alpha * t_r * (1 - t_r)

    rho_ana = jnp.exp(-((x_r - mu_tx) ** 2 + (y_r - mu_ty) ** 2) / (2 * sig2_t))
    rho_ana = rho_ana / (rho_ana.sum(axis=(1, 2), keepdims=True) * dx * dy)

    # --- mx: times (k-0.5)*dt (k=1..nt), x-staggered positions ---
    t_m = ((jnp.arange(1, nt + 1) - 0.5) * dt)[:, None, None]
    x_mx = (jnp.arange(1, nxm + 1) * dx)[None, :, None]
    y_mx = ((jnp.arange(1, ny + 1) - 0.5) * dy)[None, None, :]

    mu_tx_m = (1 - t_m) * mu0x + t_m * mu1x
    mu_ty_m = (1 - t_m) * mu0y + t_m * mu1y
    sig2_t_m = sigma**2 + 2 * alpha * t_m * (1 - t_m)

    rho_mx = jnp.exp(-((x_mx - mu_tx_m) ** 2 + (y_mx - mu_ty_m) ** 2) / (2 * sig2_t_m))
    rho_mx = rho_mx / (rho_mx.sum(axis=(1, 2), keepdims=True) * dx * dy)
    v_x = (mu1x - mu0x) + (alpha * (1 - 2 * t_m) - vareps) / sig2_t_m * (x_mx - mu_tx_m)
    mx_ana = rho_mx * v_x

    # --- my: times (k-0.5)*dt (k=1..nt), y-staggered positions ---
    x_my = ((jnp.arange(1, nx + 1) - 0.5) * dx)[None, :, None]
    y_my = (jnp.arange(1, nym + 1) * dy)[None, None, :]

    rho_my = jnp.exp(-((x_my - mu_tx_m) ** 2 + (y_my - mu_ty_m) ** 2) / (2 * sig2_t_m))
    rho_my = rho_my / (rho_my.sum(axis=(1, 2), keepdims=True) * dx * dy)
    v_y = (mu1y - mu0y) + (alpha * (1 - 2 * t_m) - vareps) / sig2_t_m * (y_my - mu_ty_m)
    my_ana = rho_my * v_y

    return rho_ana, mx_ana, my_ana
