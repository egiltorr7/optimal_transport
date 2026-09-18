"""3D marginal-density problem definitions and analytical reference solutions.

3D analogue of problems_2d.py.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class ProblemDef3D:
    name: str
    rho0_func: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    rho1_func: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    mu0: tuple[float, float, float]
    mu1: tuple[float, float, float]
    sigma: float


def normal_pdf_3d(
    xx: jnp.ndarray, yy: jnp.ndarray, zz: jnp.ndarray,
    mu: tuple[float, float, float], sigma: float,
) -> jnp.ndarray:
    mux, muy, muz = mu
    r2 = (xx - mux) ** 2 + (yy - muy) ** 2 + (zz - muz) ** 2
    return jnp.exp(-r2 / (2 * sigma**2)) / (2 * jnp.pi * sigma**2) ** 1.5


def prob_gaussian_3d(
    mu0: tuple[float, float, float] = (0.35, 0.35, 0.35),
    mu1: tuple[float, float, float] = (0.65, 0.65, 0.65),
    sigma: float = 0.1,
) -> ProblemDef3D:
    """Transport an isotropic 3D Gaussian from mu0 to mu1 on [0,1]^3.

    Defaults mirror prob_gaussian_2d's. Note the same caveat as in 2D: with
    sigma=0.1 and these means, the Gaussians are not fully boundary-free on
    [0,1]^3 at small vareps -- the tails are clipped at the walls, so the
    discrete marginals differ slightly from the continuous ones and the
    closed-form reference below picks up a corresponding O(tail) floor.
    Pass a smaller sigma (or means nearer the centre) to shrink it.
    """
    return ProblemDef3D(
        name="gaussian",
        rho0_func=lambda xx, yy, zz: normal_pdf_3d(xx, yy, zz, mu0, sigma),
        rho1_func=lambda xx, yy, zz: normal_pdf_3d(xx, yy, zz, mu1, sigma),
        mu0=mu0,
        mu1=mu1,
        sigma=sigma,
    )


def analytical_sb_gaussian_3d(problem, vareps: float):
    """Exact Schrodinger bridge solution for equal-covariance isotropic 3D Gaussians.

    Problem: N(mu0, sigma^2*I) -> N(mu1, sigma^2*I). Because the covariance is
    isotropic and equal at both ends, x, y and z decouple and the solution is
    the product of three independent 1D SB solutions -- same alpha/sigma_t^2
    formula as problems.py's 1D analytical_sb_gaussian, applied along each
    axis. Like its 2D sibling, this exercises exactly the cross-axis coupling
    that projection_3d's combined DCT eigenvalue
    lambda_xyz = lambda_x + lambda_y + lambda_z has to get right, so it
    doubles as a correctness check for that coupling.

    Returns (rho_ana, mx_ana, my_ana, mz_ana) on the same staggered grid
    convention as the numerical solver:
      rho_ana  (ntm, nx, ny, nz)   density    at times k*dt,       cell-centre
      mx_ana   (nt,  nxm, ny, nz)  x-momentum at times (k-0.5)*dt, x-staggered
      my_ana   (nt,  nx, nym, nz)  y-momentum at times (k-0.5)*dt, y-staggered
      mz_ana   (nt,  nx, ny, nzm)  z-momentum at times (k-0.5)*dt, z-staggered
    """
    mu0x, mu0y, mu0z = problem.mu0
    mu1x, mu1y, mu1z = problem.mu1
    sigma = problem.sigma
    nt, dt = problem.nt, problem.dt
    nx, dx = problem.nx, problem.dx
    ny, dy = problem.ny, problem.dy
    nz, dz = problem.nz, problem.dz
    ntm, nxm, nym, nzm = nt - 1, nx - 1, ny - 1, nz - 1
    cell_vol = dx * dy * dz

    alpha = jnp.sqrt(sigma**4 + vareps**2) - sigma**2

    # Cell-centre and node coordinate vectors, each broadcast along its own axis.
    def _centres(n, h, axis):
        c = (jnp.arange(1, n + 1) - 0.5) * h
        return c.reshape([1] + [n if a == axis else 1 for a in range(1, 4)])

    def _nodes(nm, h, axis):
        c = jnp.arange(1, nm + 1) * h
        return c.reshape([1] + [nm if a == axis else 1 for a in range(1, 4)])

    def _gaussian_at(t, xc, yc, zc):
        """Normalized SB density at times `t` on the given coordinate grids."""
        mu_tx = (1 - t) * mu0x + t * mu1x
        mu_ty = (1 - t) * mu0y + t * mu1y
        mu_tz = (1 - t) * mu0z + t * mu1z
        sig2_t = sigma**2 + 2 * alpha * t * (1 - t)
        r2 = (xc - mu_tx) ** 2 + (yc - mu_ty) ** 2 + (zc - mu_tz) ** 2
        g = jnp.exp(-r2 / (2 * sig2_t))
        g = g / (g.sum(axis=(1, 2, 3), keepdims=True) * cell_vol)
        return g, (mu_tx, mu_ty, mu_tz), sig2_t

    # --- rho: times k*dt (k=1..ntm), cell-centre in all three axes ---
    t_r = (jnp.arange(1, ntm + 1) * dt).reshape(ntm, 1, 1, 1)
    rho_ana, _, _ = _gaussian_at(t_r, _centres(nx, dx, 1), _centres(ny, dy, 2), _centres(nz, dz, 3))

    # --- momenta: times (k-0.5)*dt (k=1..nt), staggered in their own axis ---
    t_m = ((jnp.arange(1, nt + 1) - 0.5) * dt).reshape(nt, 1, 1, 1)

    # The SB velocity field is  v = (mu1-mu0) + (alpha*(1-2t) - eps)/sig2_t * (x - mu_t)
    # componentwise, and m = rho * v -- each component evaluated on its own
    # staggered grid (node positions along its axis, cell-centres along the
    # other two), which is why the density is re-sampled three times rather
    # than reused from rho_ana.
    def _momentum(xc, yc, zc, comp):
        rho_s, (mu_tx, mu_ty, mu_tz), sig2_t = _gaussian_at(t_m, xc, yc, zc)
        drift = (alpha * (1 - 2 * t_m) - vareps) / sig2_t
        if comp == 0:
            v = (mu1x - mu0x) + drift * (xc - mu_tx)
        elif comp == 1:
            v = (mu1y - mu0y) + drift * (yc - mu_ty)
        else:
            v = (mu1z - mu0z) + drift * (zc - mu_tz)
        return rho_s * v

    mx_ana = _momentum(_nodes(nxm, dx, 1), _centres(ny, dy, 2), _centres(nz, dz, 3), comp=0)
    my_ana = _momentum(_centres(nx, dx, 1), _nodes(nym, dy, 2), _centres(nz, dz, 3), comp=1)
    mz_ana = _momentum(_centres(nx, dx, 1), _centres(ny, dy, 2), _nodes(nzm, dz, 3), comp=2)

    return rho_ana, mx_ana, my_ana, mz_ana
