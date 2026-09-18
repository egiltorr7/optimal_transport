"""2D grid / problem setup on a non-uniform (in time) grid.

Cross of grid_2d.py and grid_nu.py: grid_2d.py's 2D spatial setup (rho0/rho1
sampled on the (nx, ny) cell-centre grid, dx/dy, lambda_x/lambda_y) with
grid_nu.py's TimeGrid in place of a bare nt, carrying a dt_vec array instead
of a scalar dt.

Space stays uniform in BOTH x and y -- the non-uniform formulation is in
*time* only, exactly as in the 1D case -- so lambda_x/lambda_y carry over
from grid_2d.py untouched.

No lambda_t, for grid_nu.py's reason: the uniform grid's DCT-in-t trick
(used by projection_2d.py to invert the singular (kx,ky)=(0,0) DCT mode)
only diagonalizes the *uniform* discrete time-Laplacian. projection_nu_2d.py
inverts that mode with a pseudo-inverse of the assembled M0 instead, so
lambda_t has no non-uniform analogue.

The TimeGrid arrives as numpy (time_grid.py builds the edges host-side --
pure grid construction, no array work worth dispatching to a device) and is
converted to jnp here, so everything downstream of setup is jnp-only and
follows jax_config.configure()'s device/precision choice.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from .problems_2d import ProblemDef2D
from .time_grid import TimeGrid


@dataclass
class ProblemNU2D:
    nt: int
    nx: int
    ny: int
    dt_vec: jnp.ndarray      # (nt,)   interval widths (time), diff(t_edges)
    t_edges: jnp.ndarray     # (nt+1,) time-grid edges, rho's interior nodes + BCs
    t_centers: jnp.ndarray   # (nt,)   time-grid cell centers, mx/my/phi live here
    dx: float
    dy: float
    L: float
    xx: jnp.ndarray          # (nx,) cell-center x-coordinates on [0, L]
    yy: jnp.ndarray          # (ny,) cell-center y-coordinates on [0, L]
    rho0: jnp.ndarray        # (nx, ny) discrete probability density, sum(rho0)*dx*dy == 1
    rho1: jnp.ndarray
    rho0_pdf: jnp.ndarray    # (nx, ny) raw pdf samples, integrates to ~1 over R^2
    rho1_pdf: jnp.ndarray
    mu0: tuple[float, float]
    mu1: tuple[float, float]
    sigma: float
    name: str
    lambda_x: jnp.ndarray    # (nx,) DCT eigenvalues, x (unchanged: dx uniform)
    lambda_y: jnp.ndarray    # (ny,) DCT eigenvalues, y (unchanged: dy uniform)
    ops: Any | None = None

    @property
    def dt_ref(self) -> float:
        return 1.0 / self.nt

    @property
    def dt_dual(self) -> jnp.ndarray:
        """(nt-1,) center-to-center widths, rho's interior-edge quadrature weight.

        Used by projection_nu_2d.py's N_q and pipeline_nu_2d.py's norms; a
        property rather than a stored field so it can never drift out of sync
        with dt_vec. Equals dt_vec's constant value on a uniform grid.
        """
        return 0.5 * (self.dt_vec[:-1] + self.dt_vec[1:])

    @property
    def cell_area(self) -> float:
        """dx*dy -- the spatial measure the N/M weights carry (uniform in space).

        The 1D non-uniform modules carry a bare `dx` in exactly these places;
        in 2D that role is played by the cell area. Named rather than inlined
        because it appears in both N_q and N_b and in three separate modules.
        """
        return self.dx * self.dy


def setup_problem_nu_2d(prob_def: ProblemDef2D, time_grid: TimeGrid, nx: int, ny: int, L: float = 1.0) -> ProblemNU2D:
    dx = L / nx
    dy = L / ny

    xg = jnp.linspace(0.0, L, nx + 1)
    xx = 0.5 * (xg[:-1] + xg[1:])
    yg = jnp.linspace(0.0, L, ny + 1)
    yy = 0.5 * (yg[:-1] + yg[1:])

    rho0_pdf = prob_def.rho0_func(xx[:, None], yy[None, :])
    rho1_pdf = prob_def.rho1_func(xx[:, None], yy[None, :])
    rho0 = rho0_pdf / (rho0_pdf.sum() * dx * dy)
    rho1 = rho1_pdf / (rho1_pdf.sum() * dx * dy)

    lambda_x = (2.0 - 2.0 * jnp.cos(jnp.pi * jnp.arange(nx) / nx)) / dx**2
    lambda_y = (2.0 - 2.0 * jnp.cos(jnp.pi * jnp.arange(ny) / ny)) / dy**2

    return ProblemNU2D(
        nt=time_grid.nt, nx=nx, ny=ny,
        dt_vec=jnp.asarray(time_grid.dt_vec),
        t_edges=jnp.asarray(time_grid.t_edges),
        t_centers=jnp.asarray(time_grid.t_centers),
        dx=dx, dy=dy, L=L, xx=xx, yy=yy,
        rho0=rho0, rho1=rho1, rho0_pdf=rho0_pdf, rho1_pdf=rho1_pdf,
        mu0=prob_def.mu0, mu1=prob_def.mu1, sigma=prob_def.sigma,
        name=prob_def.name,
        lambda_x=lambda_x, lambda_y=lambda_y,
    )
