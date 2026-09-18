"""2D grid / problem setup.

2D analogue of grid.py. Port of matlab/shared/2d/setup_problem.m. `ops` is
attached separately by setup.operators_2d once that module
has been called with this Problem2D, same convention as grid.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from .problems_2d import ProblemDef2D


@dataclass
class Problem2D:
    nt: int
    nx: int
    ny: int
    dt: float
    dx: float
    dy: float
    L: float
    xx: jnp.ndarray         # (nx,) cell-center x-coordinates on [0, L]
    yy: jnp.ndarray         # (ny,) cell-center y-coordinates on [0, L]
    rho0: jnp.ndarray       # (nx, ny) discrete probability density, sum(rho0)*dx*dy == 1
    rho1: jnp.ndarray
    rho0_pdf: jnp.ndarray   # (nx, ny) raw pdf samples, integrates to ~1 over R^2
    rho1_pdf: jnp.ndarray
    mu0: tuple[float, float]
    mu1: tuple[float, float]
    sigma: float
    name: str
    lambda_x: jnp.ndarray   # (nx,) DCT eigenvalues, x
    lambda_y: jnp.ndarray   # (ny,) DCT eigenvalues, y
    lambda_t: jnp.ndarray   # (nt,) DCT eigenvalues, time
    ops: Any | None = None


def setup_problem_2d(prob_def: ProblemDef2D, nt: int, nx: int, ny: int, L: float = 1.0) -> Problem2D:
    dt = 1.0 / nt
    dx = L / nx
    dy = L / ny

    xg = jnp.linspace(0.0, L, nx + 1)
    xx = 0.5 * (xg[:-1] + xg[1:])
    yg = jnp.linspace(0.0, L, ny + 1)
    yy = 0.5 * (yg[:-1] + yg[1:])

    rho0_pdf = prob_def.rho0_func(xx[:, None], yy[None, :])
    rho1_pdf = prob_def.rho1_func(xx[:, None], yy[None, :])
    # Density convention: rho0[i,j] ~= rho0_density(x_i, y_j), matching
    # grid.py's 1D convention (sum(rho0)*dx*dy == 1, not mass-per-cell).
    rho0 = rho0_pdf / (rho0_pdf.sum() * dx * dy)
    rho1 = rho1_pdf / (rho1_pdf.sum() * dx * dy)

    # DCT eigenvalues for the spectral solver in projection_2d. Same
    # dimensionless-in-grid-index form as grid.py's 1D lambda_x (the
    # argument is k*pi/n, independent of L/dx -- only the outer 1/h^2
    # scaling carries the physical grid spacing), so this generalizes
    # correctly for any L, not just the L=1 case matlab/shared/2d/
    # setup_problem.m hardcodes.
    lambda_x = (2.0 - 2.0 * jnp.cos(jnp.pi * jnp.arange(nx) / nx)) / dx**2
    lambda_y = (2.0 - 2.0 * jnp.cos(jnp.pi * jnp.arange(ny) / ny)) / dy**2
    lambda_t = (2.0 - 2.0 * jnp.cos(jnp.pi * jnp.arange(nt) / nt)) / dt**2

    return Problem2D(
        nt=nt, nx=nx, ny=ny, dt=dt, dx=dx, dy=dy, L=L, xx=xx, yy=yy,
        rho0=rho0, rho1=rho1, rho0_pdf=rho0_pdf, rho1_pdf=rho1_pdf,
        mu0=prob_def.mu0, mu1=prob_def.mu1, sigma=prob_def.sigma,
        name=prob_def.name,
        lambda_x=lambda_x, lambda_y=lambda_y, lambda_t=lambda_t,
    )
