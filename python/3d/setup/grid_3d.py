"""3D grid / problem setup.

3D analogue of grid_2d.py, extending the (t, x, y) layout with a third
spatial axis z. `ops` is attached separately by setup.operators_3d once
that module has been called with this Problem3D, same convention as
grid_2d.py / grid.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from .problems_3d import ProblemDef3D


@dataclass
class Problem3D:
    nt: int
    nx: int
    ny: int
    nz: int
    dt: float
    dx: float
    dy: float
    dz: float
    L: float
    xx: jnp.ndarray         # (nx,) cell-center x-coordinates on [0, L]
    yy: jnp.ndarray         # (ny,) cell-center y-coordinates on [0, L]
    zz: jnp.ndarray         # (nz,) cell-center z-coordinates on [0, L]
    rho0: jnp.ndarray       # (nx, ny, nz) discrete density, sum(rho0)*dx*dy*dz == 1
    rho1: jnp.ndarray
    rho0_pdf: jnp.ndarray   # (nx, ny, nz) raw pdf samples, integrates to ~1 over R^3
    rho1_pdf: jnp.ndarray
    mu0: tuple[float, float, float]
    mu1: tuple[float, float, float]
    sigma: float
    name: str
    lambda_x: jnp.ndarray   # (nx,) DCT eigenvalues, x
    lambda_y: jnp.ndarray   # (ny,) DCT eigenvalues, y
    lambda_z: jnp.ndarray   # (nz,) DCT eigenvalues, z
    lambda_t: jnp.ndarray   # (nt,) DCT eigenvalues, time
    ops: Any | None = None

    @property
    def cell_volume(self) -> float:
        """dx*dy*dz -- the spatial quadrature weight.

        The 1D modules carry a bare `dx` and grid_nu_2d.py names the 2D
        analogue `cell_area`; in 3D that role is played by the cell volume.
        Named rather than inlined because it appears in every norm and in
        every density normalization below.
        """
        return self.dx * self.dy * self.dz


def setup_problem_3d(
    prob_def: ProblemDef3D, nt: int, nx: int, ny: int, nz: int, L: float = 1.0
) -> Problem3D:
    dt = 1.0 / nt
    dx = L / nx
    dy = L / ny
    dz = L / nz

    xg = jnp.linspace(0.0, L, nx + 1)
    xx = 0.5 * (xg[:-1] + xg[1:])
    yg = jnp.linspace(0.0, L, ny + 1)
    yy = 0.5 * (yg[:-1] + yg[1:])
    zg = jnp.linspace(0.0, L, nz + 1)
    zz = 0.5 * (zg[:-1] + zg[1:])

    rho0_pdf = prob_def.rho0_func(xx[:, None, None], yy[None, :, None], zz[None, None, :])
    rho1_pdf = prob_def.rho1_func(xx[:, None, None], yy[None, :, None], zz[None, None, :])
    # Density convention: rho0[i,j,k] ~= rho0_density(x_i, y_j, z_k), matching
    # grid_2d.py's convention one dimension up (sum(rho0)*dx*dy*dz == 1, not
    # mass-per-cell).
    rho0 = rho0_pdf / (rho0_pdf.sum() * dx * dy * dz)
    rho1 = rho1_pdf / (rho1_pdf.sum() * dx * dy * dz)

    # DCT eigenvalues for the spectral solver in projection_3d. Same
    # dimensionless-in-grid-index form as grid_2d.py (the argument is
    # k*pi/n, independent of L/dx -- only the outer 1/h^2 scaling carries
    # the physical grid spacing), so this is correct for any L.
    lambda_x = (2.0 - 2.0 * jnp.cos(jnp.pi * jnp.arange(nx) / nx)) / dx**2
    lambda_y = (2.0 - 2.0 * jnp.cos(jnp.pi * jnp.arange(ny) / ny)) / dy**2
    lambda_z = (2.0 - 2.0 * jnp.cos(jnp.pi * jnp.arange(nz) / nz)) / dz**2
    lambda_t = (2.0 - 2.0 * jnp.cos(jnp.pi * jnp.arange(nt) / nt)) / dt**2

    return Problem3D(
        nt=nt, nx=nx, ny=ny, nz=nz, dt=dt, dx=dx, dy=dy, dz=dz, L=L,
        xx=xx, yy=yy, zz=zz,
        rho0=rho0, rho1=rho1, rho0_pdf=rho0_pdf, rho1_pdf=rho1_pdf,
        mu0=prob_def.mu0, mu1=prob_def.mu1, sigma=prob_def.sigma,
        name=prob_def.name,
        lambda_x=lambda_x, lambda_y=lambda_y, lambda_z=lambda_z, lambda_t=lambda_t,
    )
