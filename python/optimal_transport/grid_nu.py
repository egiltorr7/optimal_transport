"""Problem setup for a non-uniform (in time) grid.

Sibling of grid.py: same spatial setup (rho0/rho1 sampling, dx, lambda_x --
space stays uniform throughout this whole non-uniform-*time* formulation),
but takes a TimeGrid (time_grid.py) instead of a bare nt, and carries a
dt_vec array instead of a scalar dt. No lambda_t: the uniform grid's DCT-in-t
trick (grid.py's lambda_t, used by projection.py/projection_expsemi.py to
invert the singular k=0 DCT-x mode) only diagonalizes the *uniform* discrete
time-Laplacian -- projection_nu.py inverts that mode with a direct
(gauge-fixed) linear solve instead, so lambda_t has no non-uniform analogue.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .problems import ProblemDef
from .time_grid import TimeGrid


@dataclass
class ProblemNU:
    nt: int
    nx: int
    dt_vec: np.ndarray       # (nt,)   interval widths (time), t_edges[i+1]-t_edges[i]
    t_edges: np.ndarray      # (nt+1,) time-grid edges, rho's interior nodes + BCs
    t_centers: np.ndarray    # (nt,)   time-grid cell centers, mx/phi live here
    dx: float
    L: float
    xx: np.ndarray           # (nx,) cell-center spatial coordinates on [0, L]
    rho0: np.ndarray         # (nx,) discrete probability density, sum(rho0)*dx == 1
    rho1: np.ndarray
    rho0_pdf: np.ndarray     # (nx,) raw pdf samples, integrates to ~1 over R
    rho1_pdf: np.ndarray
    mu0: float
    mu1: float
    sigma: float
    name: str
    lambda_x: np.ndarray     # (nx,) DCT eigenvalues, space (unchanged: dx uniform)
    ops: Any | None = None

    @property
    def dt_ref(self) -> float:
        return 1.0 / self.nt


def setup_problem_nu(prob_def: ProblemDef, time_grid: TimeGrid, nx: int, L: float = 1.0) -> ProblemNU:
    dx = L / nx

    # Cell-center spatial grid on [0, L] -- identical to grid.py's setup_problem.
    x = np.linspace(0.0, L, nx + 1)
    xx = 0.5 * (x[:-1] + x[1:])

    rho0_pdf = prob_def.rho0_func(xx)
    rho1_pdf = prob_def.rho1_func(xx)
    rho0 = rho0_pdf / (rho0_pdf.sum() * dx)
    rho1 = rho1_pdf / (rho1_pdf.sum() * dx)

    lambda_x = (2.0 - 2.0 * np.cos(np.pi * np.arange(nx) / nx)) / dx**2  # (nx,)

    return ProblemNU(
        nt=time_grid.nt, nx=nx,
        dt_vec=time_grid.dt_vec, t_edges=time_grid.t_edges, t_centers=time_grid.t_centers,
        dx=dx, L=L, xx=xx,
        rho0=rho0, rho1=rho1, rho0_pdf=rho0_pdf, rho1_pdf=rho1_pdf,
        mu0=prob_def.mu0, mu1=prob_def.mu1, sigma=prob_def.sigma,
        name=prob_def.name,
        lambda_x=lambda_x,
    )
