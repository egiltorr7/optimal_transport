"""Grid / problem setup.

Port of matlab/shared/1d/setup_problem.m. No config-of-callables here:
this scope only has one discretization scheme, so `ops` is attached
separately by optimal_transport.operators once that module exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .problems import ProblemDef


@dataclass
class Problem:
    nt: int
    nx: int
    dt: float
    dx: float
    L: float
    xx: np.ndarray          # (nx,) cell-center spatial coordinates on [0, L]
    rho0: np.ndarray        # (nx,) discrete probability density, sum(rho0)*dx == 1
    rho1: np.ndarray
    rho0_pdf: np.ndarray    # (nx,) raw pdf samples, integrates to ~1 over R
    rho1_pdf: np.ndarray
    mu0: float
    mu1: float
    sigma: float
    name: str
    lambda_x: np.ndarray    # (nx,)   DCT eigenvalues, space
    lambda_t: np.ndarray    # (nt, 1) DCT eigenvalues, time
    ops: Any | None = None


def setup_problem(prob_def: ProblemDef, nt: int, nx: int, L: float = 1.0) -> Problem:
    dt = 1.0 / nt
    dx = L / nx

    # Cell-center spatial grid on [0, L].
    x = np.linspace(0.0, L, nx + 1)
    xx = 0.5 * (x[:-1] + x[1:])

    rho0_pdf = prob_def.rho0_func(xx)
    rho1_pdf = prob_def.rho1_func(xx)
    # Density convention: rho0[i] ~= rho0_density(x_i), sum(rho0)*dx == 1
    # (not the mass-per-cell convention sum(rho0)==1 used previously).
    rho0 = rho0_pdf / (rho0_pdf.sum() * dx)
    rho1 = rho1_pdf / (rho1_pdf.sum() * dx)

    # DCT eigenvalues for the spectral solver in projection.
    # lambda_k = (2 - 2*cos(k*pi/n)) / h^2, discrete Neumann Laplacian.
    lambda_x = (2.0 - 2.0 * np.cos(np.pi * np.arange(nx) / nx)) / dx**2         # (nx,)
    lambda_t = ((2.0 - 2.0 * np.cos(np.pi * np.arange(nt) / nt)) / dt**2)[:, None]  # (nt, 1)

    return Problem(
        nt=nt, nx=nx, dt=dt, dx=dx, L=L, xx=xx,
        rho0=rho0, rho1=rho1, rho0_pdf=rho0_pdf, rho1_pdf=rho1_pdf,
        mu0=prob_def.mu0, mu1=prob_def.mu1, sigma=prob_def.sigma,
        name=prob_def.name,
        lambda_x=lambda_x, lambda_t=lambda_t,
    )
