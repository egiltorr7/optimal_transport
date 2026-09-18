"""Wire 3D grid + operators + FP projection + KE prox into a linearized-ADMM solve.

3D analogue of pipeline_2d.py -- see pipeline.py for the general
linearized-ADMM formulation this wires up:

    min   I_FP(x) + KE(y)   s.t.   A*x - y = 0

  x = (rho, mx, my, mz) on the staggered grid   -- FP-constraint variable
  y = (rho, mx, my, mz) on the cell-centre grid -- KE variable
  A = affine interpolation staggered -> cell-centres (BCs baked in)
  B = -I  (on cell-centres),  b = 0  (BCs absorbed into affine A)

`projection` is passed in rather than hardcoded, same reason as
pipeline_2d.py: so proj_fokker_planck_banded_3d (or a future ETD/PCR 3D
projection) can be swapped in against the same pipeline.

Reuses ladmm.py unchanged -- it touches the state only via State3D's own
arithmetic/.norm()/.zeros_like(), which is duck-typed there.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp

from .ladmm import LadmmInfo, LadmmOpts, ladmm_solve
from .prox_3d import prox_ke_cc_3d
from .state_3d import State3D


@dataclass
class LadmmConfig:
    gamma: float     # penalty parameter
    tau: float       # proximal penalty for the x-update (> gamma * ||A_linear||^2 <= gamma)
    max_iter: int
    eps_abs: float   # absolute floor for the D/P stopping criterion (see ladmm.py)
    eps_rel: float   # relative tolerance for the D/P stopping criterion
    alpha: float = 1.0  # over-relaxation


@dataclass
class SolveResult3D:
    rho_stag: jnp.ndarray  # (ntm, nx, ny, nz)  FP-feasible density, staggered grid
    mx_stag: jnp.ndarray   # (nt, nxm, ny, nz)  FP-feasible x-momentum, staggered
    my_stag: jnp.ndarray   # (nt, nx, nym, nz)  FP-feasible y-momentum, staggered
    mz_stag: jnp.ndarray   # (nt, nx, ny, nzm)  FP-feasible z-momentum, staggered
    rho_cc: jnp.ndarray    # (nt, nx, ny, nz)   KE-optimal density, cell-centre grid
    mx_cc: jnp.ndarray     # (nt, nx, ny, nz)   KE-optimal x-momentum, cell-centre
    my_cc: jnp.ndarray     # (nt, nx, ny, nz)   KE-optimal y-momentum, cell-centre
    mz_cc: jnp.ndarray     # (nt, nx, ny, nz)   KE-optimal z-momentum, cell-centre
    info: LadmmInfo


def discretize_then_optimize_3d(
    problem,
    projection: Callable[[State3D, object, float, object], State3D],
    projection_state: object,
    vareps: float,
    ladmm_cfg: LadmmConfig,
    x0: State3D | None = None,
    y0: State3D | None = None,
    x_ref: State3D | None = None,
    delta_ref: State3D | None = None,
    delta_mask: State3D | None = None,
) -> SolveResult3D:

    ops = problem.ops
    rho0, rho1 = problem.rho0, problem.rho1
    nt, nx, ny, nz = problem.nt, problem.nx, problem.ny, problem.nz
    ntm, nxm, nym, nzm = nt - 1, nx - 1, ny - 1, nz - 1
    dt, dx, dy, dz = problem.dt, problem.dx, problem.dy, problem.dz
    cell_vol = dx * dy * dz

    if x0 is None or y0 is None:
        t_stag = jnp.linspace(0.0, 1.0, ntm).reshape(ntm, 1, 1, 1)
        x0 = State3D(
            rho=(1 - t_stag) * rho0[None, ...] + t_stag * rho1[None, ...],  # (ntm,nx,ny,nz)
            mx=jnp.zeros((nt, nxm, ny, nz)),
            my=jnp.zeros((nt, nx, nym, nz)),
            mz=jnp.zeros((nt, nx, ny, nzm)),
        )
        t_cc = ((jnp.arange(1, nt + 1) - 0.5) * dt).reshape(nt, 1, 1, 1)
        y0 = State3D(
            rho=(1 - t_cc) * rho0[None, ...] + t_cc * rho1[None, ...],  # (nt,nx,ny,nz)
            mx=jnp.zeros((nt, nx, ny, nz)),
            my=jnp.zeros((nt, nx, ny, nz)),
            mz=jnp.zeros((nt, nx, ny, nz)),
        )

    b = State3D.zeros_like(y0)

    gamma = ladmm_cfg.gamma
    zeros_x = jnp.zeros((nt, ny, nz))
    zeros_y = jnp.zeros((nt, nx, nz))
    zeros_z = jnp.zeros((nt, nx, ny))

    def A_fn(x: State3D) -> State3D:
        return State3D(
            rho=ops.interp_t_at_phi(x.rho, rho0, rho1),
            mx=ops.interp_x_at_phi(x.mx, zeros_x, zeros_x),
            my=ops.interp_y_at_phi(x.my, zeros_y, zeros_y),
            mz=ops.interp_z_at_phi(x.mz, zeros_z, zeros_z),
        )

    def At_fn(v: State3D) -> State3D:
        return State3D(
            rho=ops.interp_t_at_rho(v.rho),
            mx=ops.interp_x_at_m(v.mx),
            my=ops.interp_y_at_m(v.my),
            mz=ops.interp_z_at_m(v.mz),
        )

    def B_fn(y: State3D) -> State3D:
        return y * -1.0

    def prox_f1(v: State3D, step: float) -> State3D:
        return projection(v, problem, vareps, projection_state)

    def solve_y(delta: State3D, z_hat: State3D, gamma: float) -> State3D:
        # gamma passed in, not the outer closed-over gamma -- see ladmm.py's
        # solve_y docstring. KE-prox step size is 1/gamma.
        return prox_ke_cc_3d(z_hat - delta * (1.0 / gamma), 1.0 / gamma)

    def _weighted_l2(v: State3D) -> float:
        # rho, mx, my, mz are stored as genuine densities/flux (see grid_3d.py:
        # rho0.sum()*dx*dy*dz == 1), so the weighted-L2 norm is a direct
        # Riemann sum dt*dx*dy*dz -- same reasoning as pipeline_2d.py's norm_fn.
        return float(jnp.sqrt(dt * cell_vol * (
            jnp.sum(v.rho**2) + jnp.sum(v.mx**2) + jnp.sum(v.my**2) + jnp.sum(v.mz**2)
        )))

    # dual_norm_fn is the same formula as norm_fn, by pipeline_2d.py's
    # degree-0-homogeneity argument: KE's gradient scale is invariant to the
    # density/mass-per-cell convention. Kept as two named slots (rather than
    # one) because ladmm.py treats them as separate concepts.
    norm_fn = _weighted_l2
    dual_norm_fn = _weighted_l2

    opts = LadmmOpts(
        gamma=gamma,
        tau=ladmm_cfg.tau,
        max_iter=ladmm_cfg.max_iter,
        eps_abs=ladmm_cfg.eps_abs,
        eps_rel=ladmm_cfg.eps_rel,
        alpha=ladmm_cfg.alpha,
        norm_fn=norm_fn,
        dual_norm_fn=dual_norm_fn,
        x_ref=x_ref,
        delta_ref=delta_ref,
        delta_mask=delta_mask,
    )

    x, y, _, info = ladmm_solve(prox_f1, solve_y, A_fn, At_fn, B_fn, b, x0, y0, opts)

    return SolveResult3D(
        rho_stag=x.rho, mx_stag=x.mx, my_stag=x.my, mz_stag=x.mz,
        rho_cc=y.rho, mx_cc=y.mx, my_cc=y.my, mz_cc=y.mz,
        info=info,
    )
