"""Wire 2D grid + operators + FP projection + KE prox into a linearized-ADMM solve.

2D analogue of pipeline.py -- see that module for the general
linearized-ADMM formulation this wires up:

    min   I_FP(x) + KE(y)   s.t.   A*x - y = 0

  x = (rho, mx, my) on the staggered grid   -- FP-constraint variable
  y = (rho, mx, my) on the cell-centre grid -- KE variable
  A = affine interpolation staggered -> cell-centres (BCs baked in)
  B = -I  (on cell-centres),  b = 0  (BCs absorbed into affine A)

`projection` is passed in rather than hardcoded, same reason as
pipeline.py: so proj_fokker_planck_banded_2d (or a future ETD/PCR/Spike2
2D projection) can be swapped in against the same pipeline.

Reuses ladmm.py unchanged from the 1D code -- it operates on the (rho,
mx, ...) state only via State2D's own arithmetic/.norm()/.zeros_like(),
now duck-typed there (see the comment on that generic fix) rather than
hardcoded to the 1D State class.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp

from .ladmm import LadmmInfo, LadmmOpts, ladmm_solve
from .prox_2d import prox_ke_cc_2d
from .state_2d import State2D


@dataclass
class LadmmConfig:
    gamma: float     # penalty parameter
    tau: float       # proximal penalty for the x-update (> gamma * ||A_linear||^2 <= gamma)
    max_iter: int
    eps_abs: float   # absolute floor for the D/P stopping criterion (see ladmm.py)
    eps_rel: float   # relative tolerance for the D/P stopping criterion
    alpha: float = 1.0  # over-relaxation


@dataclass
class SolveResult2D:
    rho_stag: jnp.ndarray  # (ntm, nx, ny)  FP-feasible density, staggered grid
    mx_stag: jnp.ndarray   # (nt, nxm, ny)  FP-feasible x-momentum, staggered grid
    my_stag: jnp.ndarray   # (nt, nx, nym)  FP-feasible y-momentum, staggered grid
    rho_cc: jnp.ndarray    # (nt, nx, ny)   KE-optimal density, cell-centre grid
    mx_cc: jnp.ndarray     # (nt, nx, ny)   KE-optimal x-momentum, cell-centre grid
    my_cc: jnp.ndarray     # (nt, nx, ny)   KE-optimal y-momentum, cell-centre grid
    info: LadmmInfo


def discretize_then_optimize_2d(
    problem,
    projection: Callable[[State2D, object, float, object], State2D],
    projection_state: object,
    vareps: float,
    ladmm_cfg: LadmmConfig,
    x0: State2D | None = None,
    y0: State2D | None = None,
    x_ref: State2D | None = None,
    delta_ref: State2D | None = None,
    delta_mask: State2D | None = None,
) -> SolveResult2D:

    ops = problem.ops
    rho0, rho1 = problem.rho0, problem.rho1
    nt, nx, ny = problem.nt, problem.nx, problem.ny
    ntm, nxm, nym = nt - 1, nx - 1, ny - 1
    dt, dx, dy = problem.dt, problem.dx, problem.dy

    if x0 is None or y0 is None:
        t_stag = jnp.linspace(0.0, 1.0, ntm)[:, None, None]
        x0 = State2D(
            rho=(1 - t_stag) * rho0[None, :, :] + t_stag * rho1[None, :, :],  # (ntm, nx, ny)
            mx=jnp.zeros((nt, nxm, ny)),
            my=jnp.zeros((nt, nx, nym)),
        )
        t_cc = ((jnp.arange(1, nt + 1) - 0.5) * dt)[:, None, None]
        y0 = State2D(
            rho=(1 - t_cc) * rho0[None, :, :] + t_cc * rho1[None, :, :],  # (nt, nx, ny)
            mx=jnp.zeros((nt, nx, ny)),
            my=jnp.zeros((nt, nx, ny)),
        )

    b = State2D.zeros_like(y0)

    gamma = ladmm_cfg.gamma
    zeros_x = jnp.zeros((nt, ny))
    zeros_y = jnp.zeros((nt, nx))

    def A_fn(x: State2D) -> State2D:
        return State2D(
            rho=ops.interp_t_at_phi(x.rho, rho0, rho1),
            mx=ops.interp_x_at_phi(x.mx, zeros_x, zeros_x),
            my=ops.interp_y_at_phi(x.my, zeros_y, zeros_y),
        )

    def At_fn(v: State2D) -> State2D:
        return State2D(
            rho=ops.interp_t_at_rho(v.rho),
            mx=ops.interp_x_at_m(v.mx),
            my=ops.interp_y_at_m(v.my),
        )

    def B_fn(y: State2D) -> State2D:
        return y * -1.0

    def prox_f1(v: State2D, step: float) -> State2D:
        return projection(v, problem, vareps, projection_state)

    def solve_y(delta: State2D, z_hat: State2D, gamma: float) -> State2D:
        # gamma passed in, not the outer closed-over gamma (used only for
        # x0/y0/opts' value) -- ladmm_reordered.py's sibling solver adapts it,
        # see ladmm.py's solve_y docstring note.
        # KE-prox step size is 1/gamma -- see pipeline.py's solve_y for why
        # this is kept inline rather than as a named `sigma` variable.
        return prox_ke_cc_2d(z_hat - delta * (1.0 / gamma), 1.0 / gamma)

    def norm_fn(v: State2D) -> float:
        # rho, mx, my are stored as genuine densities/flux (see grid_2d.py:
        # rho0.sum()*dx*dy == 1), so the weighted-L2 norm is a direct
        # Riemann sum dt*dx*dy -- same reasoning as pipeline.py's norm_fn.
        return float(jnp.sqrt(dt * dx * dy * (jnp.sum(v.rho**2) + jnp.sum(v.mx**2) + jnp.sum(v.my**2))))

    def dual_norm_fn(v: State2D) -> float:
        # Same degree-0-homogeneity argument as pipeline.py's dual_norm_fn:
        # KE's gradient scale is invariant to the density/mass-per-cell
        # convention, so this is identical to norm_fn.
        return float(jnp.sqrt(dt * dx * dy * (jnp.sum(v.rho**2) + jnp.sum(v.mx**2) + jnp.sum(v.my**2))))

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

    return SolveResult2D(
        rho_stag=x.rho, mx_stag=x.mx, my_stag=x.my,
        rho_cc=y.rho, mx_cc=y.mx, my_cc=y.my,
        info=info,
    )
