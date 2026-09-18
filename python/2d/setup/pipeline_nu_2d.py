"""Wire grid_nu_2d + operators_nu_2d + FP projection_nu_2d + KE prox_nu_2d
into a linearized-ADMM solve on a non-uniform time grid.

Cross of pipeline_2d.py and pipeline_nu.py -- see pipeline_2d.py for the
A/B/prox_f1 wiring of

    min  I_FP(x) + KE(y)   s.t.   A*x - y = 0

which is unchanged here (ladmm.py itself is entirely grid-agnostic). What
actually depends on the grid being non-uniform is pipeline_nu.py's list,
carried over with the extra momentum component:

  - solve_y calls prox_ke_cc_nu_2d -- which is prox_ke_cc_2d verbatim; see
    prox_nu_2d.py for why the dt_vec weight cancels out of the pointwise
    argmin exactly rather than being neglected.
  - At_fn's rho piece uses interp_t_at_rho_weighted, the genuine N,M-weighted
    adjoint of the consensus map's time-average, NOT ops.interp_t_at_rho
    (only the *Euclidean* adjoint, and equal to the weighted one only where
    dt_vec is locally constant). At_fn's mx/my pieces are untouched: space is
    uniform in both axes, so their Euclidean and weighted adjoints already
    coincide.
  - prox_f1's projection_nu_2d uses R's own N-weighted adjoint internally.
  - norm_fn/dual_norm_fn use a genuine non-uniform Riemann sum,
    sum_n dt_vec[n] * dx*dy * (...), instead of pipeline_2d.py's constant
    dt*dx*dy scaling -- and these are now literally the metrics N and M that
    At_fn's adjoint is weighted by, not just a convergence diagnostic.

IMPORTANT, inherited from pipeline_nu.py and not re-litigated here: those
items are NOT independently swappable. Each was individually correct in
isolation, but enabling only one or two of them made the 1D LADMM iteration
either diverge or converge to a visibly wrong answer
(docs/nonuniform_grid_norms.tex Sec 8). They are all on together here.

x0/y0 are built from the actual t_edges/t_centers rather than linspace, so
the initial interpolant between rho0 and rho1 is keyed to the real grid.

Only the plain (r, x, y, delta) LADMM cycle is wired up, following
pipeline_2d.py rather than pipeline_nu.py -- the 1D module's reordered /
Goldstein-Li-Yuan / Malitsky-Pock-linesearch / ALiA solvers have no 2D
counterpart yet (ladmm_reordered.py, ladmm_linesearch.py, alia.py are not
part of the 2D tree), so there is nothing here to dispatch between.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp

from .ladmm import LadmmInfo, LadmmOpts, ladmm_solve
from .operators_nu_2d import interp_t_at_rho_weighted
from .prox_nu_2d import prox_ke_cc_nu_2d
from .state_2d import State2D


@dataclass
class LadmmConfig:
    gamma: float     # penalty parameter
    tau: float       # proximal penalty for the x-update
    max_iter: int
    eps_abs: float   # absolute floor for the D/P stopping criterion (see ladmm.py)
    eps_rel: float   # relative tolerance for the D/P stopping criterion
    alpha: float = 1.0  # over-relaxation


@dataclass
class SolveResultNU2D:
    rho_stag: jnp.ndarray  # (ntm, nx, ny)  FP-feasible density, staggered grid
    mx_stag: jnp.ndarray   # (nt, nxm, ny)  FP-feasible x-momentum, staggered grid
    my_stag: jnp.ndarray   # (nt, nx, nym)  FP-feasible y-momentum, staggered grid
    rho_cc: jnp.ndarray    # (nt, nx, ny)   KE-optimal density, cell-centre grid
    mx_cc: jnp.ndarray     # (nt, nx, ny)   KE-optimal x-momentum, cell-centre grid
    my_cc: jnp.ndarray     # (nt, nx, ny)   KE-optimal y-momentum, cell-centre grid
    info: LadmmInfo


def discretize_then_optimize_nu_2d(
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
) -> SolveResultNU2D:

    ops = problem.ops
    rho0, rho1 = problem.rho0, problem.rho1
    nt, nx, ny = problem.nt, problem.nx, problem.ny
    ntm, nxm, nym = nt - 1, nx - 1, ny - 1
    dt_vec, dt_dual = problem.dt_vec, problem.dt_dual
    cell_area = problem.cell_area

    if x0 is None or y0 is None:
        t_stag = problem.t_edges[1:-1][:, None, None]  # (ntm,1,1) -- interior rho nodes
        x0 = State2D(
            rho=(1 - t_stag) * rho0[None, :, :] + t_stag * rho1[None, :, :],  # (ntm, nx, ny)
            mx=jnp.zeros((nt, nxm, ny)),
            my=jnp.zeros((nt, nx, nym)),
        )
        t_cc = problem.t_centers[:, None, None]  # (nt,1,1)
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
            rho=interp_t_at_rho_weighted(v.rho, dt_vec),
            mx=ops.interp_x_at_m(v.mx),
            my=ops.interp_y_at_m(v.my),
        )

    def B_fn(y: State2D) -> State2D:
        return y * -1.0

    def prox_f1(v: State2D, step: float) -> State2D:
        return projection(v, problem, vareps, projection_state)

    def solve_y(delta: State2D, z_hat: State2D, gamma: float) -> State2D:
        # gamma passed in, not the outer closed-over gamma -- see
        # pipeline_2d.py's solve_y.
        return prox_ke_cc_nu_2d(z_hat - delta * (1.0 / gamma), 1.0 / gamma)

    def _weighted_sq_sum(v: State2D) -> float:
        # Non-uniform Riemann sum, pipeline_nu.py's with dx -> dx*dy: rho/mx/my
        # are stored as genuine densities, so weighting row n by its local time
        # quadrature weight and the cell area gives the L^2 approximation
        # directly, and reduces to pipeline_2d.py's dt*dx*dy*sum(...) exactly
        # when dt_vec is constant. mx/my always live on the nt cell centres
        # (dt_vec) whether v is the staggered (x) or cell-centre (y) state;
        # rho lives there too for y, but on the ntm *interior interfaces* for
        # x, whose natural weight is the centre-to-centre dual width.
        w_rho = dt_vec if v.rho.shape[0] == nt else dt_dual
        return cell_area * (
            jnp.sum(w_rho[:, None, None] * v.rho**2)
            + jnp.sum(dt_vec[:, None, None] * v.mx**2)
            + jnp.sum(dt_vec[:, None, None] * v.my**2)
        )

    def norm_fn(v: State2D) -> float:
        return float(jnp.sqrt(_weighted_sq_sum(v)))

    def dual_norm_fn(v: State2D) -> float:
        # delta's numeric value is grid-convention-independent (homogeneous
        # of degree 0), so the same weighted formula carries over -- see
        # pipeline_2d.py's dual_norm_fn.
        return float(jnp.sqrt(_weighted_sq_sum(v)))

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

    return SolveResultNU2D(
        rho_stag=x.rho, mx_stag=x.mx, my_stag=x.my,
        rho_cc=y.rho, mx_cc=y.mx, my_cc=y.my,
        info=info,
    )
