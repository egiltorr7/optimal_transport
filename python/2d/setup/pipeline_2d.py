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
from .ladmm_jit import ladmm_solve_jit
from .prox_2d import prox_ke_cc_2d
from .state_2d import State2D


def _cast_state(v: State2D, dtype) -> State2D:
    return v if v.rho.dtype == dtype else State2D(v.rho.astype(dtype), v.mx.astype(dtype), v.my.astype(dtype))


@dataclass
class LadmmConfig:
    gamma: float     # penalty parameter
    tau: float       # proximal penalty for the x-update (> gamma * ||A_linear||^2 <= gamma)
    max_iter: int
    eps_abs: float   # absolute floor for the D/P stopping criterion (see ladmm.py)
    eps_rel: float   # relative tolerance for the D/P stopping criterion
    alpha: float = 1.0  # over-relaxation
    jit: bool = False   # run ladmm_jit.ladmm_solve_jit instead of the eager ladmm.ladmm_solve
    check_every: int = 1  # jit only: iterations per compiled call (see ladmm_jit.py)
    diag_level: str = "full"  # jit only: "full" or "stopping" (see ladmm_jit._make_step)
    master_dtype: object = None  # jit only: keep x and delta in this dtype, e.g. jnp.float64, while
    #                              the rest runs in the problem's (see ladmm_jit.py / precision.py)
    round_x_dtype: object = None  # jit only: round x to this dtype before A_fn (ladmm_jit.py)
    prox_dtype: object = None    # run the KE prox arithmetic in this dtype (default: z_hat's)
    y_dtype: object = None       # store the prox output (y, hence By and D) in this dtype (default: z_hat's)
    consistent_dual: bool = False  # jit only: dual = -grad KE at the stored y, in master_dtype
    #                              (ladmm_jit.py docstring: required when y is float32)


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
    x_refs: dict[str, State2D] | None = None,
    track_rho: bool = False,
) -> SolveResult2D:

    ops = problem.ops
    rho0, rho1 = problem.rho0, problem.rho1
    nt, nx, ny = problem.nt, problem.nx, problem.ny
    ntm, nxm, nym = nt - 1, nx - 1, ny - 1
    dt, dx, dy = problem.dt, problem.dx, problem.dy
    # Every array built here takes the problem's dtype, so a float32 problem
    # (precision.py's cast_problem) gives a float32 iteration even with x64
    # enabled -- a single float64 zero plane would promote the whole solve.
    dtype = rho0.dtype

    if x0 is None or y0 is None:
        # rho's INTERIOR time edges, t = k*dt for k=1..ntm -- not
        # linspace(0, 1, ntm), which put the first node at t=0 and the last
        # at t=1. Those are the two Dirichlet BOUNDARIES (rho0 and rho1), not
        # interior nodes, so the old seed was the right interpolant sampled at
        # the wrong places -- a worse starting point, and it made this pipeline
        # disagree with the *_nu sibling (which reads t_edges[1:-1] correctly)
        # even on a uniform grid, where the two are the same problem.
        t_stag = (jnp.arange(1, ntm + 1) * dt)[:, None, None].astype(dtype)
        x0 = State2D(
            rho=(1 - t_stag) * rho0[None, :, :] + t_stag * rho1[None, :, :],  # (ntm, nx, ny)
            mx=jnp.zeros((nt, nxm, ny), dtype),
            my=jnp.zeros((nt, nx, nym), dtype),
        )
        t_cc = ((jnp.arange(1, nt + 1) - 0.5) * dt)[:, None, None].astype(dtype)
        y0 = State2D(
            rho=(1 - t_cc) * rho0[None, :, :] + t_cc * rho1[None, :, :],  # (nt, nx, ny)
            mx=jnp.zeros((nt, nx, ny), dtype),
            my=jnp.zeros((nt, nx, ny), dtype),
        )

    if ladmm_cfg.y_dtype is not None:
        y0 = _cast_state(y0, ladmm_cfg.y_dtype)   # y carries the prox output dtype
    b = State2D.zeros_like(y0)

    gamma = ladmm_cfg.gamma
    zeros_x = jnp.zeros((nt, ny), dtype)
    zeros_y = jnp.zeros((nt, nx), dtype)

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

    def prox_f1(v: State2D, step: float, ps=projection_state) -> State2D:
        # `ps` defaults to the closed-over precompute for the eager solver;
        # the jit solver passes it explicitly (ladmm_jit.py).
        return projection(v, problem, vareps, ps)

    def solve_y(delta: State2D, z_hat: State2D, gamma: float) -> State2D:
        # gamma passed in, not the outer closed-over gamma (used only for
        # x0/y0/opts' value) -- ladmm_reordered.py's sibling solver adapts it,
        # see ladmm.py's solve_y docstring note.
        # KE-prox step size is 1/gamma -- see pipeline.py's solve_y for why
        # this is kept inline rather than as a named `sigma` variable.
        # Arithmetic in prox_dtype, storage in y_dtype, both defaulting to
        # z_hat's dtype (the subtraction promotes when delta is a float64
        # master copy). The mixed modes run the arithmetic in float64 -- the
        # float32 Cardano root is ~1e-3 off in near-empty cells -- and store
        # float32; the casts fuse into the elementwise kernel.
        arg = z_hat - delta * (1.0 / gamma)
        y = prox_ke_cc_2d(_cast_state(arg, ladmm_cfg.prox_dtype or z_hat.rho.dtype), 1.0 / gamma)
        return _cast_state(y, ladmm_cfg.y_dtype or z_hat.rho.dtype)

    def _norm_jnp(v: State2D):
        # rho, mx, my are stored as genuine densities/flux (see grid_2d.py:
        # rho0.sum()*dx*dy == 1), so the weighted-L2 norm is a direct
        # Riemann sum dt*dx*dy -- same reasoning as pipeline.py's norm_fn.
        return jnp.sqrt(dt * dx * dy * (jnp.sum(v.rho**2) + jnp.sum(v.mx**2) + jnp.sum(v.my**2)))

    def _rho_norm_jnp(v: State2D):
        # rho alone, for x_refs error tracking (track_rho=True): the density
        # error is usually the quantity of interest and norm_fn mixes in mx/my.
        return jnp.sqrt(dt * dx * dy * jnp.sum(v.rho**2))

    # The jit solver needs jnp scalars (float() cannot trace); the eager one
    # wants Python floats. Same formula either way.
    def norm_fn(v: State2D) -> float:
        return float(_norm_jnp(v))

    def dual_norm_fn(v: State2D) -> float:
        # Same degree-0-homogeneity argument as pipeline.py's dual_norm_fn:
        # KE's gradient scale is invariant to the density/mass-per-cell
        # convention, so this is identical to norm_fn.
        return float(_norm_jnp(v))

    opts = LadmmOpts(
        gamma=gamma,
        tau=ladmm_cfg.tau,
        max_iter=ladmm_cfg.max_iter,
        eps_abs=ladmm_cfg.eps_abs,
        eps_rel=ladmm_cfg.eps_rel,
        alpha=ladmm_cfg.alpha,
        norm_fn=_norm_jnp if ladmm_cfg.jit else norm_fn,
        dual_norm_fn=_norm_jnp if ladmm_cfg.jit else dual_norm_fn,
        x_ref=x_ref,
        delta_ref=delta_ref,
        delta_mask=delta_mask,
        x_refs=x_refs,
        rho_norm_fn=(_rho_norm_jnp if ladmm_cfg.jit else (lambda v: float(_rho_norm_jnp(v))))
        if track_rho else None,
    )

    def _dual_from_y(y: State2D, delta_std: State2D) -> State2D:
        # -grad KE(y) = (|m|^2/(2 rho^2), -mx/rho, -my/rho) where rho > 0, in
        # the dual's (master) dtype from the stored y; the prox's clamped cells
        # (rho = 0) keep the standard update, which is a valid subgradient there.
        dt_ = delta_std.rho.dtype
        rho, mx, my = y.rho.astype(dt_), y.mx.astype(dt_), y.my.astype(dt_)
        pos = rho > 0
        rs = jnp.where(pos, rho, 1.0)
        return State2D(jnp.where(pos, (mx**2 + my**2) / (2 * rs**2), delta_std.rho),
                       jnp.where(pos, -mx / rs, delta_std.mx),
                       jnp.where(pos, -my / rs, delta_std.my))

    if ladmm_cfg.jit:
        x, y, _, info = ladmm_solve_jit(prox_f1, solve_y, A_fn, At_fn, B_fn, b, x0, y0, opts,
                                        aux=projection_state, check_every=ladmm_cfg.check_every,
                                        diag_level=ladmm_cfg.diag_level, master_dtype=ladmm_cfg.master_dtype,
                                        round_x_dtype=ladmm_cfg.round_x_dtype,
                                        dual_fn=(_dual_from_y if ladmm_cfg.consistent_dual else None))
    else:
        x, y, _, info = ladmm_solve(prox_f1, solve_y, A_fn, At_fn, B_fn, b, x0, y0, opts)

    return SolveResult2D(
        rho_stag=x.rho, mx_stag=x.mx, my_stag=x.my,
        rho_cc=y.rho, mx_cc=y.mx, my_cc=y.my,
        info=info,
    )
