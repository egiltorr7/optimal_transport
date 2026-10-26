"""Wire 3D grid + operators + FP projection + KE prox into a linearized-ADMM solve.

3D analogue of pipeline_2d.py (GPU version) -- see pipeline.py for the
general linearized-ADMM formulation:

    min   I_FP(x) + KE(y)   s.t.   A*x - y = 0

  x = (rho, mx, my, mz) on the staggered grid   -- FP-constraint variable
  y = (rho, mx, my, mz) on the cell-centre grid -- KE variable
  A = affine interpolation staggered -> cell-centres (BCs baked in)
  B = -I  (on cell-centres),  b = 0

Two solver paths (LadmmConfig.jit): the eager ladmm.py loop, or the compiled
ladmm_jit.py loop (one iteration traced into a single XLA executable,
check_every iterations per lax.scan chunk, one host sync per chunk). The
jit path also carries the precision options established in 2D
(setup/precision.py): master_dtype, round_x_dtype, prox_dtype, y_dtype,
consistent_dual.

Every array built here takes the dtype of problem.rho0, so a float32 problem
(precision.cast_problem) gives a float32 iteration even with x64 enabled.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp

from .ladmm import LadmmInfo, LadmmOpts, ladmm_solve
from .ladmm_jit import ladmm_solve_jit
from .prox_3d import prox_ke_cc_3d
from .state_3d import State3D


def _cast_state(v: State3D, dtype) -> State3D:
    return v if v.rho.dtype == dtype else State3D(*(getattr(v, f).astype(dtype) for f in v.__dataclass_fields__))


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
    master_dtype: object = None  # jit only: keep x and delta in this dtype (e.g. jnp.float64)
    round_x_dtype: object = None  # jit only: round x to this dtype before A_fn
    prox_dtype: object = None    # KE prox arithmetic dtype (default: z_hat's)
    y_dtype: object = None       # storage dtype of the prox output y (default: z_hat's)
    consistent_dual: bool = False  # jit only: dual = -grad KE at the stored y (precision.py)


@dataclass
class SolveResult3D:
    rho_stag: jnp.ndarray  # (ntm, nx, ny, nz)  FP-feasible density, staggered grid
    mx_stag: jnp.ndarray   # (nt, nxm, ny, nz)
    my_stag: jnp.ndarray   # (nt, nx, nym, nz)
    mz_stag: jnp.ndarray   # (nt, nx, ny, nzm)
    rho_cc: jnp.ndarray    # (nt, nx, ny, nz)   KE-optimal density, cell-centre grid
    mx_cc: jnp.ndarray
    my_cc: jnp.ndarray
    mz_cc: jnp.ndarray
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
    x_refs: dict[str, State3D] | None = None,
    track_rho: bool = False,
) -> SolveResult3D:

    ops = problem.ops
    rho0, rho1 = problem.rho0, problem.rho1
    nt, nx, ny, nz = problem.nt, problem.nx, problem.ny, problem.nz
    ntm, nxm, nym, nzm = nt - 1, nx - 1, ny - 1, nz - 1
    dt, dx, dy, dz = problem.dt, problem.dx, problem.dy, problem.dz
    cell_vol = dx * dy * dz
    dtype = rho0.dtype

    if x0 is None or y0 is None:
        # rho's INTERIOR time edges t = k*dt, k = 1..ntm (not linspace(0,1)).
        t_stag = (jnp.arange(1, ntm + 1) * dt).reshape(ntm, 1, 1, 1).astype(dtype)
        x0 = State3D(
            rho=(1 - t_stag) * rho0[None, ...] + t_stag * rho1[None, ...],
            mx=jnp.zeros((nt, nxm, ny, nz), dtype),
            my=jnp.zeros((nt, nx, nym, nz), dtype),
            mz=jnp.zeros((nt, nx, ny, nzm), dtype),
        )
        t_cc = ((jnp.arange(1, nt + 1) - 0.5) * dt).reshape(nt, 1, 1, 1).astype(dtype)
        y0 = State3D(
            rho=(1 - t_cc) * rho0[None, ...] + t_cc * rho1[None, ...],
            mx=jnp.zeros((nt, nx, ny, nz), dtype),
            my=jnp.zeros((nt, nx, ny, nz), dtype),
            mz=jnp.zeros((nt, nx, ny, nz), dtype),
        )
    if ladmm_cfg.y_dtype is not None:
        y0 = _cast_state(y0, ladmm_cfg.y_dtype)   # y carries the prox output dtype

    b = State3D.zeros_like(y0)

    gamma = ladmm_cfg.gamma
    zeros_x = jnp.zeros((nt, ny, nz), dtype)
    zeros_y = jnp.zeros((nt, nx, nz), dtype)
    zeros_z = jnp.zeros((nt, nx, ny), dtype)

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

    def prox_f1(v: State3D, step: float, ps=projection_state) -> State3D:
        # `ps` defaults to the closed-over precompute for the eager solver;
        # the jit solver passes it explicitly (ladmm_jit.py).
        return projection(v, problem, vareps, ps)

    def solve_y(delta: State3D, z_hat: State3D, gamma: float) -> State3D:
        # KE prox with step 1/gamma. Arithmetic in prox_dtype, storage in
        # y_dtype, both defaulting to z_hat's dtype (the subtraction promotes
        # when delta is a float64 master copy) -- see 2d pipeline_2d.solve_y.
        arg = z_hat - delta * (1.0 / gamma)
        y = prox_ke_cc_3d(_cast_state(arg, ladmm_cfg.prox_dtype or z_hat.rho.dtype), 1.0 / gamma)
        return _cast_state(y, ladmm_cfg.y_dtype or z_hat.rho.dtype)

    def _norm_jnp(v: State3D):
        # Genuine densities/flux (rho0.sum()*dx*dy*dz == 1): weighted L2 is
        # the Riemann sum dt*dx*dy*dz.
        return jnp.sqrt(dt * cell_vol * (jnp.sum(v.rho**2) + jnp.sum(v.mx**2)
                                         + jnp.sum(v.my**2) + jnp.sum(v.mz**2)))

    def _rho_norm_jnp(v: State3D):
        return jnp.sqrt(dt * cell_vol * jnp.sum(v.rho**2))

    def norm_fn(v: State3D) -> float:
        return float(_norm_jnp(v))

    def _dual_from_y(y: State3D, delta_std: State3D) -> State3D:
        # -grad KE(y) = (|m|^2/(2 rho^2), -m/rho) where rho > 0, in the dual's
        # dtype from the STORED y; clamped cells keep the standard update.
        dt_ = delta_std.rho.dtype
        rho = y.rho.astype(dt_)
        mx, my, mz = y.mx.astype(dt_), y.my.astype(dt_), y.mz.astype(dt_)
        pos = rho > 0
        rs = jnp.where(pos, rho, 1.0)
        return State3D(jnp.where(pos, (mx**2 + my**2 + mz**2) / (2 * rs**2), delta_std.rho),
                       jnp.where(pos, -mx / rs, delta_std.mx),
                       jnp.where(pos, -my / rs, delta_std.my),
                       jnp.where(pos, -mz / rs, delta_std.mz))

    opts = LadmmOpts(
        gamma=gamma,
        tau=ladmm_cfg.tau,
        max_iter=ladmm_cfg.max_iter,
        eps_abs=ladmm_cfg.eps_abs,
        eps_rel=ladmm_cfg.eps_rel,
        alpha=ladmm_cfg.alpha,
        norm_fn=_norm_jnp if ladmm_cfg.jit else norm_fn,
        dual_norm_fn=_norm_jnp if ladmm_cfg.jit else norm_fn,
        x_ref=x_ref,
        delta_ref=delta_ref,
        delta_mask=delta_mask,
        x_refs=x_refs,
        rho_norm_fn=(_rho_norm_jnp if ladmm_cfg.jit else (lambda v: float(_rho_norm_jnp(v))))
        if track_rho else None,
    )

    if ladmm_cfg.jit:
        x, y, _, info = ladmm_solve_jit(prox_f1, solve_y, A_fn, At_fn, B_fn, b, x0, y0, opts,
                                        aux=projection_state, check_every=ladmm_cfg.check_every,
                                        diag_level=ladmm_cfg.diag_level,
                                        master_dtype=ladmm_cfg.master_dtype,
                                        round_x_dtype=ladmm_cfg.round_x_dtype,
                                        dual_fn=(_dual_from_y if ladmm_cfg.consistent_dual else None))
    else:
        x, y, _, info = ladmm_solve(prox_f1, solve_y, A_fn, At_fn, B_fn, b, x0, y0, opts)

    return SolveResult3D(
        rho_stag=x.rho, mx_stag=x.mx, my_stag=x.my, mz_stag=x.mz,
        rho_cc=y.rho, mx_cc=y.mx, my_cc=y.my, mz_cc=y.mz,
        info=info,
    )
