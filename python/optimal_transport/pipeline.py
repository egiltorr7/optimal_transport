"""Wire grid + operators + FP projection + KE prox into a linearized-ADMM solve.

Port of matlab/shared/1d/pipelines/discretize_then_optimize.m.

Solves:
    min   I_FP(x) + KE(y)   s.t.   A*x - y = 0

  x = (rho, mx) on the staggered grid   -- FP-constraint variable
  y = (rho, mx) on the cell-centre grid -- KE variable
  A = affine interpolation staggered -> cell-centres (BCs baked in)
  B = -I  (on cell-centres),  b = 0  (BCs absorbed into affine A)

`projection` is passed in rather than hardcoded, so either
proj_fokker_planck_banded or proj_fokker_planck_expsemi can be used with the
same pipeline -- both share the signature (x_in, problem, vareps, precomputed).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .ladmm import LadmmInfo, LadmmOpts, ladmm_solve
from .prox import prox_ke_cc
from .state import State


@dataclass
class LadmmConfig:
    gamma: float     # penalty parameter
    tau: float       # proximal penalty for the x-update (> gamma * ||A_linear||^2 <= gamma)
    max_iter: int
    eps_abs: float   # absolute floor for the D/P stopping criterion (see ladmm.py)
    eps_rel: float   # relative tolerance for the D/P stopping criterion
    alpha: float = 1.0  # over-relaxation


@dataclass
class SolveResult:
    rho_stag: np.ndarray  # (ntm, nx)  FP-feasible density, staggered grid
    mx_stag: np.ndarray   # (nt, nxm)  FP-feasible momentum, staggered grid
    rho_cc: np.ndarray    # (nt, nx)   KE-optimal density, cell-centre grid
    mx_cc: np.ndarray     # (nt, nx)   KE-optimal momentum, cell-centre grid
    info: LadmmInfo


def discretize_then_optimize(
    problem,
    projection: Callable[[State, object, float, object], State],
    projection_state: object,
    vareps: float,
    ladmm_cfg: LadmmConfig,
    x0: State | None = None,
    y0: State | None = None,
    x_ref: State | None = None,
    delta_ref: State | None = None,
    delta_mask: State | None = None,
    x_refs: dict | None = None,
) -> SolveResult:

    ops = problem.ops
    rho0, rho1 = problem.rho0, problem.rho1
    nt, nx = problem.nt, problem.nx
    ntm, nxm = nt - 1, nx - 1
    dt, dx = problem.dt, problem.dx

    if x0 is None or y0 is None:
        t_stag = np.linspace(0.0, 1.0, ntm)[:, None]
        x0 = State(
            rho=(1 - t_stag) * rho0 + t_stag * rho1,  # (ntm, nx)
            mx=np.zeros((nt, nxm)),
        )
        t_cc = ((np.arange(1, nt + 1) - 0.5) * dt)[:, None]
        y0 = State(
            rho=(1 - t_cc) * rho0 + t_cc * rho1,  # (nt, nx)
            mx=np.zeros((nt, nx)),
        )

    b = State.zeros_like(y0)

    gamma = ladmm_cfg.gamma
    zeros_nt = np.zeros(nt)

    def A_fn(x: State) -> State:
        return State(
            rho=ops.interp_t_at_phi(x.rho, rho0, rho1),
            mx=ops.interp_x_at_phi(x.mx, zeros_nt, zeros_nt),
        )

    def At_fn(v: State) -> State:
        return State(
            rho=ops.interp_t_at_rho(v.rho),
            mx=ops.interp_x_at_m(v.mx),
        )

    def B_fn(y: State) -> State:
        return y * -1.0

    def prox_f1(v: State, step: float) -> State:
        return projection(v, problem, vareps, projection_state)

    def solve_y(delta: State, z_hat: State, gamma: float) -> State:
        # gamma is passed in (not the outer closed-over gamma above, used
        # only for x0/y0/opts' value) since ladmm_reordered.py's sibling solver
        # may adapt it mid-run -- see ladmm.py's solve_y docstring note.
        # KE-prox step size is 1/gamma. Kept inline (not a named `sigma`
        # variable) so this stays unambiguous next to problem.sigma (the
        # Gaussian std dev) when debugging.
        return prox_ke_cc(z_hat - delta * (1.0 / gamma), 1.0 / gamma)

    def norm_fn(v: State) -> float:
        # rho, mx are stored as genuine densities (rho[i] ~= rho_density(x_i),
        # confirmed by rho0.sum()*dx==1) -- so the weighted-L2 norm of the
        # underlying density/flux fields is a direct Riemann-sum: dt*dx,
        # no dx-correction needed (unlike the old mass-per-cell convention,
        # where rho[i]=rho_density(x_i)*dx and dt/dx was needed to convert
        # back to density scale).
        return float(np.sqrt(dt * dx * (np.sum(v.rho**2) + np.sum(v.mx**2))))

    def dual_norm_fn(v: State) -> float:
        # delta (the KKT multiplier) is ~ d f2(y)/dy -- a *gradient* w.r.t.
        # the state, not a state itself. f2(y) = sum m^2/(2*rho) is
        # homogeneous of degree 0 in its gradient under uniform (rho,m) ->
        # (c*rho, c*m) rescaling (d/drho[m^2/2rho] = -m^2/2rho^2, and
        # (c*m)^2/(c*rho)^2 = m^2/rho^2 -- the c's cancel exactly), so delta's
        # numeric value is actually the SAME regardless of whether y is
        # stored as mass-per-cell or density -- it was already a raw,
        # unscaled density-type quantity even under the old mass convention.
        # That means this formula (dt*dx-weighted) carries over unchanged;
        # it's now identical to norm_fn (no more asymmetry to correct for,
        # since primal quantities are directly density-scale too).
        return float(np.sqrt(dt * dx * (np.sum(v.rho**2) + np.sum(v.mx**2))))

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
        x_refs=x_refs,
        # rho-only companion to norm_fn, so x_refs are scored in the density
        # alone as well as in the full state.
        rho_norm_fn=(lambda v: float(np.sqrt(dt * dx * np.sum(v.rho**2)))),
    )

    x, y, _, info = ladmm_solve(prox_f1, solve_y, A_fn, At_fn, B_fn, b, x0, y0, opts)

    return SolveResult(rho_stag=x.rho, mx_stag=x.mx, rho_cc=y.rho, mx_cc=y.mx, info=info)
