"""Unbalanced (Wasserstein-Fisher-Rao + diffusion) dynamic transport by LADMM.

    min  int int  |m|^2/(2 rho) + (delta^2/2) zeta^2/rho
    s.t. d_t rho + d_x m - eps d_xx rho = zeta,   rho(0)=rho0, rho(1)=rho1, m=0 on the walls.

With s = delta*zeta this is  min I_FP_source(x) + KE(y)  s.t. A x - y = 0, with
x = (rho_stag, m_stag, s), y = (rho_cc, m_cc, s_cc), A = interpolation on
(rho, m) and identity on s -- so ||A|| <= 1 as before and the same LADMM
loop, step sizes and stopping rule apply.  delta is the WFR length scale:
mass moves if it is cheaper than destroying and recreating it, which for two
far-apart lumps happens below a separation of about pi*delta.  rho0 and rho1
may have different masses.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ladmm import LadmmInfo, LadmmOpts, ladmm_solve
from .pipeline import LadmmConfig
from .projection_unbalanced import precomp_banded_proj_unbalanced, proj_fp_source_banded
from .prox_unbalanced import prox_ke_reaction_cc
from .state_unbalanced import StateU


@dataclass
class SolveResultU:
    rho_stag: np.ndarray  # (ntm, nx)
    mx_stag: np.ndarray   # (nt, nxm)
    s_stag: np.ndarray    # (nt, nx)   scaled source, FP-feasible copy
    rho_cc: np.ndarray    # (nt, nx)
    mx_cc: np.ndarray     # (nt, nx)
    s_cc: np.ndarray      # (nt, nx)
    delta: float
    ke: float             # sum m^2/(2 rho) dt dx on the cc grid
    reaction: float       # sum s^2/(2 rho) dt dx  (= delta^2/2 int zeta^2/rho)
    mass_t: np.ndarray    # (nt+1,) mass at t = 0, dt, ..., 1
    info: LadmmInfo


def discretize_then_optimize_unbalanced(problem, vareps: float, delta: float,
                                        ladmm_cfg: LadmmConfig) -> SolveResultU:
    ops = problem.ops
    rho0, rho1 = problem.rho0, problem.rho1
    nt, nx = problem.nt, problem.nx
    ntm, nxm = nt - 1, nx - 1
    dt, dx = problem.dt, problem.dx

    bp = precomp_banded_proj_unbalanced(problem, vareps, delta)

    t_stag = (np.arange(1, ntm + 1) * dt)[:, None]
    x0 = StateU(rho=(1 - t_stag) * rho0 + t_stag * rho1, mx=np.zeros((nt, nxm)), s=np.zeros((nt, nx)))
    t_cc = ((np.arange(1, nt + 1) - 0.5) * dt)[:, None]
    y0 = StateU(rho=(1 - t_cc) * rho0 + t_cc * rho1, mx=np.zeros((nt, nx)), s=np.zeros((nt, nx)))
    b = StateU.zeros_like(y0)
    zeros_nt = np.zeros(nt)

    def A_fn(x):
        return StateU(ops.interp_t_at_phi(x.rho, rho0, rho1), ops.interp_x_at_phi(x.mx, zeros_nt, zeros_nt), x.s)

    def At_fn(v):
        return StateU(ops.interp_t_at_rho(v.rho), ops.interp_x_at_m(v.mx), v.s)

    def B_fn(y):
        return y * -1.0

    def prox_f1(v, step):
        return proj_fp_source_banded(v, problem, vareps, bp)

    def solve_y(delta_mult, z_hat, gamma):
        return prox_ke_reaction_cc(z_hat - delta_mult * (1.0 / gamma), 1.0 / gamma)

    def norm_fn(v):
        return float(np.sqrt(dt * dx * (np.sum(v.rho**2) + np.sum(v.mx**2) + np.sum(v.s**2))))

    opts = LadmmOpts(gamma=ladmm_cfg.gamma, tau=ladmm_cfg.tau, max_iter=ladmm_cfg.max_iter,
                     eps_abs=ladmm_cfg.eps_abs, eps_rel=ladmm_cfg.eps_rel, alpha=ladmm_cfg.alpha,
                     norm_fn=norm_fn, dual_norm_fn=norm_fn)
    x, y, _, info = ladmm_solve(prox_f1, solve_y, A_fn, At_fn, B_fn, b, x0, y0, opts)

    pos = y.rho > 1e-12
    ke = float(np.sum(np.where(pos, y.mx**2 / (2 * np.where(pos, y.rho, 1.0)), 0.0)) * dt * dx)
    reaction = float(np.sum(np.where(pos, y.s**2 / (2 * np.where(pos, y.rho, 1.0)), 0.0)) * dt * dx)
    mass_t = np.concatenate([[rho0.sum() * dx], x.rho.sum(axis=1) * dx, [rho1.sum() * dx]])
    return SolveResultU(rho_stag=x.rho, mx_stag=x.mx, s_stag=x.s, rho_cc=y.rho, mx_cc=y.mx, s_cc=y.s,
                        delta=delta, ke=ke, reaction=reaction, mass_t=mass_t, info=info)
