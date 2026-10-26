"""Projection onto the Fokker-Planck constraint WITH a source term,

    d_t rho + d_x m - eps d_xx rho = s / delta,        (s = delta * zeta)

by the same DCT-in-x + batched tridiagonal-in-t solve as projection.py (CN /
banded scheme).  The constraint operator is M = [A_FP, -(1/delta) I], so the
normal operator is M M^T = A_FP A_FP^T + (1/delta^2) I: every per-mode
tridiagonal system gets +1/delta^2 on its diagonal, and the DC spatial mode,
singular in the balanced case (mass conservation = gauge freedom), becomes
regular, because total mass is now free.  The correction for s is
s_out = s_in + phi/delta.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import dct, idct

from .projection import thomas_batch_precomp, thomas_batch_solve
from .state_unbalanced import StateU


@dataclass
class BandedProjU:
    D_mod: np.ndarray   # (nt, nx-1) Thomas-modified diagonal, modes k=1..nx-1
    e: np.ndarray        # (nx-1,)    constant off-diagonal per mode
    dc_den: np.ndarray   # (nt,)      lambda_t + 1/delta^2, for the k=0 mode
    delta: float


def precomp_banded_proj_unbalanced(problem, vareps: float, delta: float) -> BandedProjU:
    nt, dt = problem.nt, problem.dt
    lx = problem.lambda_x[1:]
    inv_d2 = 1.0 / delta**2

    D = np.broadcast_to(2 / dt**2 + lx + lx**2 * vareps**2 / 2 + inv_d2, (nt, lx.shape[0])).copy()
    D[0, :] = 1 / dt**2 + lx * (1 + vareps / dt) + lx**2 * vareps**2 / 4 + inv_d2
    D[-1, :] = 1 / dt**2 + lx * (1 - vareps / dt) + lx**2 * vareps**2 / 4 + inv_d2
    e = -1 / dt**2 + lx**2 * vareps**2 / 4

    D_mod = thomas_batch_precomp(D, e)
    dc_den = problem.lambda_t[:, 0] + inv_d2
    return BandedProjU(D_mod=D_mod, e=e, dc_den=dc_den, delta=delta)


def proj_fp_source_banded(x_in: StateU, problem, vareps: float, bp: BandedProjU) -> StateU:
    ops = problem.ops
    rho0, rho1 = problem.rho0, problem.rho1
    nt, nx = problem.nt, problem.nx
    delta = bp.delta

    mu, psi, s = x_in.rho, x_in.mx, x_in.s
    zeros_x = np.zeros(nt)

    laplacian_mu = ops.deriv_x_at_phi(
        ops.deriv_x_at_m(ops.interp_t_at_phi(mu, rho0, rho1)), zeros_x, zeros_x
    )
    f = (
        ops.deriv_t_at_phi(mu, rho0, rho1)
        + ops.deriv_x_at_phi(psi, zeros_x, zeros_x)
        - vareps * laplacian_mu
        - s / delta
    )

    if np.linalg.norm(f) * np.sqrt(problem.dt * problem.dx) < 1e-14:
        return StateU(mu.copy(), psi.copy(), s.copy())

    f_hat = dct(f, type=2, norm="ortho", axis=1)
    phi_hat = np.zeros((nt, nx))

    # k = 0 spatial mode: Neumann-in-t Laplacian + 1/delta^2, diagonal in the t-DCT basis
    f1_t = dct(f_hat[:, 0], type=2, norm="ortho")
    phi_hat[:, 0] = idct(f1_t / bp.dc_den, type=2, norm="ortho")

    phi_hat[:, 1:] = thomas_batch_solve(bp.D_mod, bp.e, f_hat[:, 1:])
    phi = idct(phi_hat, type=2, norm="ortho", axis=1)

    dphi_dx = ops.deriv_x_at_m(phi)
    nablax_phi = ops.interp_t_at_rho(ops.deriv_x_at_phi(dphi_dx, zeros_x, zeros_x))

    rho_out = mu + ops.deriv_t_at_rho(phi) + vareps * nablax_phi
    mx_out = psi + dphi_dx
    s_out = s + phi / delta
    return StateU(rho_out, mx_out, s_out)
