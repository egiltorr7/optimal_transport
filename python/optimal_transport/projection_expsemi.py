"""Projection onto the Fokker-Planck constraint via an ETD (exponential time
differencing) scheme -- integrates the diffusion term exactly per DCT-x mode
instead of the backward-Euler treatment used in projection.py.

Port of matlab/shared/1d/utils/precomp_expsemi_proj.m and
matlab/shared/1d/projection/proj_fokker_planck_expsemi.m, structured after
the *_gpu variants the same way projection.py mirrors the banded *_gpu
variant: reuses the same thomas_batch_precomp/solve batched over all nx-1
non-DC modes instead of MATLAB CPU's per-mode LU-factorization loop.

ETD scheme: for each DCT-x mode j with alpha_j = vareps*lambda_x(j)*dt,
    c_j   = exp(-alpha_j)                  (exact semigroup factor)
    phi_j = (1 - c_j) / alpha_j            (ETD forcing weight, phi(0)=1)
the FP constraint becomes  (rho_{k+1} - c_j*rho_k)/dt + phi_j*D_x m = 0,
i.e. the diffusion is integrated exactly (matrix exponential) rather than
approximated by a finite difference.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import dct, idct

from .projection import thomas_batch_precomp, thomas_batch_solve
from .state import State


@dataclass
class ExpsemiProj:
    c_vals: np.ndarray    # (nx,)   semigroup coefficients c_j = exp(-alpha_j)
    phi_vals: np.ndarray  # (nx,)   ETD weights phi_j = (1-c_j)/alpha_j
    D_mod: np.ndarray     # (nt, nx-1) Thomas-modified diagonal, modes j=1..nx-1
    e: np.ndarray          # (nx-1,)    constant off-diagonal per mode


def precomp_expsemi_proj(problem, vareps: float) -> ExpsemiProj:
    """Precompute ETD coefficients and the Thomas forward sweep.

    Call once per (grid, vareps) pair; reused every projection call.
    """
    nt, dt = problem.nt, problem.dt

    alpha_vals = vareps * problem.lambda_x * dt  # (nx,)
    c_vals = np.exp(-alpha_vals)                  # (nx,)

    # phi(alpha) = (1 - exp(-alpha)) / alpha, with phi(0) = 1 by L'Hopital.
    phi_vals = np.ones_like(alpha_vals)
    nz = alpha_vals > 1e-14
    phi_vals[nz] = (1 - c_vals[nz]) / alpha_vals[nz]

    # Non-DC modes j=1..nx-1 (0-indexed) for the tridiagonal solve.
    c = c_vals[1:]
    phi = phi_vals[1:]
    lx = problem.lambda_x[1:]

    D = np.broadcast_to((1 + c**2) / dt**2 + phi**2 * lx, (nt, lx.shape[0])).copy()
    D[0, :] = 1 / dt**2 + phi**2 * lx
    D[-1, :] = c**2 / dt**2 + phi**2 * lx

    e = -c / dt**2  # (nx-1,)

    D_mod = thomas_batch_precomp(D, e)
    return ExpsemiProj(c_vals=c_vals, phi_vals=phi_vals, D_mod=D_mod, e=e)


def _apply_semigroup(rho: np.ndarray, c_vals: np.ndarray) -> np.ndarray:
    """S*rho = IDCT_x(c_vals .* DCT_x(rho)) -- exact heat semigroup in x."""
    rho_hat = dct(rho, type=2, norm="ortho", axis=1)
    return idct(rho_hat * c_vals, type=2, norm="ortho", axis=1)


def proj_fokker_planck_expsemi(x_in: State, problem, vareps: float, ep: ExpsemiProj) -> State:
    """Project (rho, mx) onto the FP constraint via the ETD scheme.

    BCs: rho(0,.)=rho0, rho(1,.)=rho1, m=0 at x=0,1 (baked into `problem.ops`).
    """
    ops = problem.ops
    rho0, rho1 = problem.rho0, problem.rho1
    nt, nx = problem.nt, problem.nx
    ntm = nt - 1
    dt = problem.dt
    c_vals, phi_vals = ep.c_vals, ep.phi_vals

    mu, psi = x_in.rho, x_in.mx
    zeros_x = np.zeros(nt)

    # --- ETD FP residual in DCT space ---
    mu_prev = np.concatenate([rho0[None, :], mu], axis=0)  # (nt, nx)
    mu_curr = np.concatenate([mu, rho1[None, :]], axis=0)  # (nt, nx)
    s_prev = _apply_semigroup(mu_prev, c_vals)
    f_rho_hat = dct((mu_curr - s_prev) / dt, type=2, norm="ortho", axis=1)

    dxm = ops.deriv_x_at_phi(psi, zeros_x, zeros_x)
    dxm_hat = dct(dxm, type=2, norm="ortho", axis=1)

    f_hat = f_rho_hat + phi_vals * dxm_hat

    if np.linalg.norm(f_hat) * np.sqrt(dt * problem.dx) < 1e-12:
        return State(mu.copy(), psi.copy())

    # --- Solve T_j phi_j = f_hat(:,j) per DCT-x mode ---
    phi_hat = np.zeros((nt, nx))

    # j=0: DC mode (lambda_x=0, c=1, phi=1), T_0 singular -> DCT-in-t gauge fix.
    f1_t = dct(f_hat[:, 0], type=2, norm="ortho")
    phi1_t = np.zeros(nt)
    lambda_t_col = problem.lambda_t[:, 0]
    phi1_t[1:] = f1_t[1:] / lambda_t_col[1:]
    phi_hat[:, 0] = idct(phi1_t, type=2, norm="ortho")

    # j=1..nx-1: batched tridiagonal solve
    phi_hat[:, 1:] = thomas_batch_solve(ep.D_mod, ep.e, f_hat[:, 1:])

    # --- rho update: adj_rho_hat = (c_j * phi_hat_next - phi_hat_curr)/dt ---
    phi_hat_curr = phi_hat[:ntm, :]
    phi_hat_next = phi_hat[1:nt, :]
    adj_rho_hat = (c_vals * phi_hat_next - phi_hat_curr) / dt
    adj_rho = idct(adj_rho_hat, type=2, norm="ortho", axis=1)

    rho_out = mu + adj_rho

    # --- m update: delta_m = D_x^T (Phi_op * phi) ---
    phi_weighted = idct(phi_vals * phi_hat, type=2, norm="ortho", axis=1)
    mx_out = psi + ops.deriv_x_at_m(phi_weighted)

    return State(rho_out, mx_out)
