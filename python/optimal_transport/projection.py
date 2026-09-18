"""Projection onto the Fokker-Planck constraint set via DCT-in-x + batched
tridiagonal (Thomas) solve in t.

Port of matlab/shared/1d/utils/precomp_banded_proj.m and
matlab/shared/1d/projection/proj_fokker_planck_banded.m -- but structured
after the *_gpu variants (thomas_batch_precomp/solve,
precomp_banded_proj_gpu), which derive the tridiagonal system in closed
form (constant off-diagonal per spatial mode) instead of building dense
nt x nt matrices and LU-factorizing each of the nx-1 modes separately. That
closed form is vectorized here across all nx-1 modes at once with numpy
broadcasting, replacing MATLAB's per-mode `for k=2:nx` LU-factorization
loop with a `for t in range(nt)` sweep -- so it's O(nt) Python-level loop
iterations (a small constant, unrelated to grid resolution) instead of
O(nx) tridiagonal factorizations.

DCT convention: MATLAB's `dct`/`idct` (Signal Processing Toolbox) are the
orthonormal DCT-II / DCT-III pair, confirmed by
matlab/shared/1d/gpu/dct_rows.m ("Equivalent to dct(x')'... orthonormal
DCT-II"). That's scipy.fft.dct/idct with type=2, norm='ortho'.
"""
from dataclasses import dataclass

import numpy as np
from scipy.fft import dct, idct

from .state import State


def thomas_batch_precomp(D: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Forward sweep for K tridiagonal systems with constant off-diagonal e.

    D: (nt, K) diagonals, column k = diagonal of system k.
    e: (K,)    constant off-diagonal per system (same for all nt-1 entries).
    Returns D_mod: (nt, K) modified diagonal such that thomas_batch_solve
    with (D_mod, e) solves T_k x = f for all K systems simultaneously.
    """
    nt = D.shape[0]
    D_mod = D.copy()
    for t in range(1, nt):
        m = e / D_mod[t - 1]
        D_mod[t] = D_mod[t] - m * e
    return D_mod


def thomas_batch_solve(D_mod: np.ndarray, e: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Back-substitute K tridiagonal systems precomputed by thomas_batch_precomp.

    D_mod: (nt, K) from thomas_batch_precomp.  e: (K,).  F: (nt, K) right-hand
    sides.  Returns X: (nt, K).
    """
    nt = F.shape[0]

    F_mod = np.empty_like(F)
    F_mod[0] = F[0]
    for t in range(1, nt):
        m = e / D_mod[t - 1]
        F_mod[t] = F[t] - m * F_mod[t - 1]

    X = np.empty_like(F)
    X[-1] = F_mod[-1] / D_mod[-1]
    for t in range(nt - 2, -1, -1):
        X[t] = (F_mod[t] - e * X[t + 1]) / D_mod[t]

    return X


@dataclass
class BandedProj:
    D_mod: np.ndarray  # (nt, nx-1) Thomas-modified diagonal, modes k=1..nx-1
    e: np.ndarray       # (nx-1,)    constant off-diagonal per mode


def precomp_banded_proj(problem, vareps: float) -> BandedProj:
    """Precompute the Thomas forward sweep for the nx-1 non-DC spatial modes.

    Call once per (grid, vareps) pair; reused every projection call.

    Per-mode system T_k = M0 + lx_k*diag(M1d) + lx_k^2*M2 (k = the nx-1
    non-zero spatial DCT modes) has closed-form tridiagonal entries:

      diagonal:
        d[0]    = 1/dt^2 + lx_k*(1+eps/dt) + lx_k^2*eps^2/4
        d[1:-1] = 2/dt^2 + lx_k            + lx_k^2*eps^2/2
        d[-1]   = 1/dt^2 + lx_k*(1-eps/dt) + lx_k^2*eps^2/4
      off-diagonal (constant per mode):
        e_k = -1/dt^2 + lx_k^2*eps^2/4

    The k=0 (DC, lx=0) mode is singular here and handled separately at
    projection time via a 1D DCT-in-t.
    """
    nt, dt = problem.nt, problem.dt
    lx = problem.lambda_x[1:]  # (nx-1,) non-DC spatial modes

    D = np.broadcast_to(2 / dt**2 + lx + lx**2 * vareps**2 / 2, (nt, lx.shape[0])).copy()
    D[0, :] = 1 / dt**2 + lx * (1 + vareps / dt) + lx**2 * vareps**2 / 4
    D[-1, :] = 1 / dt**2 + lx * (1 - vareps / dt) + lx**2 * vareps**2 / 4

    e = -1 / dt**2 + lx**2 * vareps**2 / 4  # (nx-1,)

    D_mod = thomas_batch_precomp(D, e)
    return BandedProj(D_mod=D_mod, e=e)


def proj_fokker_planck_banded(x_in: State, problem, vareps: float, bp: BandedProj) -> State:
    """Project (rho, mx) onto the FP constraint  d_t rho + d_x m = eps d_xx rho.

    BCs: rho(0,.)=rho0, rho(1,.)=rho1, m=0 at x=0,1 (baked into `problem.ops`).
    """
    ops = problem.ops
    rho0, rho1 = problem.rho0, problem.rho1
    nt, nx = problem.nt, problem.nx

    mu, psi = x_in.rho, x_in.mx
    zeros_x = np.zeros(nt)

    # --- FP residual  f = d_t mu + d_x psi - eps * d_xx mu  (nt, nx) ---
    laplacian_mu = ops.deriv_x_at_phi(
        ops.deriv_x_at_m(ops.interp_t_at_phi(mu, rho0, rho1)), zeros_x, zeros_x
    )
    f = (
        ops.deriv_t_at_phi(mu, rho0, rho1)
        + ops.deriv_x_at_phi(psi, zeros_x, zeros_x)
        - vareps * laplacian_mu
    )

    if np.linalg.norm(f) * np.sqrt(problem.dt * problem.dx) < 1e-12:
        return State(mu.copy(), psi.copy())

    # --- DCT in x (each time-row independently) ---
    f_hat = dct(f, type=2, norm="ortho", axis=1)  # (nt, nx)
    phi_hat = np.zeros((nt, nx))

    # k=0: DC mode, lambda_x=0 => T_0 singular. Invert via DCT in t;
    # (k=0, l=0) is the gauge freedom -> zero.
    f1_t = dct(f_hat[:, 0], type=2, norm="ortho")
    phi1_t = np.zeros(nt)
    lambda_t_col = problem.lambda_t[:, 0]
    phi1_t[1:] = f1_t[1:] / lambda_t_col[1:]
    phi_hat[:, 0] = idct(phi1_t, type=2, norm="ortho")

    # k=1..nx-1: batched tridiagonal solve
    phi_hat[:, 1:] = thomas_batch_solve(bp.D_mod, bp.e, f_hat[:, 1:])

    # --- IDCT in x -> phi in physical space ---
    phi = idct(phi_hat, type=2, norm="ortho", axis=1)

    # --- Apply A* to get corrections ---
    dphi_dx = ops.deriv_x_at_m(phi)
    nablax_phi = ops.interp_t_at_rho(ops.deriv_x_at_phi(dphi_dx, zeros_x, zeros_x))

    rho_out = mu + ops.deriv_t_at_rho(phi) + vareps * nablax_phi
    mx_out = psi + dphi_dx

    return State(rho_out, mx_out)
