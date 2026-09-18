"""ETD (exponential time differencing) FP projection on a non-uniform time
grid. Sibling of projection_expsemi.py, generalized the same way
projection_nu.py generalizes projection.py -- see
docs/dyadic_graded_ladmm.tex Section 7.3 for the motivation (grading
shrinks dt near the boundary, which *increases* the local eps/dt ratio
exactly where CN/projection_nu.py is weakest; ETD is unconditionally more
stable there).

N-weighted (docs/nonuniform_grid_norms.tex Sec 9, generalized): R^*=N^{-1}R^T
here too, N_q^{-1}=1/(dt_dual*dx) on the q-correction, N_b^{-1}=1/(dt_vec*dx)
on the b-correction -- same two weights, same reasons (Lemma 3: the
residual/phi space needs no weight of its own) as projection_nu.py.

What's genuinely different from projection_nu.py, and why this module can't
just reuse that derivation: there, the per-mode system T_k = M0 + lx_k*M1 +
lx_k^2*M2 is quadratic in the space eigenvalue lx_k, with M0,M1,M2 built
ONCE (mode-independent) via a dummy nx=1 probe -- the diffusion enters only
through the outer lx_k substitution. Here the ETD coefficients c_vals[n,j],
phi_vals[n,j] = exp(-vareps*lambda_x[j]*dt_vec[n]), (1-c)/alpha depend on
BOTH the time-row n AND the space-DCT-mode j simultaneously (the diffusion
is folded into the time recursion itself, not applied afterward), so the
q-side system genuinely differs mode by mode -- there is no mode-independent
"M0" to build once. Rather than hand-rederive a new closed form incorporating
N_q^{-1} (real risk of an index/bandwidth-tracking error -- the whole reason
projection_nu.py itself uses probing instead of hand algebra), the q-side
system here is built by the same numerical-probing philosophy, just with the
probe vectorized across all nx modes at once (unit impulse in time, kept as
an (nt, nx) array so c_vals/phi_vals's actual per-mode values are used
directly) instead of a dummy nx=1 -- this can't use projection_nu.py's dummy
trick, since the forward map genuinely depends on which mode is being
probed. The b-side's contribution to the system stays closed-form (it was
already a pure diagonal term, phi_vals**2*lambda_x, with no time-coupling to
probe) -- inserting N_b^{-1} there is a single division, verified the same
way as the rest: adjoint identity + a direct feasibility check
(R(x_out)=0), not by trusting the hand algebra alone. See
docs/nonuniform_grid_norms.tex Sec 9 for the CN derivation this generalizes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import dct, idct

from .projection_nu import thomas_batch_precomp_general, thomas_batch_solve_general
from .state import State


@dataclass
class ExpsemiProjNU:
    c_vals: np.ndarray    # (nt, nx)   semigroup coefficients c_vals[n,j] = exp(-vareps*lambda_x[j]*dt_vec[n])
    phi_vals: np.ndarray  # (nt, nx)   ETD weights phi_vals[n,j], ->1 as vareps*lambda_x[j]*dt_vec[n]->0
    dt_dual: np.ndarray    # (nt-1,)    dual-cell width, N_q^{-1} = 1/(dt_dual*dx)
    dt_vec: np.ndarray     # (nt,)      N_b^{-1} = 1/(dt_vec*dx)
    dx: float
    D_mod: np.ndarray      # (nt, nx-1) Thomas-modified diagonal, modes j=1..nx-1
    off: np.ndarray        # (nt-1, nx-1) row-varying off-diagonal, modes j=1..nx-1
    M0_pinv: np.ndarray    # (nt, nt)   pseudo-inverse of the singular j=0 (DC) system


def _probe_RRstar_expsemi(
    nt: int, nx: int, dt_vec: np.ndarray, dx: float, c_vals: np.ndarray, phi_vals: np.ndarray,
    lam_x: np.ndarray, dt_dual: np.ndarray,
) -> np.ndarray:
    """T[:, :, j]: (nt, nt) dense system for mode j, = R (N-weighted R^*), built
    by probing rather than a hand-derived closed form -- see module docstring.
    Probed once per time-index p (nt probes total), vectorized across all nx
    modes simultaneously since c_vals/phi_vals are needed at their actual,
    mode-specific values throughout.
    """
    dt_dual_dx = (dt_dual * dx)[:, None]  # (ntm, 1)
    dt_vec_col = dt_vec[:, None]           # (nt, 1)

    T = np.zeros((nt, nt, nx))
    for p in range(nt):
        v = np.zeros((nt, nx))
        v[p, :] = 1.0
        # q-side: exactly proj_fokker_planck_expsemi_nu's adj_rho_hat formula
        # (== -R_q^T, the module's own established sign convention -- verified
        # against projection.py's uniform-grid closed form), then N_q^{-1}
        # applied once to the whole thing.
        g = c_vals[1:, :] * v[1:, :] / dt_vec[1:, None] - v[:-1, :] / dt_vec[:-1, None]  # (ntm, nx)
        rstar_q = g / dt_dual_dx  # (ntm, nx) = N_q^{-1} * (-R_q^T v)
        # forward R's q-term, applied to rstar_q (zero BC -- linear part only)
        # -- the leading minus here cancels adj_rho_hat's own "-R_q^T" sign,
        # exactly as projection_nu.py's M0 = -deriv_t_at_phi(drho, ...) cancels
        # deriv_t_at_rho's "-D_t^T": without it this column is -R_q N_q^{-1}
        # R_q^T instead of +R_q N_q^{-1} R_q^T, which is what a genuine
        # (positive-semi-definite) Gram-type system needs to be.
        q_curr = np.concatenate([rstar_q, np.zeros((1, nx))], axis=0)
        q_prev = np.concatenate([np.zeros((1, nx)), rstar_q], axis=0)
        T[:, p, :] = -(q_curr - c_vals * q_prev) / dt_vec_col

    # b-side: purely diagonal (no time-coupling to probe -- same closed form
    # as before, N_b^{-1} inserted once, sandwiched between the two phi_vals
    # factors exactly as N_q^{-1} is sandwiched between the two D_t-type
    # applications above).
    for n in range(nt):
        T[n, n, :] += phi_vals[n, :] ** 2 * lam_x / (dt_vec[n] * dx)
    return T


def precomp_expsemi_proj_nu(problem, vareps: float) -> ExpsemiProjNU:
    """Precompute ETD coefficients and the (row-varying) Thomas forward sweep.

    Call once per (grid, vareps) pair; reused every projection call -- same
    contract as precomp_expsemi_proj / precomp_banded_proj_nu.
    """
    nt, nx = problem.nt, problem.nx
    dt_vec = problem.dt_vec  # (nt,)
    dx = problem.dx
    lam_x = problem.lambda_x  # (nx,)
    dt_dual = 0.5 * (dt_vec[:-1] + dt_vec[1:])

    alpha = vareps * dt_vec[:, None] * lam_x[None, :]  # (nt, nx)
    c_vals = np.exp(-alpha)
    phi_vals = np.ones_like(alpha)
    nz = alpha > 1e-14
    phi_vals[nz] = (1.0 - c_vals[nz]) / alpha[nz]

    T = _probe_RRstar_expsemi(nt, nx, dt_vec, dx, c_vals, phi_vals, lam_x, dt_dual)

    diag = np.stack([np.diagonal(T[:, :, j]) for j in range(nx)], axis=1)       # (nt, nx)
    off_full = np.stack([np.diagonal(T[:, :, j], -1) for j in range(nx)], axis=1)  # (nt-1, nx)

    # j=0 (DC, lambda_x=0): dense pseudo-inverse -- see module docstring.
    M0_pinv = np.linalg.pinv(T[:, :, 0])

    # j=1..nx-1: batched row-varying Thomas forward sweep.
    D_mod = thomas_batch_precomp_general(diag[:, 1:], off_full[:, 1:])

    return ExpsemiProjNU(
        c_vals=c_vals, phi_vals=phi_vals, dt_dual=dt_dual, dt_vec=dt_vec, dx=dx,
        D_mod=D_mod, off=off_full[:, 1:], M0_pinv=M0_pinv,
    )


def proj_fokker_planck_expsemi_nu(x_in: State, problem, vareps: float, ep: ExpsemiProjNU) -> State:
    """Project (rho, mx) onto the FP constraint via the ETD scheme, on a
    non-uniform time grid.

    BCs: rho(0,.)=rho0, rho(1,.)=rho1, m=0 at x=0,1 (baked into `problem.ops`).
    """
    ops = problem.ops
    rho0, rho1 = problem.rho0, problem.rho1
    nt, nx = problem.nt, problem.nx
    dt_vec = problem.dt_vec
    dt_dual, dx = ep.dt_dual, ep.dx
    c_vals, phi_vals = ep.c_vals, ep.phi_vals

    mu, psi = x_in.rho, x_in.mx
    zeros_x = np.zeros(nt)

    # --- ETD FP residual in DCT-x space, row n's own dt_vec[n]/c_vals[n,:] ---
    mu_prev = np.concatenate([rho0[None, :], mu], axis=0)  # (nt, nx)
    mu_curr = np.concatenate([mu, rho1[None, :]], axis=0)  # (nt, nx)

    mu_prev_hat = dct(mu_prev, type=2, norm="ortho", axis=1)
    mu_curr_hat = dct(mu_curr, type=2, norm="ortho", axis=1)
    f_rho_hat = (mu_curr_hat - c_vals * mu_prev_hat) / dt_vec[:, None]

    dxm = ops.deriv_x_at_phi(psi, zeros_x, zeros_x)
    dxm_hat = dct(dxm, type=2, norm="ortho", axis=1)

    f_hat = f_rho_hat + phi_vals * dxm_hat

    if np.linalg.norm(f_hat) * np.sqrt(np.mean(dt_vec) * problem.dx) < 1e-12:
        return State(mu.copy(), psi.copy())

    # --- Solve T_j phi_j = f_hat[:,j] per DCT-x mode ---
    phi_hat = np.zeros((nt, nx))
    phi_hat[:, 0] = ep.M0_pinv @ f_hat[:, 0]
    phi_hat[:, 1:] = thomas_batch_solve_general(ep.D_mod, ep.off, f_hat[:, 1:])

    # --- rho correction: adj_rho_hat[q] = c_vals[q+1,:]*phi_hat[q+1,:]/dt_vec[q+1] - phi_hat[q,:]/dt_vec[q] ---
    # then N_q^{-1} = 1/(dt_dual*dx), applied once to the whole thing (docs/
    # nonuniform_grid_norms.tex Sec 9.2, eq. 9 -- linear, so same either way).
    dt_curr = dt_vec[:-1, None]
    dt_next = dt_vec[1:, None]
    c_next = c_vals[1:, :]
    adj_rho_hat = c_next * phi_hat[1:, :] / dt_next - phi_hat[:-1, :] / dt_curr
    adj_rho = idct(adj_rho_hat, type=2, norm="ortho", axis=1)
    rho_out = mu + adj_rho / (dt_dual[:, None] * dx)

    # --- m correction: N_b^{-1} = 1/(dt_vec*dx), a *different* weight. ---
    phi_weighted = idct(phi_vals * phi_hat, type=2, norm="ortho", axis=1)
    mx_out = psi + ops.deriv_x_at_m(phi_weighted) / (dt_vec[:, None] * dx)

    return State(rho_out, mx_out)
