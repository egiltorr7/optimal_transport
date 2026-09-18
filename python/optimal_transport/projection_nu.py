"""FP projection on a non-uniform time grid, via DCT-in-x + a numerically
assembled per-mode banded solve in t. Sibling of projection.py's "banded"
(Crank-Nicolson-style) FP projection.

Space stays uniform, so the x-part of projection.py's derivation carries
over unchanged: DCT-in-x block-diagonalizes the problem into nx independent
per-spatial-mode systems (deriv_x_at_phi(deriv_x_at_m(.)) is diagonalized by
DCT-II with eigenvalue lambda_x[k], same as in projection.py -- this fact
never depended on the time grid). What differs is the time part: with a
non-uniform dt_vec, the per-mode system is no longer Toeplitz, so it can't
be written in projection.py's closed form (nor can the singular k=0 mode be
inverted by the DCT-in-t trick, which relies on the same uniform-grid
eigenstructure).

Rather than hand-rederive the closed-form coefficients for a variable dt_vec
(easy to get a sign or index wrong with no way to self-check the algebra),
this module builds the per-mode operator by *literal substitution* into
projection.py's own residual/correction formulas -- reusing
ops.deriv_t_at_phi / ops.deriv_t_at_rho / ops.interp_t_at_phi /
ops.interp_t_at_rho exactly as proj_fokker_planck_banded does, with
deriv_x_at_phi(deriv_x_at_m(.)) replaced by multiplication by the scalar
-lambda_x[k] (verified numerically -- the composed operator's DCT-x
eigenvalue is the *negative* of lambda_x's usual positive-Laplacian-eigenvalue
convention). Concretely, for a single x-mode's time-profile v (an nt-vector
standing in for phi_hat[:, k]):

    nablax_phi = -lx * interp_t_at_rho(v)
    drho       = deriv_t_at_rho(v) + vareps * nablax_phi
    check      = -(deriv_t_at_phi(drho, 0, 0) - lx*v - vareps*lx*interp_t_at_phi(drho, 0, 0))

reproduces proj_fokker_planck_banded's rho_out/mx_out update chain and its
"check" (the residual `f` recomputed from the correction) verbatim, mode by
mode, up to the overall sign fixed below -- so whatever sign convention that
formula relies on is inherited automatically, not re-derived (the outer
negation is a free choice: T_k phi=f_hat and -T_k phi=-f_hat have the same
solution, but flipping only T_k while solving against the *original* f_hat
would not, so it has to be applied consistently -- checked directly against
projection.py's closed-form T_k, see validate_nu.py). `check` is linear in
v; expanding it out (deriv_t_at_phi/interp_t_at_phi are linear in their
first argument at zero BC) gives exactly the quadratic-in-lx structure
projection.py's closed form has, T_k = M0 + lx*M1 + vareps^2*lx^2*M2, but
with M0/M1/M2 assembled numerically (one nt x nt probe each, independent of
nx) instead of derived by hand. All three are exactly tridiagonal
(compositions of the bidiagonal deriv_t/interp_t maps), verified by
regression against projection.py on a uniform time_grid (see
validate_nu.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import dct, idct

from .state import State


def thomas_batch_precomp_general(main: np.ndarray, sub: np.ndarray) -> np.ndarray:
    """Forward sweep for K tridiagonal systems with row-varying, symmetric
    off-diagonals (sub[i, k] used as both the sub- and super-diagonal entry
    of row i+1/i -- valid here since every T_k built below is symmetric).

    main: (nt, K) diagonals.  sub: (nt-1, K) off-diagonal, sub[i,k] links
    rows i and i+1 of system k.  Generalizes projection.py's
    thomas_batch_precomp (which assumes a single scalar e shared by every
    row) to a row-varying off-diagonal; reduces to it when sub is constant
    down its first axis.
    """
    nt = main.shape[0]
    main_mod = main.copy()
    for i in range(1, nt):
        w = sub[i - 1] / main_mod[i - 1]
        main_mod[i] = main[i] - w * sub[i - 1]
    return main_mod


def thomas_batch_solve_general(main_mod: np.ndarray, sub: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    nt = rhs.shape[0]
    f_mod = np.empty_like(rhs)
    f_mod[0] = rhs[0]
    for i in range(1, nt):
        w = sub[i - 1] / main_mod[i - 1]
        f_mod[i] = rhs[i] - w * f_mod[i - 1]

    x = np.empty_like(rhs)
    x[-1] = f_mod[-1] / main_mod[-1]
    for i in range(nt - 2, -1, -1):
        x[i] = (f_mod[i] - sub[i] * x[i + 1]) / main_mod[i]
    return x


def _probe_composite_ops(
    ops, nt: int, vareps: float, dt_dual: np.ndarray, dt_vec: np.ndarray, dx: float
):
    """M0, M1, M2: (nt, nt) dense (exactly tridiagonal) matrices such that
    T_k = M0 + lx_k*M1 + vareps^2*lx_k^2*M2 is the mode-k operator -- see
    module docstring for the derivation. Probed once with unit vectors
    against a dummy singleton spatial axis (the time closures act
    independently per x-column, so nx=1 here is exact, not an approximation).

    N-weighted (docs/nonuniform_grid_norms.tex Sec 9.5, eq. 14): N_q^{-1} =
    1/(dt_dual*dx) divides drho/irho before they feed the outer forward
    operators -- linear, so it distributes automatically through M0 and
    M1's eps-term and M2 -- and N_b^{-1} = 1/(dt_vec*dx), a *different*
    weight, divides M1's bare identity-in-time term separately, since that
    piece is the b-side D_x D_x^T contribution, not the q-side one (both
    N_q, N_b carry the same dx -- space uniform -- so it's passed once, not
    threaded separately per side). The space eigenvalues (lx_k, entering
    only outside this function, in precomp_banded_proj_nu) are untouched by
    either -- Sec 9.5 shows every term here factors as (time-only) x
    (space-only), so N never lands inside a space-direction factor; dx
    itself is a scalar, so it doesn't touch that argument either. Reduces
    exactly to the original (Euclidean) M0,M1,M2 up to an overall constant
    (1/(dt*dx)) on a uniform grid, which does not change the projection
    (Lemma 3, docs/nonuniform_grid_norms.tex Sec 9.2: a global rescaling of
    N cancels between the system solve and the correction).
    """
    zero1 = np.zeros(1)
    M0 = np.zeros((nt, nt))
    M1 = np.zeros((nt, nt))
    M2 = np.zeros((nt, nt))
    dt_dual_col = (dt_dual * dx)[:, None]
    dt_vec_dx = dt_vec * dx
    for j in range(nt):
        v = np.zeros((nt, 1))
        v[j, 0] = 1.0
        drho = ops.deriv_t_at_rho(v) / dt_dual_col     # (ntm, 1), N_q^{-1} applied once
        irho = ops.interp_t_at_rho(v) / dt_dual_col     # (ntm, 1), N_q^{-1} applied once
        M0[:, j] = -ops.deriv_t_at_phi(drho, zero1, zero1)[:, 0]
        M1[:, j] = (
            vareps * (
                ops.deriv_t_at_phi(irho, zero1, zero1)[:, 0]
                - ops.interp_t_at_phi(drho, zero1, zero1)[:, 0]
            )
            + v[:, 0] / dt_vec_dx                        # N_b^{-1}, NOT N_q^{-1}
        )
        M2[:, j] = ops.interp_t_at_phi(irho, zero1, zero1)[:, 0]
    return M0, M1, M2


@dataclass
class BandedProjNU:
    main0: np.ndarray  # (nt,)   diagonal of M0
    sub0: np.ndarray    # (nt-1,) off-diagonal of M0
    main1: np.ndarray
    sub1: np.ndarray
    main2: np.ndarray
    sub2: np.ndarray
    D_mod: np.ndarray   # (nt, nx-1) Thomas-modified diagonal, modes k=1..nx-1
    sub: np.ndarray      # (nt-1, nx-1) off-diagonal, modes k=1..nx-1
    M0_pinv: np.ndarray  # (nt, nt) pseudo-inverse of the singular k=0 (lx=0) system, T_0 = M0


def precomp_banded_proj_nu(problem, vareps: float) -> BandedProjNU:
    """Precompute the per-mode tridiagonal systems. Call once per
    (grid, vareps) pair; reused every projection call -- same contract as
    projection.py's precomp_banded_proj.
    """
    ops = problem.ops
    nt = problem.nt
    dt_vec = problem.dt_vec
    dt_dual = 0.5 * (dt_vec[:-1] + dt_vec[1:])
    M0, M1, M2 = _probe_composite_ops(ops, nt, vareps, dt_dual, dt_vec, problem.dx)

    main0, sub0 = np.diagonal(M0).copy(), np.diagonal(M0, -1).copy()
    main1, sub1 = np.diagonal(M1).copy(), np.diagonal(M1, -1).copy()
    main2, sub2 = np.diagonal(M2).copy(), np.diagonal(M2, -1).copy()

    lx = problem.lambda_x[1:]  # (nx-1,) non-DC spatial modes
    main = main0[:, None] + lx[None, :] * main1[:, None] + vareps**2 * lx[None, :]**2 * main2[:, None]
    sub = sub0[:, None] + lx[None, :] * sub1[:, None] + vareps**2 * lx[None, :]**2 * sub2[:, None]

    D_mod = thomas_batch_precomp_general(main, sub)

    # k=0 (lx=0): T_0 = M0, singular -- see module docstring / gauge-freedom
    # note in operators_nu.py's counterpart: any particular solution gives
    # the same physical correction, so lstsq's minimum-norm choice is fine.
    M0_pinv = np.linalg.pinv(M0)

    return BandedProjNU(
        main0=main0, sub0=sub0, main1=main1, sub1=sub1, main2=main2, sub2=sub2,
        D_mod=D_mod, sub=sub, M0_pinv=M0_pinv,
    )


def proj_fokker_planck_banded_nu(x_in: State, problem, vareps: float, bp: BandedProjNU) -> State:
    """Project (rho, mx) onto the FP constraint on a non-uniform time grid.

    Same structure as projection.py's proj_fokker_planck_banded: DCT in x,
    solve the reduced system per mode, IDCT, apply the correction. Only the
    time-direction solve differs (generic banded instead of closed-form
    Toeplitz + DCT-in-t).
    """
    ops = problem.ops
    rho0, rho1 = problem.rho0, problem.rho1
    nt, nx = problem.nt, problem.nx
    dt_vec = problem.dt_vec
    dt_dual = 0.5 * (dt_vec[:-1] + dt_vec[1:])

    mu, psi = x_in.rho, x_in.mx
    zeros_x = np.zeros(nt)

    laplacian_mu = ops.deriv_x_at_phi(
        ops.deriv_x_at_m(ops.interp_t_at_phi(mu, rho0, rho1)), zeros_x, zeros_x
    )
    f = (
        ops.deriv_t_at_phi(mu, rho0, rho1)
        + ops.deriv_x_at_phi(psi, zeros_x, zeros_x)
        - vareps * laplacian_mu
    )

    if np.linalg.norm(f) * np.sqrt(np.mean(problem.dt_vec) * problem.dx) < 1e-12:
        return State(mu.copy(), psi.copy())

    f_hat = dct(f, type=2, norm="ortho", axis=1)
    phi_hat = np.zeros((nt, nx))

    phi_hat[:, 0] = bp.M0_pinv @ f_hat[:, 0]
    phi_hat[:, 1:] = thomas_batch_solve_general(bp.D_mod, bp.sub, f_hat[:, 1:])

    phi = idct(phi_hat, type=2, norm="ortho", axis=1)

    dphi_dx = ops.deriv_x_at_m(phi)
    nablax_phi = ops.interp_t_at_rho(ops.deriv_x_at_phi(dphi_dx, zeros_x, zeros_x))

    # N-weighted R^* (docs/nonuniform_grid_norms.tex Sec 9.2, eq. 9): N_q^{-1}
    # = 1/(dt_dual*dx), applied once to the whole q-correction (linear, so
    # same either way); N_b^{-1} = 1/(dt_vec*dx) -- a *different* time
    # weight, same dx -- to the b-correction.
    rho_out = mu + (ops.deriv_t_at_rho(phi) + vareps * nablax_phi) / (dt_dual[:, None] * problem.dx)
    mx_out = psi + dphi_dx / (dt_vec[:, None] * problem.dx)

    return State(rho_out, mx_out)
