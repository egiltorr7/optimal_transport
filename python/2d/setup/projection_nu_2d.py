"""FP projection in 2D on a non-uniform time grid: DCT-in-(x,y) + a
numerically assembled per-mode banded solve in t.

Cross of projection_2d.py (uniform 2D) and projection_nu.py (non-uniform
1D). Each parent contributes the half it already settled:

  * From projection_2d.py -- the space part. Space stays uniform in x and y,
    so DCT-in-(x,y) still block-diagonalizes the problem, and x/y still
    couple in exactly one place: the *combined* eigenvalue

        lxy = lambda_x[kx] + lambda_y[ky]

    replaces the 1D lx. Also its JAX mechanics: dct2_xy/idct2_xy are
    imported from it rather than re-derived, the Thomas sweeps accumulate
    into a Python list and stack once (JAX arrays are immutable), and the
    (0,0) mode is neutralized inside the batch (diagonal 1, off-diagonal 0,
    zeroed RHS) and overwritten afterwards, so the batch stays rectangular
    instead of carving out a ragged "all modes but one" slice the way the 1D
    code's phi_hat[:, 1:] does.

  * From projection_nu.py -- the time part. With a non-uniform dt_vec the
    per-mode system is no longer Toeplitz, so projection_2d.py's closed-form
    diagonal/off-diagonal entries do not exist, and the singular DC mode
    can no longer be inverted by the DCT-in-t trick (that trick diagonalizes
    the *uniform* time-Laplacian only -- which is why grid_nu_2d.py carries
    no lambda_t). So, exactly as in 1D, the per-mode operator is built by
    *literal substitution* into the projection's own residual/correction
    formulas rather than by hand-rederiving coefficients for a variable
    dt_vec, and the DC mode is inverted with a pseudo-inverse.

The substitution, for a single (kx,ky)-mode's time-profile v (an nt-vector
standing in for phi_hat[:, kx, ky]), is projection_nu.py's verbatim with
lx -> lxy -- the Laplacian's DCT eigenvalue is the only thing the extra y
axis changes:

    nablaxy_phi = -lxy * interp_t_at_rho(v)
    drho        = deriv_t_at_rho(v) + vareps * nablaxy_phi
    check       = -(deriv_t_at_phi(drho, 0, 0) - lxy*v - vareps*lxy*interp_t_at_phi(drho, 0, 0))

`check` is linear in v, and expanding it gives the same quadratic-in-lxy
structure the uniform closed form has,

    T = M0 + lxy*M1 + vareps^2 * lxy^2 * M2

with M0/M1/M2 assembled numerically (one probe, independent of nx and ny)
instead of derived by hand. All three are exactly tridiagonal, being
compositions of the bidiagonal deriv_t/interp_t maps.

N-weighting: the 1D module's N_q^{-1} = 1/(dt_dual*dx) and N_b^{-1} =
1/(dt_vec*dx) become 1/(dt_dual*dx*dy) and 1/(dt_vec*dx*dy) -- the spatial
factor is the cell *area* here (grid_nu_2d.ProblemNU2D.cell_area), space
being uniform in both axes so it stays a scalar. The two momentum
components share one N_b: mx and my live on the same time-centres with the
same cell area, so the b-side contribution is
(1/(dt_vec*dx*dy)) * (D_x D_x^T + D_y D_y^T), whose DCT eigenvalue is
lambda_x + lambda_y = lxy -- i.e. the bare identity-in-time term keeps its
1D form and just picks up the combined eigenvalue, like every other term.

On a uniform grid every weight collapses to the same constant 1/(dt*dx*dy),
which rescales T and the correction by reciprocal factors and cancels
exactly (docs/nonuniform_grid_norms.tex Sec 9.2, Lemma 3) -- which is why
projection_2d.py carries no weights at all, and is the basis of
scripts/validate_nu_2d.py's reduces-to-uniform regression check.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .projection_2d import dct2_xy, idct2_xy
from .state_2d import State2D


def thomas_batch_precomp_general(main: jnp.ndarray, sub: jnp.ndarray) -> jnp.ndarray:
    """Forward sweep for K tridiagonal systems with row-varying, symmetric
    off-diagonals (sub[i] serves as both the sub- and super-diagonal entry
    linking rows i and i+1 -- valid since every T built here is symmetric).

    main: (nt, ...) diagonals.  sub: (nt-1, ...) off-diagonals, trailing dims
    broadcasting against main's (here (nx, ny)). Returns main_mod: (nt, ...).

    Generalizes projection_2d.py's thomas_batch_precomp (single scalar `e`
    shared by every row, the uniform grid's Toeplitz case) to a row-varying
    off-diagonal; reduces to it when sub is constant down its first axis.
    Accumulates into a list and stacks once, for that module's reason: JAX
    arrays are immutable, so projection_nu.py's in-place main_mod[i] = ...
    has no equivalent here.
    """
    nt = main.shape[0]
    rows = [main[0]]
    for i in range(1, nt):
        w = sub[i - 1] / rows[-1]
        rows.append(main[i] - w * sub[i - 1])
    return jnp.stack(rows, axis=0)


def thomas_batch_solve_general(main_mod: jnp.ndarray, sub: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
    """Back-substitute systems precomputed by thomas_batch_precomp_general."""
    nt = rhs.shape[0]

    f_rows = [rhs[0]]
    for i in range(1, nt):
        w = sub[i - 1] / main_mod[i - 1]
        f_rows.append(rhs[i] - w * f_rows[-1])

    x_rows = [f_rows[-1] / main_mod[-1]]
    for i in range(nt - 2, -1, -1):
        x_rows.append((f_rows[i] - sub[i] * x_rows[-1]) / main_mod[i])
    x_rows.reverse()

    return jnp.stack(x_rows, axis=0)


def _probe_composite_ops(ops, nt: int, vareps: float, dt_dual: jnp.ndarray,
                         dt_vec: jnp.ndarray, cell_area: float):
    """M0, M1, M2: (nt, nt) dense (exactly tridiagonal) matrices with
    T = M0 + lxy*M1 + vareps^2*lxy^2*M2 the mode operator (module docstring).

    Probed with the whole identity at once rather than projection_nu.py's
    one-unit-vector-per-iteration Python loop: the time closures act
    independently per spatial column, so laying the nt probes out *along a
    dummy spatial axis* -- v of shape (nt, nt, 1), column j being e_j --
    makes a single call return every column of the matrix. Exact, not an
    approximation, for the same reason the 1D version's nx=1 dummy axis is,
    and it removes the only remaining O(nt) Python loop in the precompute.

    The weights are N_q^{-1} = 1/(dt_dual*cell_area), applied once to drho
    and irho before they feed the outer forward operators (linear, so it
    distributes through M0, M1's eps-term and M2), and N_b^{-1} =
    1/(dt_vec*cell_area) -- a *different* time weight -- on M1's bare
    identity-in-time term, that piece being the b-side (D_x D_x^T +
    D_y D_y^T) contribution rather than the q-side one. See the module
    docstring and docs/nonuniform_grid_norms.tex Sec 9.5.
    """
    zeros_bc = jnp.zeros((nt, 1))            # BC plane for the (nt, nt, 1) probe layout
    v = jnp.eye(nt)[:, :, None]              # (nt, nt, 1): column j is e_j
    dt_dual_col = (dt_dual * cell_area)[:, None, None]
    dt_vec_col = (dt_vec * cell_area)[:, None, None]

    drho = ops.deriv_t_at_rho(v) / dt_dual_col   # (ntm, nt, 1), N_q^{-1} applied once
    irho = ops.interp_t_at_rho(v) / dt_dual_col  # (ntm, nt, 1), N_q^{-1} applied once

    M0 = -ops.deriv_t_at_phi(drho, zeros_bc, zeros_bc)[:, :, 0]
    M1 = (
        vareps * (
            ops.deriv_t_at_phi(irho, zeros_bc, zeros_bc)
            - ops.interp_t_at_phi(drho, zeros_bc, zeros_bc)
        )
        + v / dt_vec_col                          # N_b^{-1}, NOT N_q^{-1}
    )[:, :, 0]
    M2 = ops.interp_t_at_phi(irho, zeros_bc, zeros_bc)[:, :, 0]
    return M0, M1, M2


@dataclass
class BandedProjNU2D:
    main_mod: jnp.ndarray  # (nt, nx, ny)   Thomas-modified diagonal, all modes
    sub: jnp.ndarray       # (nt-1, nx, ny) off-diagonal, all modes
    M0_pinv: jnp.ndarray   # (nt, nt) pseudo-inverse of the singular DC system, T = M0


def precomp_banded_proj_nu_2d(problem, vareps: float) -> BandedProjNU2D:
    """Precompute the per-mode tridiagonal systems. Call once per
    (grid, vareps) pair; reused by every projection call -- same contract as
    projection_2d.py's precomp_banded_proj_2d.
    """
    ops = problem.ops
    nt = problem.nt
    dt_vec, dt_dual = problem.dt_vec, problem.dt_dual
    M0, M1, M2 = _probe_composite_ops(ops, nt, vareps, dt_dual, dt_vec, problem.cell_area)

    main0, sub0 = jnp.diagonal(M0), jnp.diagonal(M0, -1)
    main1, sub1 = jnp.diagonal(M1), jnp.diagonal(M1, -1)
    main2, sub2 = jnp.diagonal(M2), jnp.diagonal(M2, -1)

    lxy = problem.lambda_x[:, None] + problem.lambda_y[None, :]  # (nx, ny), lxy[0,0] = 0
    lxy_b = lxy[None, :, :]

    main = main0[:, None, None] + lxy_b * main1[:, None, None] + vareps**2 * lxy_b**2 * main2[:, None, None]
    sub = sub0[:, None, None] + lxy_b * sub1[:, None, None] + vareps**2 * lxy_b**2 * sub2[:, None, None]

    # (kx,ky)=(0,0) has lxy=0, so T = M0, which is singular (M0 is the
    # discrete Neumann time-Laplacian: constants are in its nullspace).
    # Neutralize it inside the batch so the Thomas sweep stays well defined
    # and rectangular, and invert it separately below -- same device as
    # projection_2d.py's DC handling, with pinv standing in for that
    # module's DCT-in-t (which needs the uniform grid's eigenstructure).
    main = main.at[:, 0, 0].set(1.0)
    sub = sub.at[:, 0, 0].set(0.0)

    main_mod = thomas_batch_precomp_general(main, sub)

    # Gauge freedom: T phi = f fixes phi only up to the nullspace, and any
    # particular solution gives the same physical correction (the correction
    # only ever sees derivatives of phi), so pinv's minimum-norm choice is
    # as good as any -- projection_nu.py's note.
    M0_pinv = jnp.linalg.pinv(M0)

    return BandedProjNU2D(main_mod=main_mod, sub=sub, M0_pinv=M0_pinv)


def proj_fokker_planck_banded_nu_2d(x_in: State2D, problem, vareps: float,
                                    bp: BandedProjNU2D) -> State2D:
    """Project (rho, mx, my) onto the FP constraint

        d_t rho + d_x mx + d_y my = eps * (d_xx + d_yy) rho

    on a non-uniform time grid. BCs: rho(0,.,.)=rho0, rho(1,.,.)=rho1,
    mx=my=0 on the domain walls (baked into `problem.ops`).
    """
    ops = problem.ops
    rho0, rho1 = problem.rho0, problem.rho1
    nt, nx, ny = problem.nt, problem.nx, problem.ny
    dt_vec, dt_dual = problem.dt_vec, problem.dt_dual
    cell_area = problem.cell_area

    mu, psi_x, psi_y = x_in.rho, x_in.mx, x_in.my
    zeros_x = jnp.zeros((nt, ny))
    zeros_y = jnp.zeros((nt, nx))

    # --- FP residual  f = d_t mu + d_x psi_x + d_y psi_y - eps*Delta mu ---
    mu_phi = ops.interp_t_at_phi(mu, rho0, rho1)
    laplacian_mu = (
        ops.deriv_x_at_phi(ops.deriv_x_at_m(mu_phi), zeros_x, zeros_x)
        + ops.deriv_y_at_phi(ops.deriv_y_at_m(mu_phi), zeros_y, zeros_y)
    )
    f = (
        ops.deriv_t_at_phi(mu, rho0, rho1)
        + ops.deriv_x_at_phi(psi_x, zeros_x, zeros_x)
        + ops.deriv_y_at_phi(psi_y, zeros_y, zeros_y)
        - vareps * laplacian_mu
    )

    # No near-zero-residual early exit, unlike projection_nu.py's 1D version:
    # projection_2d.py already dropped it on the grounds that the norm check
    # forces a device->host sync every iteration. Same call here.

    f_hat = dct2_xy(f)  # (nt, nx, ny)

    rhs = f_hat.at[:, 0, 0].set(0.0)
    phi_hat = thomas_batch_solve_general(bp.main_mod, bp.sub, rhs)
    phi_hat = phi_hat.at[:, 0, 0].set(bp.M0_pinv @ f_hat[:, 0, 0])

    phi = idct2_xy(phi_hat)

    # --- Apply the N-weighted R* correction ---
    dphi_dx = ops.deriv_x_at_m(phi)
    dphi_dy = ops.deriv_y_at_m(phi)
    nabla_phi = ops.interp_t_at_rho(
        ops.deriv_x_at_phi(dphi_dx, zeros_x, zeros_x)
        + ops.deriv_y_at_phi(dphi_dy, zeros_y, zeros_y)
    )

    w_q = (dt_dual * cell_area)[:, None, None]  # N_q^{-1}, q-side (rho)
    w_b = (dt_vec * cell_area)[:, None, None]   # N_b^{-1}, b-side (mx, my)

    rho_out = mu + (ops.deriv_t_at_rho(phi) + vareps * nabla_phi) / w_q
    mx_out = psi_x + dphi_dx / w_b
    my_out = psi_y + dphi_dy / w_b

    return State2D(rho_out, mx_out, my_out)
