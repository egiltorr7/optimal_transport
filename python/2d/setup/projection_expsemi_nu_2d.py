"""ETD FP projection in 2D on a NON-UNIFORM time grid.

Cross of projection_expsemi_nu.py (1D non-uniform ETD) and
projection_expsemi_2d.py (2D uniform ETD). Each parent contributes the half it
already settled:

  * From projection_expsemi_2d.py -- the space part: DCT in (x,y) still
    block-diagonalizes, x and y still couple only through
    lxy = lambda_x[kx] + lambda_y[ky], and the two momentum components still
    share one phi because they enter only via d_x mx + d_y my.
  * From projection_expsemi_nu.py -- the time part, including the reason this
    module cannot reuse projection_nu_2d.py's derivation. There, the per-mode
    system is T = M0 + lxy*M1 + vareps^2*lxy^2*M2 with M0/M1/M2 built ONCE
    from a mode-independent probe, because the diffusion enters only through
    the outer lxy substitution. Here it does not: the ETD coefficients

        alpha[n,kx,ky] = vareps * dt_vec[n] * lxy[kx,ky]
        c   = exp(-alpha),   phi = (1-c)/alpha

    depend on the time row n AND the spatial mode at once -- the diffusion is
    folded into the time recursion itself -- so the q-side system genuinely
    differs mode by mode and there is no "M0" to factor out.

N-weighting carries over from projection_nu_2d.py with dx -> cell_area:
N_q^{-1} = 1/(dt_dual*dx*dy) on the q (rho) correction, N_b^{-1} =
1/(dt_vec*dx*dy) on the b (mx, my) correction. mx and my share one N_b -- they
live on the same time centres with the same cell area.

Probing strategy, and the one place this improves on the 1D parent rather than
translating it. projection_expsemi_nu.py probes with one unit impulse per time
index (nt probes) and stores a DENSE (nt, nt, nx) tensor, then reads off two
diagonals. The 2D analogue of that is (nt, nt, nx, ny), which is where it stops
being reasonable: at nt=nx=ny=64 it is 134 MB of which all but 3/nt is thrown
away. But T is exactly tridiagonal and symmetric (it is a Gram-type product of
bidiagonal time maps -- asserted, not assumed: scripts/validate_expsemi_nu_2d.py
checks both against a dense build), and a tridiagonal matrix needs only THREE
probes, not nt: probe with the indicator of each residue class p = r (mod 3).
For row k the band {k-1, k, k+1} contains exactly one index of each residue, so

    T[k, k]     = out_{k mod 3}[k]          (diagonal)
    T[k+1, k]   = out_{k mod 3}[k+1]        (sub-diagonal)

reads both diagonals off three (nt, nx, ny) arrays, independent of nt. Columns
within a residue class are 3 apart and their bands are disjoint, so no probe
contaminates another. _probe_T_diagonals below does this; _probe_T_dense is the
1D-style full build kept for the validation script to check it against.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .projection_2d import dct2_xy, idct2_xy
from .projection_nu_2d import (thomas_batch_precomp_general,
                               thomas_batch_solve_general)
from .state_2d import State2D


@dataclass
class ExpsemiProjNU2D:
    c_vals: jnp.ndarray    # (nt, nx, ny) semigroup factors exp(-alpha)
    phi_vals: jnp.ndarray  # (nt, nx, ny) ETD weights (1-c)/alpha
    main_mod: jnp.ndarray  # (nt, nx, ny)   Thomas-modified diagonal, all modes
    sub: jnp.ndarray       # (nt-1, nx, ny) off-diagonal, all modes
    M0_pinv: jnp.ndarray   # (nt, nt) pseudo-inverse of the singular (0,0) system


jax.tree_util.register_dataclass(
    ExpsemiProjNU2D, data_fields=["c_vals", "phi_vals", "main_mod", "sub", "M0_pinv"], meta_fields=[])


def _etd_coeffs_nu(lxy: jnp.ndarray, vareps: float, dt_vec: jnp.ndarray):
    """c, phi of shape (nt, nx, ny). phi via -expm1(-alpha)/alpha -- see
    projection_expsemi_2d._etd_coeffs for why the naive (1-exp(-a))/a form is
    not good enough in the small-vareps regime."""
    alpha = vareps * dt_vec[:, None, None] * lxy[None, :, :]
    c_vals = jnp.exp(-alpha)
    alpha_safe = jnp.where(alpha == 0, 1.0, alpha)
    phi_vals = jnp.where(alpha == 0, 1.0, -jnp.expm1(-alpha) / alpha_safe)
    return c_vals, phi_vals


def _T_action(v, c_vals, phi_vals, lxy, dt_vec, dt_dual, cell_area):
    """T v = R (N^{-1} R^T) v for every (kx,ky) mode at once.

    Term for term the composite that projection_expsemi_nu.py probes, with
    dx -> cell_area and lam_x -> lxy. The leading minus on the q-side cancels
    the "-R_q^T" sign built into the adjoint formula, so the result is
    +R_q N_q^{-1} R_q^T (positive semi-definite) rather than its negative --
    projection_nu_2d.py's M0 = -deriv_t_at_phi(drho, ...) does the same thing.
    """
    dtv = dt_vec[:, None, None]
    g = c_vals[1:] * v[1:] / dtv[1:] - v[:-1] / dtv[:-1]      # (ntm, ...) = -R_q^T v
    rstar_q = g / (dt_dual * cell_area)[:, None, None]         # N_q^{-1} applied once
    zero = jnp.zeros_like(v[:1])
    q_curr = jnp.concatenate([rstar_q, zero], axis=0)
    q_prev = jnp.concatenate([zero, rstar_q], axis=0)
    q_side = -(q_curr - c_vals * q_prev) / dtv
    # b-side is a pure diagonal (no time coupling), N_b^{-1} sandwiched between
    # the two phi factors exactly as N_q^{-1} sits between the two D_t maps.
    b_side = phi_vals**2 * lxy[None, :, :] / (dtv * cell_area) * v
    return q_side + b_side


def _probe_T_diagonals(c_vals, phi_vals, lxy, dt_vec, dt_dual, cell_area):
    """(main, sub) of shapes (nt, nx, ny) and (nt-1, nx, ny) via 3 probes."""
    nt = dt_vec.shape[0]
    idx = jnp.arange(nt)
    outs = []
    for r in range(3):
        mask = (idx % 3 == r).astype(c_vals.dtype)[:, None, None]
        v = jnp.broadcast_to(mask, c_vals.shape)
        outs.append(_T_action(v, c_vals, phi_vals, lxy, dt_vec, dt_dual, cell_area))
    stack = jnp.stack(outs, axis=0)                       # (3, nt, nx, ny)
    # diagonal: row k comes from the probe whose class is k mod 3
    main = jnp.take_along_axis(stack, (idx % 3)[None, :, None, None], axis=0)[0]
    # sub-diagonal T[k+1, k]: row k+1 from the probe of class k mod 3
    sub = jnp.take_along_axis(stack[:, 1:], (idx[:-1] % 3)[None, :, None, None], axis=0)[0]
    return main, sub


def _probe_T_dense(c_vals, phi_vals, lxy, dt_vec, dt_dual, cell_area):
    """The 1D-style dense (nt, nt, nx, ny) build: T[:, p] = T e_p. Only for the
    validation script to check _probe_T_diagonals (and the tridiagonality and
    symmetry it relies on) against -- never called by the solver."""
    nt = dt_vec.shape[0]
    cols = []
    for p in range(nt):
        v = jnp.zeros_like(c_vals).at[p].set(1.0)
        cols.append(_T_action(v, c_vals, phi_vals, lxy, dt_vec, dt_dual, cell_area))
    return jnp.stack(cols, axis=1)                        # (nt, nt, nx, ny)


def precomp_expsemi_proj_nu_2d(problem, vareps: float) -> ExpsemiProjNU2D:
    """Precompute the ETD coefficients and the row-varying Thomas forward sweep.

    Call once per (grid, vareps) pair -- same contract as
    precomp_expsemi_proj_2d / precomp_banded_proj_nu_2d.
    """
    nt = problem.nt
    dt_vec, dt_dual = problem.dt_vec, problem.dt_dual
    cell_area = problem.cell_area
    lxy = problem.lambda_x[:, None] + problem.lambda_y[None, :]  # (nx, ny), lxy[0,0]=0

    c_vals, phi_vals = _etd_coeffs_nu(lxy, vareps, dt_vec)
    main, sub = _probe_T_diagonals(c_vals, phi_vals, lxy, dt_vec, dt_dual, cell_area)

    # (kx,ky)=(0,0): lxy=0 => alpha=0 => c=1, phi=1, the b-side vanishes and T
    # collapses to the non-uniform discrete Neumann time-Laplacian, which is
    # singular (constants are in its nullspace). Built densely on a dummy
    # spatial axis of size 1 -- projection_nu_2d.py's device -- and inverted
    # with a pseudo-inverse; the gauge freedom is harmless because the
    # correction only ever sees differences of phi.
    ones = jnp.ones((nt, 1, 1))
    T_dc = _probe_T_dense(ones, ones, jnp.zeros((1, 1)), dt_vec, dt_dual, cell_area)
    M0_pinv = jnp.linalg.pinv(T_dc[:, :, 0, 0])

    # Neutralize that mode inside the batch so the sweep stays rectangular.
    main = main.at[:, 0, 0].set(1.0)
    sub = sub.at[:, 0, 0].set(0.0)
    main_mod = thomas_batch_precomp_general(main, sub)

    return ExpsemiProjNU2D(c_vals=c_vals, phi_vals=phi_vals,
                           main_mod=main_mod, sub=sub, M0_pinv=M0_pinv)


def proj_fokker_planck_expsemi_nu_2d(x_in: State2D, problem, vareps: float,
                                     ep: ExpsemiProjNU2D) -> State2D:
    """Project (rho, mx, my) onto the ETD-discretized FP constraint on a
    non-uniform time grid.

    BCs: rho(0,.,.)=rho0, rho(1,.,.)=rho1, mx=my=0 on the walls (in problem.ops).
    """
    ops = problem.ops
    rho0, rho1 = problem.rho0, problem.rho1
    nt, nx, ny = problem.nt, problem.nx, problem.ny
    dt_vec, dt_dual = problem.dt_vec, problem.dt_dual
    cell_area = problem.cell_area
    c_vals, phi_vals = ep.c_vals, ep.phi_vals

    mu, psi_x, psi_y = x_in.rho, x_in.mx, x_in.my
    zeros_x = jnp.zeros((nt, ny))
    zeros_y = jnp.zeros((nt, nx))
    dtv = dt_vec[:, None, None]

    # --- ETD FP residual in DCT-(x,y) space, each row with its own dt_vec[n] ---
    mu_prev = jnp.concatenate([rho0[None, :, :], mu], axis=0)
    mu_curr = jnp.concatenate([mu, rho1[None, :, :]], axis=0)
    f_rho_hat = (dct2_xy(mu_curr) - c_vals * dct2_xy(mu_prev)) / dtv

    div_m = (ops.deriv_x_at_phi(psi_x, zeros_x, zeros_x)
             + ops.deriv_y_at_phi(psi_y, zeros_y, zeros_y))
    f_hat = f_rho_hat + phi_vals * dct2_xy(div_m)

    rhs = f_hat.at[:, 0, 0].set(0.0)
    phi_hat = thomas_batch_solve_general(ep.main_mod, ep.sub, rhs)
    phi_hat = phi_hat.at[:, 0, 0].set(ep.M0_pinv @ f_hat[:, 0, 0])

    # --- N-weighted R* corrections ---
    adj_rho_hat = c_vals[1:] * phi_hat[1:] / dtv[1:] - phi_hat[:-1] / dtv[:-1]
    adj_rho = idct2_xy(adj_rho_hat) / (dt_dual * cell_area)[:, None, None]
    phi_weighted = idct2_xy(phi_vals * phi_hat)
    w_b = dtv * cell_area

    return State2D(mu + adj_rho,
                   psi_x + ops.deriv_x_at_m(phi_weighted) / w_b,
                   psi_y + ops.deriv_y_at_m(phi_weighted) / w_b)
