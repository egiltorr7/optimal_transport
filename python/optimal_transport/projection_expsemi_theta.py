"""Projection onto the Fokker-Planck constraint via centroid-corrected ETD
("ETD-theta") -- ETD1 with the source-term quadrature bias removed.

Sibling of projection_expsemi.py, which is left untouched as the reference
ETD1 scheme; this module shares its semigroup helper and projection.py's
generalized Thomas sweep rather than re-deriving either.

WHY: ETD1's remaining error is not a semigroup error -- projection_expsemi.py
integrates the diffusion EXACTLY via c_j = exp(-alpha_j).  It is a *quadrature*
error.  The exact step for each DCT-x mode is

    rho_{k+1} = c_j rho_k - int_{t_k}^{t_k+dt} exp(-vareps*lam_j*(t_{k+1}-s)) g(s) ds

with g = D_x m.  ETD1 evaluates g at the interval MIDPOINT t_{k+1/2} (which is
where the staggered grid puts m) and multiplies by the exact weight integral
dt*phi_j.  But the weight exp(-vareps*lam_j*(t_{k+1}-s)) is not symmetric about
the midpoint -- its centroid sits at

    theta(a) = ((a-1) + exp(-a)) / (a*(1 - exp(-a))),     a = vareps*lam_j*dt
    theta -> 1/2 + a/12   (a -> 0)        theta -> 1 - 1/a   (a -> inf)

so the midpoint rule carries a first-order bias dt*(theta-1/2)*g'.  That bias
is zero at vareps = 0 and grows to dt/2 as a -> inf: it is exactly the
vareps-dependent error constant ETD1 suffers from.  Measured effect on the FP
solution error (relative L2, rough sigma=0.05 data, nt=nx=512): ETD1 decays
from 2nd order to p = 1.28 between vareps = 0.01 and vareps = 10, losing a
factor 60 in accuracy, while ETD-theta holds p ~ 2 with a constant flat in
vareps.

FIX: evaluate g at the weight's centroid instead of the midpoint, by
interpolating the two m rows the staggered grid ALREADY provides, with

    omega = theta(alpha) - 1/2   in [0, 1/2)

    g(t_k + theta*dt) ~ (1 - omega)*g_k + omega*g_{k+1}

This reproduces ETD2's quadrature accuracy without ETD2's regridding: the
unbuilt cfg_ladmm_gaussian_expsemi_etd2.m assumed x.mx had to move to nt+1
integer time nodes, but interpolating between the existing half-integer nodes
achieves the same order on the grid the rest of the code already uses.
omega = 0 recovers projection_expsemi.py exactly, so vareps -> 0 and the
lam_j = 0 (DC) mode both degenerate to ETD1 by construction.

STRUCTURE: the m stencil still spans only two adjacent TIME rows, so R R*
stays tridiagonal in t and the same batched Thomas sweep applies.  Relative to
projection_expsemi.py the per-mode system picks up

    diagonal      += phi^2 * lam * ((1-omega)^2 + omega^2)     [was phi^2*lam]
    off-diagonal  += phi^2 * lam * omega*(1-omega)             [was 0]

The one wrinkle: row nt-1 would need g_{nt}, which does not exist (m has nt
rows, the last at t = 1 - dt/2).  It is supplied by constant extrapolation,
g_nt := g_{nt-1}, which collapses that row back to the plain midpoint sample.

That choice is forced, not a convenience.  LINEAR extrapolation
(g_nt := 2*g_{nt-1} - g_{nt-2}) keeps the last row 2nd-order consistent, but
it makes row nt-1 of Theta span columns {nt-2, nt-1}, which then overlaps row
nt-3 (spanning {nt-3, nt-2}) and puts a -omega^2 entry at [nt-3, nt-1]:
Theta Theta^* becomes bandwidth 2, and the Thomas sweep no longer applies.
Any two-point final row does this, so a tridiagonal R R^* requires the final
row to touch column nt-1 alone.  The upside is that Theta is then upper
bidiagonal with diagonal (1-omega, ..., 1-omega, 1), hence nonsingular for
omega < 1 by inspection, so Theta Theta^* is SPD with no extra argument.
scripts/validate_expsemi_theta.py checks the band structure and SPD-ness
directly, and measures what the last row costs.

The off-diagonal is still row-varying in one place (the final entry is
omega rather than omega*(1-omega)), hence thomas_batch_*_general from
projection_nu.py rather than projection.py's constant-off-diagonal pair.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import dct, idct

from .projection_expsemi import _apply_semigroup
from .projection_nu import thomas_batch_precomp_general, thomas_batch_solve_general
from .state import State

# Below this alpha the direct theta(alpha) formula loses its leading digits to
# cancellation: ((alpha-1) + exp(-alpha)) ~ alpha^2/2 is a difference of
# O(1) quantities, so at alpha = 1e-4 the direct form is already wrong in the
# 4th significant digit OF OMEGA (relative error 1.5e-4) even though theta
# itself still looks accurate -- theta ~ 1/2 hides it.  Above the cutoff
# exp(-alpha) underflows harmlessly and the direct form is well conditioned.
# The two branches were compared against adaptive quadrature over 1e-6..3 and
# cross near alpha = 0.02, where both hold ~1e-10 relative error on omega.
_ALPHA_SERIES_CUTOFF = 0.02


def theta_centroid(alpha: np.ndarray) -> np.ndarray:
    """Centroid of the ETD weight exp(-alpha*(1-u)) over u in [0,1].

    Returns values in [1/2, 1): 1/2 at alpha = 0 (the weight is flat, so the
    midpoint is unbiased) rising to 1 as alpha -> inf (the weight collapses
    onto the right endpoint).
    """
    alpha = np.asarray(alpha, dtype=float)
    out = np.empty_like(alpha)

    # theta = 1/2 + a/12 - a^3/720 + O(a^5)  (the a^2 and a^4 terms vanish)
    small = alpha < _ALPHA_SERIES_CUTOFF
    a_s = alpha[small]
    out[small] = 0.5 + a_s / 12.0 - a_s**3 / 720.0

    big = ~small
    a = alpha[big]
    ea = np.exp(-a)
    out[big] = ((a - 1.0) + ea) / (a * (1.0 - ea))
    return out


def omega_weights(alpha: np.ndarray) -> np.ndarray:
    """Interpolation weight onto the next m row: omega = theta(alpha) - 1/2."""
    return theta_centroid(alpha) - 0.5


def _apply_theta(g_hat: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Theta: sample g at the weight centroid instead of the midpoint.

    g_hat: (nt, nx) in DCT-x space.  omega: (nx,), per mode.
      rows 0..nt-2:  (1-omega)*g_k + omega*g_{k+1}
      row  nt-1:     g_{nt-1}         (g_nt unavailable -> constant extrapolation)
    """
    out = np.empty_like(g_hat)
    out[:-1] = (1.0 - omega) * g_hat[:-1] + omega * g_hat[1:]
    out[-1] = g_hat[-1]
    return out


def _apply_theta_adj(v_hat: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Euclidean adjoint of _apply_theta (same shapes).

    Accumulated column-wise from Theta's rows: row k (k <= nt-2) contributes
    (1-omega) to column k and omega to column k+1; row nt-1 contributes 1 to
    column nt-1 alone.
    """
    out = np.zeros_like(v_hat)
    out[:-1] += (1.0 - omega) * v_hat[:-1]
    out[1:] += omega * v_hat[:-1]
    out[-1] += v_hat[-1]
    return out


@dataclass
class ExpsemiThetaProj:
    c_vals: np.ndarray    # (nx,)      semigroup factors c_j = exp(-alpha_j)
    phi_vals: np.ndarray  # (nx,)      ETD weights phi_j = (1-c_j)/alpha_j
    omega: np.ndarray     # (nx,)      centroid weights, omega_j = theta(alpha_j) - 1/2
    main_mod: np.ndarray  # (nt, nx-1) Thomas-modified diagonal, modes j=1..nx-1
    sub: np.ndarray       # (nt-1, nx-1) row-varying off-diagonal, modes j=1..nx-1


def precomp_expsemi_theta_proj(problem, vareps: float) -> ExpsemiThetaProj:
    """Precompute ETD-theta coefficients and the Thomas forward sweep.

    Call once per (grid, vareps) pair; reused every projection call.
    """
    nt, dt = problem.nt, problem.dt
    if nt < 3:
        raise ValueError(f"ETD-theta needs nt >= 3 for the final-row stencil, got {nt}")

    alpha_vals = vareps * problem.lambda_x * dt   # (nx,)
    c_vals = np.exp(-alpha_vals)

    # phi(alpha) = (1 - exp(-alpha))/alpha, with phi(0) = 1 by L'Hopital
    # (same guard as projection_expsemi.py).
    phi_vals = np.ones_like(alpha_vals)
    nz = alpha_vals > 1e-14
    phi_vals[nz] = (1 - c_vals[nz]) / alpha_vals[nz]

    omega_vals = omega_weights(alpha_vals)        # (nx,)

    # --- non-DC modes j=1..nx-1: build the tridiagonal-in-t system ---
    c, phi, w = c_vals[1:], phi_vals[1:], omega_vals[1:]
    lx = problem.lambda_x[1:]
    k = lx.shape[0]

    # R_rho R_rho^*: unchanged from ETD1 (ETD-theta touches only the m block).
    main = np.broadcast_to((1 + c**2) / dt**2, (nt, k)).copy()
    main[0, :] = 1 / dt**2
    main[-1, :] = c**2 / dt**2
    sub = np.broadcast_to(-c / dt**2, (nt - 1, k)).copy()

    # R_m R_m^* = phi^2 * lx * Theta Theta^*.  Theta's final row is a lone 1,
    # so that row's Gram diagonal is 1 and the final off-diagonal is omega
    # (not omega*(1-omega)); every other entry is uniform down the time axis.
    gram = phi**2 * lx
    main += gram * ((1 - w) ** 2 + w**2)
    main[-1, :] = c**2 / dt**2 + gram
    sub += gram * w * (1 - w)
    sub[-1, :] = -c / dt**2 + gram * w

    main_mod = thomas_batch_precomp_general(main, sub)
    return ExpsemiThetaProj(
        c_vals=c_vals, phi_vals=phi_vals, omega=omega_vals, main_mod=main_mod, sub=sub
    )


def proj_fokker_planck_expsemi_theta(
    x_in: State, problem, vareps: float, ep: ExpsemiThetaProj
) -> State:
    """Project (rho, mx) onto the FP constraint via the ETD-theta scheme.

    BCs: rho(0,.)=rho0, rho(1,.)=rho1, m=0 at x=0,1 (baked into `problem.ops`).
    """
    ops = problem.ops
    rho0, rho1 = problem.rho0, problem.rho1
    nt, nx = problem.nt, problem.nx
    ntm = nt - 1
    dt = problem.dt
    c_vals, phi_vals, omega = ep.c_vals, ep.phi_vals, ep.omega

    mu, psi = x_in.rho, x_in.mx
    zeros_x = np.zeros(nt)

    # --- ETD-theta FP residual in DCT space ---
    mu_prev = np.concatenate([rho0[None, :], mu], axis=0)  # (nt, nx)
    mu_curr = np.concatenate([mu, rho1[None, :]], axis=0)  # (nt, nx)
    s_prev = _apply_semigroup(mu_prev, c_vals)
    f_rho_hat = dct((mu_curr - s_prev) / dt, type=2, norm="ortho", axis=1)

    dxm = ops.deriv_x_at_phi(psi, zeros_x, zeros_x)
    dxm_hat = dct(dxm, type=2, norm="ortho", axis=1)

    # Only this line differs from ETD1's residual: g is sampled at the weight
    # centroid rather than at the midpoint.
    f_hat = f_rho_hat + phi_vals * _apply_theta(dxm_hat, omega)

    if np.linalg.norm(f_hat) * np.sqrt(dt * problem.dx) < 1e-12:
        return State(mu.copy(), psi.copy())

    # --- Solve T_j p_j = f_hat(:,j) per DCT-x mode ---
    p_hat = np.zeros((nt, nx))

    # j=0: DC mode (lambda_x=0 => alpha=0, c=1, phi=1, omega=0, and the whole
    # m block drops out), so T_0 is ETD1's singular Neumann-in-t Laplacian --
    # same DCT-in-t gauge fix, unchanged.
    f1_t = dct(f_hat[:, 0], type=2, norm="ortho")
    p1_t = np.zeros(nt)
    lambda_t_col = problem.lambda_t[:, 0]
    p1_t[1:] = f1_t[1:] / lambda_t_col[1:]
    p_hat[:, 0] = idct(p1_t, type=2, norm="ortho")

    # j=1..nx-1: batched tridiagonal solve with a row-varying off-diagonal.
    p_hat[:, 1:] = thomas_batch_solve_general(ep.main_mod, ep.sub, f_hat[:, 1:])

    # --- rho update: R_rho^* is untouched by the theta correction ---
    adj_rho_hat = (c_vals * p_hat[1:nt, :] - p_hat[:ntm, :]) / dt
    rho_out = mu + idct(adj_rho_hat, type=2, norm="ortho", axis=1)

    # --- m update: R_m^* = D_x^* Theta^* Phi ---
    p_weighted = idct(
        _apply_theta_adj(phi_vals * p_hat, omega), type=2, norm="ortho", axis=1
    )
    mx_out = psi + ops.deriv_x_at_m(p_weighted)

    return State(rho_out, mx_out)
