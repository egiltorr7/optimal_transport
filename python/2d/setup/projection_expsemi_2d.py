"""Projection onto the 2D Fokker-Planck constraint via an ETD (exponential
time differencing) scheme -- integrates the diffusion exactly per DCT-(x,y)
mode instead of the backward-Euler/CN treatment used in projection_2d.py.

Cross of projection_expsemi.py (1D ETD) and projection_2d.py (2D CN). Each
parent contributes the half it already settled, and the seam between them is
the same one-line substitution that produced projection_2d.py from
projection.py:

    lx = lambda_x[k]   ->   lxy = lambda_x[kx] + lambda_y[ky]

because the discrete Neumann Laplacian's spectrum is additive across axes.
Everything the ETD scheme does is a function of lxy alone, so per DCT mode:

    alpha = vareps * lxy * dt          (dimensionless diffusion per step)
    c     = exp(-alpha)                 (exact semigroup factor)
    phi   = (1 - c) / alpha,  phi(0)=1  (ETD forcing weight)

and the FP constraint becomes, per mode,

    (rho_{k+1} - c*rho_k)/dt + phi*(d_x mx + d_y my) = 0

-- i.e. the diffusion is integrated by a matrix exponential rather than
approximated by a finite difference in t. The *two* momentum components share
a single phi: they enter only through d_x mx + d_y my, whose DCT symbol is the
same lxy that sets alpha (see the T assembly below).

Structure of the mode operator. Writing E for the (nt, ntm) bidiagonal map
q -> (q_k - c*q_{k-1})/dt that the residual applies in time, the per-mode
system is

    T = E E^T + phi^2 * lxy * I

with E E^T tridiagonal, constant-off-diagonal, and closed-form:

    diagonal:      d[0]    = 1/dt^2   + phi^2*lxy
                   d[1:-1] = (1+c^2)/dt^2 + phi^2*lxy
                   d[-1]   = c^2/dt^2 + phi^2*lxy
    off-diagonal:  e = -c/dt^2                    (constant in t per mode)

so the whole Thomas sweep vectorizes over every spatial mode by broadcasting,
exactly as in projection_2d.py, and thomas_batch_precomp/solve are imported
from that module rather than re-derived.

The (kx,ky)=(0,0) DC mode has lxy=0, hence alpha=0, c=1, phi=1 -- so E becomes
the plain first difference and T collapses to the SAME singular discrete
Neumann time-Laplacian M0 that projection_2d.py's DC mode does. It is
therefore inverted by the identical DCT-in-t / lambda_t trick, not by anything
ETD-specific.

Consistency at vareps=0: c=1 and phi=1 for EVERY mode, the semigroup is the
identity, and T reduces to M0 + lxy*I -- which is precomp_banded_proj_2d's
closed form at vareps=0 term for term. So ETD and CN must agree to machine
precision there, and that is the cheapest available check on this module
(scripts/validate_expsemi_2d.py asserts it).

One deliberate numerical deviation from projection_expsemi.py: phi is computed
as -expm1(-alpha)/alpha rather than (1 - exp(-alpha))/alpha with a 1e-14
threshold. The two agree wherever the latter is accurate, but 1 - exp(-alpha)
cancels catastrophically as alpha -> 0, losing ~|log10(alpha)| digits, and
alpha = vareps*lxy*dt is genuinely tiny in the small-vareps regime this
module exists to be tested in (at vareps=1e-8, nx=32 it is ~1e-6). expm1 is
exact there by construction, so the threshold branch is unnecessary; only the
alpha == 0 point itself still needs the L'Hopital value phi=1.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .projection_2d import (DctModes, _stack_diag, dct2_xy, idct2_xy,
                            precomp_dct_modes, solve_time_modes,
                            thomas_batch_precomp)
from .state_2d import State2D


@dataclass
class ExpsemiProj2D:
    c_vals: jnp.ndarray    # (nx, ny)     semigroup factors c = exp(-alpha)
    phi_vals: jnp.ndarray  # (nx, ny)     ETD weights phi = (1-c)/alpha
    D_mod: jnp.ndarray     # (nt, nx, ny) Thomas-modified diagonal, all modes
    e: jnp.ndarray         # (nx, ny)     constant off-diagonal per mode
    dct: DctModes | None = None   # set when method == "dct" (projection_2d.py)


jax.tree_util.register_dataclass(
    ExpsemiProj2D, data_fields=["c_vals", "phi_vals", "D_mod", "e", "dct"], meta_fields=[])


def _etd_coeffs(lxy: jnp.ndarray, vareps: float, dt: float):
    """c = exp(-alpha), phi = (1-c)/alpha for alpha = vareps*lxy*dt.

    phi via -expm1(-alpha)/alpha (see the module docstring on cancellation),
    with the alpha == 0 modes -- always including (kx,ky)=(0,0), and every
    mode when vareps == 0 -- set to phi=1 by L'Hopital.
    """
    alpha = vareps * lxy * dt
    c_vals = jnp.exp(-alpha)
    alpha_safe = jnp.where(alpha == 0, 1.0, alpha)
    phi_vals = jnp.where(alpha == 0, 1.0, -jnp.expm1(-alpha) / alpha_safe)
    return c_vals, phi_vals


def precomp_expsemi_proj_2d(problem, vareps: float, method: str = "thomas") -> ExpsemiProj2D:
    """Precompute the ETD coefficients and the per-mode time systems for all
    (kx,ky) modes.

    Call once per (grid, vareps) pair; reused by every projection call -- same
    contract (and same `method` choice, "thomas" or "dct") as
    projection_2d.py's precomp_banded_proj_2d.
    """
    nt, dt = problem.nt, problem.dt
    lxy = problem.lambda_x[:, None] + problem.lambda_y[None, :]  # (nx, ny), lxy[0,0]=0

    c_vals, phi_vals = _etd_coeffs(lxy, vareps, dt)
    p2l = phi_vals**2 * lxy

    mid = (1.0 + c_vals**2) / dt**2 + p2l
    top = 1.0 / dt**2 + p2l
    bot = c_vals**2 / dt**2 + p2l
    e = -c_vals / dt**2  # (nx, ny)
    D = _stack_diag(nt, top, mid, bot)

    # DC mode: lxy=0 => T = M0, singular. Neutralized inside the batch (diag 1,
    # off-diag 0, RHS zeroed at projection time) and overwritten afterwards via
    # DCT-in-t, exactly as in projection_2d.py.
    D = D.at[:, 0, 0].set(1.0)
    e_th = e.at[0, 0].set(0.0)

    D_mod = thomas_batch_precomp(D, e_th)
    if method not in ("thomas", "dct"):
        raise ValueError(f"unknown method {method!r}")
    dct = precomp_dct_modes(nt, top, mid, bot, e, problem.lambda_t) if method == "dct" else None
    return ExpsemiProj2D(c_vals=c_vals, phi_vals=phi_vals, D_mod=D_mod, e=e_th, dct=dct)


def proj_fokker_planck_expsemi_2d(x_in: State2D, problem, vareps: float,
                                  ep: ExpsemiProj2D) -> State2D:
    """Project (rho, mx, my) onto the ETD-discretized FP constraint

        (rho_{k+1} - S rho_k)/dt + Phi*(d_x mx + d_y my) = 0

    where S = IDCT2 diag(c) DCT2 is the exact heat semigroup over one step and
    Phi = IDCT2 diag(phi) DCT2. BCs: rho(0,.,.)=rho0, rho(1,.,.)=rho1,
    mx=my=0 on the domain walls (baked into `problem.ops`).
    """
    ops = problem.ops
    rho0, rho1 = problem.rho0, problem.rho1
    nt, nx, ny = problem.nt, problem.nx, problem.ny
    ntm = nt - 1
    dt = problem.dt
    c_vals, phi_vals = ep.c_vals, ep.phi_vals

    mu, psi_x, psi_y = x_in.rho, x_in.mx, x_in.my
    zeros_x = jnp.zeros((nt, ny))
    zeros_y = jnp.zeros((nt, nx))

    # --- ETD FP residual, assembled directly in DCT space ---
    # The semigroup is diagonal there, so rho's DCT is taken once and reused
    # for both mu_curr and S*mu_prev rather than transforming twice in
    # physical space the way projection_expsemi.py's _apply_semigroup does
    # (it round-trips through idct only to have the caller dct it straight
    # back). Same arithmetic, two fewer transforms per call.
    # mu_prev = [rho0, mu] and mu_curr = [mu, rho1] share nt-1 of their nt
    # planes, so transform the (nt+1)-plane stack once and slice, rather
    # than transforming each separately: two 2D DCTs instead of four.
    mu_all_hat = dct2_xy(jnp.concatenate([rho0[None, :, :], mu, rho1[None, :, :]], axis=0))  # (nt+1, nx, ny)
    f_rho_hat = (mu_all_hat[1:] - c_vals * mu_all_hat[:-1]) / dt

    div_m = (ops.deriv_x_at_phi(psi_x, zeros_x, zeros_x)
             + ops.deriv_y_at_phi(psi_y, zeros_y, zeros_y))
    f_hat = f_rho_hat + phi_vals * dct2_xy(div_m)

    # No near-zero-residual early exit, for projection_2d.py's reason: the norm
    # check forces a device->host sync every iteration.

    # --- Per-mode time solve (DC mode: alpha=0 => c=1, phi=1 => T = M0,
    #     inverted by DCT-in-t inside solve_time_modes) ---
    phi_hat = solve_time_modes(ep, f_hat, problem.lambda_t)

    # --- Apply R* corrections. -R* is (E^T, -Phi G_x, -Phi G_y) negated; in
    #     DCT space E^T's action is (c*phi_{k+1} - phi_k)/dt, and the momentum
    #     side applies Phi FIRST and then the difference (Phi is diagonal in
    #     DCT space, the difference is not, so the order is not free).
    adj_rho = idct2_xy((c_vals * phi_hat[1:nt] - phi_hat[:ntm]) / dt)
    phi_weighted = idct2_xy(phi_vals * phi_hat)

    rho_out = mu + adj_rho
    mx_out = psi_x + ops.deriv_x_at_m(phi_weighted)
    my_out = psi_y + ops.deriv_y_at_m(phi_weighted)

    return State2D(rho_out, mx_out, my_out)
