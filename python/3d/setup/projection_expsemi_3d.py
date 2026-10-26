"""ETD (exponential time differencing) FP projection in 3D.

Port of 2d/setup/projection_expsemi_2d.py with lxy -> lxyz and dct2 -> dct3;
see that module for the derivation. Per DCT mode (kx,ky,kz):

    alpha = vareps * lxyz * dt,   c = exp(-alpha),   phi = (1 - c)/alpha   (phi(0)=1)

and the FP constraint (rho_{k+1} - c rho_k)/dt + phi*(div m) = 0. The per-mode
time operator T = E E^T + phi^2 lxyz I is tridiagonal with

    d[0] = 1/dt^2 + p2l,  d[1:-1] = (1+c^2)/dt^2 + p2l,  d[-1] = c^2/dt^2 + p2l,
    e = -c/dt^2,          p2l = phi^2 * lxyz,

i.e. the same "constant interior + two end deviations" structure projection_3d's
two solvers handle (scan-Thomas or DCT-in-t + Woodbury). At vareps = 0 it
reduces to the CN/banded operator term for term, which validate_3d.py checks.

phi is computed as -expm1(-alpha)/alpha (no cancellation as alpha -> 0).
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .projection_3d import (DctModes, _stack_diag, dct3_xyz, idct3_xyz, lambda_xyz,
                            precomp_dct_modes, solve_time_modes, thomas_batch_precomp)
from .state_3d import State3D


@dataclass
class ExpsemiProj3D:
    c_vals: jnp.ndarray    # (nx, ny, nz)     semigroup factors c = exp(-alpha)
    phi_vals: jnp.ndarray  # (nx, ny, nz)     ETD weights phi = (1-c)/alpha
    D_mod: jnp.ndarray     # (nt, nx, ny, nz) Thomas-modified diagonal, all modes
    e: jnp.ndarray         # (nx, ny, nz)     constant off-diagonal per mode
    dct: DctModes | None = None   # set when method == "dct"


jax.tree_util.register_dataclass(
    ExpsemiProj3D, data_fields=["c_vals", "phi_vals", "D_mod", "e", "dct"], meta_fields=[])


def _etd_coeffs(lxyz: jnp.ndarray, vareps: float, dt: float):
    alpha = vareps * lxyz * dt
    c_vals = jnp.exp(-alpha)
    alpha_safe = jnp.where(alpha == 0, 1.0, alpha)
    phi_vals = jnp.where(alpha == 0, 1.0, -jnp.expm1(-alpha) / alpha_safe)
    return c_vals, phi_vals


def precomp_expsemi_proj_3d(problem, vareps: float, method: str = "thomas") -> ExpsemiProj3D:
    """Precompute the ETD coefficients and the per-mode time systems. Call
    once per (grid, vareps); `method` is "thomas" or "dct" as in
    projection_3d.precomp_banded_proj_3d."""
    if method not in ("thomas", "dct"):
        raise ValueError(f"unknown method {method!r}")
    nt, dt = problem.nt, problem.dt
    lxyz = lambda_xyz(problem)

    c_vals, phi_vals = _etd_coeffs(lxyz, vareps, dt)
    p2l = phi_vals**2 * lxyz
    mid = (1.0 + c_vals**2) / dt**2 + p2l
    top = 1.0 / dt**2 + p2l
    bot = c_vals**2 / dt**2 + p2l
    e = -c_vals / dt**2
    D = _stack_diag(nt, top, mid, bot)

    # DC mode: singular T; neutralised in the batch, overwritten in solve_time_modes.
    D = D.at[:, 0, 0, 0].set(1.0)
    e_th = e.at[0, 0, 0].set(0.0)
    D_mod = thomas_batch_precomp(D, e_th)
    dct = precomp_dct_modes(nt, top, mid, bot, e, problem.lambda_t) if method == "dct" else None
    return ExpsemiProj3D(c_vals=c_vals, phi_vals=phi_vals, D_mod=D_mod, e=e_th, dct=dct)


def proj_fokker_planck_expsemi_3d(x_in: State3D, problem, vareps: float,
                                  ep: ExpsemiProj3D) -> State3D:
    """Project (rho, mx, my, mz) onto the ETD-discretised FP constraint
        (rho_{k+1} - S rho_k)/dt + Phi*(d_x mx + d_y my + d_z mz) = 0,
    S = IDCT3 diag(c) DCT3 the exact heat semigroup over one step."""
    ops = problem.ops
    rho0, rho1 = problem.rho0, problem.rho1
    nt, nx, ny, nz = problem.nt, problem.nx, problem.ny, problem.nz
    ntm = nt - 1
    dt = problem.dt
    c_vals, phi_vals = ep.c_vals, ep.phi_vals
    dtype = x_in.rho.dtype

    mu, psi_x, psi_y, psi_z = x_in.rho, x_in.mx, x_in.my, x_in.mz
    zeros_x = jnp.zeros((nt, ny, nz), dtype)
    zeros_y = jnp.zeros((nt, nx, nz), dtype)
    zeros_z = jnp.zeros((nt, nx, ny), dtype)

    # ETD residual assembled in DCT space: one transform of the (nt+1)-plane
    # stack [rho0, mu, rho1] serves both mu_curr and S*mu_prev.
    mu_all_hat = dct3_xyz(jnp.concatenate([rho0[None], mu, rho1[None]], axis=0))  # (nt+1, ...)
    f_rho_hat = (mu_all_hat[1:] - c_vals * mu_all_hat[:-1]) / dt

    div_m = (ops.deriv_x_at_phi(psi_x, zeros_x, zeros_x)
             + ops.deriv_y_at_phi(psi_y, zeros_y, zeros_y)
             + ops.deriv_z_at_phi(psi_z, zeros_z, zeros_z))
    f_hat = f_rho_hat + phi_vals * dct3_xyz(div_m)

    phi_hat = solve_time_modes(ep, f_hat, problem.lambda_t)

    # R* corrections: E^T in DCT space is (c*phi_{k+1} - phi_k)/dt; the
    # momentum side applies Phi first (diagonal in DCT space), then the difference.
    adj_rho = idct3_xyz((c_vals * phi_hat[1:nt] - phi_hat[:ntm]) / dt)
    phi_weighted = idct3_xyz(phi_vals * phi_hat)

    return State3D(mu + adj_rho,
                   psi_x + ops.deriv_x_at_m(phi_weighted),
                   psi_y + ops.deriv_y_at_m(phi_weighted),
                   psi_z + ops.deriv_z_at_m(phi_weighted))
