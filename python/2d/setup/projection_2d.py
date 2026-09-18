"""Projection onto the 2D Fokker-Planck constraint set via DCT-in-(x,y) +
batched tridiagonal (Thomas) solve in t.

Port of matlab/shared/2d/utils/precomp_banded_proj.m and
matlab/shared/2d/projection/proj_fokker_planck_banded.m -- but, like
projection.py's 1D port, structured after the closed-form *_gpu variant
(matlab/shared/1d/utils/precomp_banded_proj_gpu.m) rather than the
dense-per-mode-LU version: the per-(kx,ky) tridiagonal system has a
*constant* off-diagonal (independent of the time index), so the whole
Thomas sweep vectorizes over every spatial mode via broadcasting instead
of factorizing nx*ny separate systems. Under DCT-xy the FP operator
decouples into nx*ny independent systems indexed by the *combined*
eigenvalue lambda_xy = lambda_x[kx] + lambda_y[ky] -- otherwise the
per-mode system (and its closed-form diagonal/off-diagonal entries) is
exactly the 1D one with lambda_x[k] replaced by lambda_xy.

Two things are done differently here than in the MATLAB code specifically
because this runs on JAX, not because they're "improvements" over it:

  1. No dct/idct call exists in jax.numpy or jax.scipy (unlike scipy.fft,
     which projection.py's 1D code uses directly), so the DCT-II is
     implemented via the mirror-and-FFT trick from
     matlab/shared/1d/gpu/dct_rows.m -- and, like
     matlab/shared/2d/projection/proj_fokker_planck_expsemi_gpu.m's
     dct2_xy (its most GPU-polished 2D projection), applied along an
     `axis=` argument directly rather than proj_fokker_planck_banded.m's
     permute+reshape+dct2_xy, since jnp.fft.fft(..., axis=k) needs no
     permute to hit any axis and dispatches to cuFFT on GPU.
  2. JAX arrays are immutable, so the Thomas forward/backward sweeps
     accumulate into a Python list and jnp.stack once at the end, instead
     of writing into a preallocated array slice-by-slice as
     projection.py's 1D thomas_batch_precomp/solve (or MATLAB's D_mod(k,:)
     = ...) do. Still O(nt) *Python-level* loop iterations, unrelated to
     grid resolution, fully vectorized over (nx, ny) each iteration -- a
     jax.lax.scan version that traces once instead of unrolling nt Python
     steps is a natural follow-up once this is otherwise correct and
     benchmarked (see the module-level GPU/memory notes below).
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .state_2d import State2D


def _dct_axis(f: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Orthonormal DCT-II along one axis of an n-D array, via FFT + mirroring.

    xe = mirror(f, axis)                    -- symmetric extension, length 2N
    V  = fft(xe, axis)                      -- 2N-point FFT along `axis`
    raw = real(V[..., :N, ...] * tw)        -- tw = exp(-i*pi*k/(2N)), k=0..N-1
    out = raw * (w/2)                       -- orthonormal scaling

    Matches scipy.fft.dct(..., type=2, norm='ortho') along `axis` (checked
    numerically in scratchpad against scipy for random 1D/2D/3D arrays).
    """
    n = f.shape[axis]
    xe = jnp.concatenate([f, jnp.flip(f, axis=axis)], axis=axis)
    V = jnp.fft.fft(xe, axis=axis)

    shape = [1] * f.ndim
    shape[axis] = n
    k = jnp.arange(n)
    tw = jnp.exp(-1j * jnp.pi * k / (2 * n)).reshape(shape)
    w = jnp.concatenate([jnp.ones(1) / jnp.sqrt(n), jnp.sqrt(2.0 / n) * jnp.ones(n - 1)]).reshape(shape)

    V_trunc = jax.lax.slice_in_dim(V, 0, n, axis=axis)
    return jnp.real(V_trunc * tw) * (w / 2)


def _idct_axis(X: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Inverse of _dct_axis (orthonormal DCT-III) along one axis."""
    n = X.shape[axis]
    shape = [1] * X.ndim
    shape[axis] = n
    k = jnp.arange(n)
    w = jnp.concatenate([jnp.ones(1) / jnp.sqrt(n), jnp.sqrt(2.0 / n) * jnp.ones(n - 1)]).reshape(shape)
    tw = jnp.exp(1j * jnp.pi * k / (2 * n)).reshape(shape)

    U = (X * w).astype(complex) * tw
    pad_shape = list(X.shape)
    pad_shape[axis] = n
    xe = jnp.concatenate([U, jnp.zeros(pad_shape, dtype=U.dtype)], axis=axis)
    x = jnp.real(jnp.fft.ifft(xe, axis=axis)) * (2 * n)
    return jax.lax.slice_in_dim(x, 0, n, axis=axis)


def dct2_xy(f: jnp.ndarray) -> jnp.ndarray:
    """2D DCT-II along the x axis (1) then y axis (2) of an (nt, nx, ny) array."""
    return _dct_axis(_dct_axis(f, axis=1), axis=2)


def idct2_xy(f_hat: jnp.ndarray) -> jnp.ndarray:
    """Inverse of dct2_xy."""
    return _idct_axis(_idct_axis(f_hat, axis=2), axis=1)


def thomas_batch_precomp(D: jnp.ndarray, e: jnp.ndarray) -> jnp.ndarray:
    """Forward sweep for K tridiagonal systems with constant off-diagonal e.

    D: (nt, ...) diagonals. e: (...,) constant off-diagonal, broadcastable
    against D's trailing dims (here (nx, ny)). Returns D_mod: (nt, ...)
    such that thomas_batch_solve(D_mod, e, F) solves every system at once.
    """
    nt = D.shape[0]
    rows = [D[0]]
    for t in range(1, nt):
        m = e / rows[-1]
        rows.append(D[t] - m * e)
    return jnp.stack(rows, axis=0)


def thomas_batch_solve(D_mod: jnp.ndarray, e: jnp.ndarray, F: jnp.ndarray) -> jnp.ndarray:
    """Back-substitute K tridiagonal systems precomputed by thomas_batch_precomp."""
    nt = F.shape[0]

    f_rows = [F[0]]
    for t in range(1, nt):
        m = e / D_mod[t - 1]
        f_rows.append(F[t] - m * f_rows[-1])

    x_rows = [f_rows[-1] / D_mod[-1]]
    for t in range(nt - 2, -1, -1):
        x_rows.append((f_rows[t] - e * x_rows[-1]) / D_mod[t])
    x_rows.reverse()

    return jnp.stack(x_rows, axis=0)


@dataclass
class BandedProj2D:
    D_mod: jnp.ndarray  # (nt, nx, ny) Thomas-modified diagonal, all modes
    e: jnp.ndarray       # (nx, ny)     constant off-diagonal per mode


def precomp_banded_proj_2d(problem, vareps: float) -> BandedProj2D:
    """Precompute the Thomas forward sweep for all (kx,ky) modes except the DC one.

    Per-mode system T_{kx,ky} = M0 + lxy*diag(M1d) + lxy^2*M2, where
    lxy = lambda_x[kx] + lambda_y[ky] is the only place x and y couple --
    M0, M1d, M2 are the same time-direction building blocks as the 1D
    case (see projection.py's precomp_banded_proj), so the closed-form
    entries carry over with lx -> lxy:

      diagonal:
        d[0]    = 1/dt^2 + lxy*(1+eps/dt) + lxy^2*eps^2/4
        d[1:-1] = 2/dt^2 + lxy            + lxy^2*eps^2/2
        d[-1]   = 1/dt^2 + lxy*(1-eps/dt) + lxy^2*eps^2/4
      off-diagonal (constant per mode):
        e = -1/dt^2 + lxy^2*eps^2/4

    The (kx,ky)=(0,0) DC mode has lxy=0 and a singular T; its diagonal is
    set to 1 (off-diagonal 0) so a zeroed RHS there solves to zero and
    proj_fokker_planck_banded_2d overwrites that mode via 1D DCT-in-t,
    exactly as in the 1D banded projection and matlab/shared/2d/utils/
    precomp_banded_proj.m.
    """
    nt, dt = problem.nt, problem.dt
    lx, ly = problem.lambda_x, problem.lambda_y  # (nx,), (ny,); lx[0]=ly[0]=0
    lxy = lx[:, None] + ly[None, :]              # (nx, ny), lxy[0, 0] = 0

    mid = (2 / dt**2 + lxy + lxy**2 * vareps**2 / 2)[None, :, :]
    top = (1 / dt**2 + lxy * (1 + vareps / dt) + lxy**2 * vareps**2 / 4)[None, :, :]
    bot = (1 / dt**2 + lxy * (1 - vareps / dt) + lxy**2 * vareps**2 / 4)[None, :, :]
    D = jnp.concatenate([top, jnp.broadcast_to(mid, (nt - 2, *lxy.shape)), bot], axis=0)

    e = -1 / dt**2 + lxy**2 * vareps**2 / 4  # (nx, ny)

    D = D.at[:, 0, 0].set(1.0)
    e = e.at[0, 0].set(0.0)

    D_mod = thomas_batch_precomp(D, e)
    return BandedProj2D(D_mod=D_mod, e=e)


def proj_fokker_planck_banded_2d(x_in: State2D, problem, vareps: float, bp: BandedProj2D) -> State2D:
    """Project (rho, mx, my) onto the FP constraint

        d_t rho + d_x mx + d_y my = eps * (d_xx + d_yy) rho

    BCs: rho(0,.,.)=rho0, rho(1,.,.)=rho1, mx=my=0 on the domain walls
    (baked into `problem.ops`).
    """
    ops = problem.ops
    rho0, rho1 = problem.rho0, problem.rho1
    nt, nx, ny = problem.nt, problem.nx, problem.ny

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

    # No early-exit near-zero-residual check: matlab's proj_fokker_planck_
    # banded.m has one ("This can probably be removed"), but its own *_gpu
    # sibling drops it with the comment "gather() would force GPU->CPU
    # sync every iteration" -- adopting that already-settled call here
    # rather than reintroducing the sync it was removed to avoid.

    # --- 2D DCT in x and y ---
    f_hat = dct2_xy(f)  # (nt, nx, ny)

    # --- Batched Thomas solve; zero RHS at the (kx,ky)=(0,0) DC mode ---
    rhs = f_hat.at[:, 0, 0].set(0.0)
    phi_hat = thomas_batch_solve(bp.D_mod, bp.e, rhs)

    # --- DC mode: lambda_xy=0 -> T singular, invert via 1D DCT-in-t ---
    f1_t = _dct_axis(f_hat[:, 0, 0], axis=0)
    lambda_t = problem.lambda_t
    phi1_t = jnp.concatenate([jnp.zeros(1), f1_t[1:] / lambda_t[1:]])
    phi_hat = phi_hat.at[:, 0, 0].set(_idct_axis(phi1_t, axis=0))

    # --- 2D IDCT back to physical space ---
    phi = idct2_xy(phi_hat)

    # --- Apply A* corrections ---
    dphi_dx = ops.deriv_x_at_m(phi)
    dphi_dy = ops.deriv_y_at_m(phi)
    nabla_phi = ops.interp_t_at_rho(
        ops.deriv_x_at_phi(dphi_dx, zeros_x, zeros_x)
        + ops.deriv_y_at_phi(dphi_dy, zeros_y, zeros_y)
    )

    rho_out = mu + ops.deriv_t_at_rho(phi) + vareps * nabla_phi
    mx_out = psi_x + dphi_dx
    my_out = psi_y + dphi_dy

    return State2D(rho_out, mx_out, my_out)
