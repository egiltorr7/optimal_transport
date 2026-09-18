"""Projection onto the 3D Fokker-Planck constraint set via DCT-in-(x,y,z) +
batched tridiagonal (Thomas) solve in t.

3D analogue of projection_2d.py. The structure is unchanged, and so is the
reason it works: under a DCT in every spatial axis the FP operator decouples
into nx*ny*nz independent tridiagonal-in-time systems, indexed by the
*combined* eigenvalue

    lambda_xyz = lambda_x[kx] + lambda_y[ky] + lambda_z[kz]

-- and each per-mode system (and its closed-form diagonal/off-diagonal
entries) is then exactly the 1D one with lambda_x[k] replaced by
lambda_xyz. That the discrete Laplacian's spectrum is additive across axes
is the whole reason this generalizes to any dimension for free; nothing but
the number of terms in that sum changes from 2D.

The two JAX-specific choices carry over verbatim from projection_2d.py:

  1. No dct/idct exists in jax.numpy or jax.scipy, so the DCT-II is the
     mirror-and-FFT trick, applied along an `axis=` argument. _dct_axis /
     _idct_axis are projection_2d.py's verbatim -- they are already
     rank-agnostic (twiddle and weight vectors are built from f.ndim), so
     the 4-D (nt,nx,ny,nz) arrays here need no new code. Copied rather than
     imported because 2d/setup and 3d/setup are both top-level packages
     named `setup`, so a cross-directory import collides in sys.modules.
  2. JAX arrays are immutable, so the Thomas forward/backward sweeps
     accumulate into a Python list and jnp.stack once at the end. Still
     O(nt) *Python-level* iterations, unrelated to spatial resolution, fully
     vectorized over (nx, ny, nz) each iteration.

Memory, which is the one genuinely new constraint in 3D: every array here is
(nt, nx, ny, nz), and the list-based Thomas sweeps hold a full such array
per sweep (three of them live at the peak of thomas_batch_solve: D_mod,
f_rows, x_rows), on top of f/f_hat/phi and the transient length-2N mirror
that _dct_axis builds along whichever axis it is working on. Budget roughly
8*nt*nx*ny*nz bytes per float64 array and ~8 live arrays: 32^4 is ~8 MB per
array (fine anywhere), 64^4 is ~134 MB per array (~1 GB peak, fits a GPU),
96^4 is ~680 MB per array and will not. Beyond that the list-based sweeps
want replacing with jax.lax.scan (which traces once instead of unrolling nt
Python steps and lets XLA reuse buffers) and/or float32 via
jax_config.configure(x64=False) -- both deliberately left out of this first
port, see the module notes in projection_2d.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .state_3d import State3D


def _dct_axis(f: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Orthonormal DCT-II along one axis of an n-D array, via FFT + mirroring.

    projection_2d.py's, verbatim and unchanged -- already rank-agnostic, so
    it serves the 4-D arrays here as-is. Matches
    scipy.fft.dct(..., type=2, norm='ortho') along `axis`.
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


def dct3_xyz(f: jnp.ndarray) -> jnp.ndarray:
    """3D DCT-II along the x (1), y (2) and z (3) axes of an (nt,nx,ny,nz) array."""
    return _dct_axis(_dct_axis(_dct_axis(f, axis=1), axis=2), axis=3)


def idct3_xyz(f_hat: jnp.ndarray) -> jnp.ndarray:
    """Inverse of dct3_xyz."""
    return _idct_axis(_idct_axis(_idct_axis(f_hat, axis=3), axis=2), axis=1)


def thomas_batch_precomp(D: jnp.ndarray, e: jnp.ndarray) -> jnp.ndarray:
    """Forward sweep for K tridiagonal systems with constant off-diagonal e.

    D: (nt, ...) diagonals. e: (...,) constant off-diagonal, broadcastable
    against D's trailing dims (here (nx, ny, nz)). Returns D_mod: (nt, ...)
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
class BandedProj3D:
    D_mod: jnp.ndarray  # (nt, nx, ny, nz) Thomas-modified diagonal, all modes
    e: jnp.ndarray      # (nx, ny, nz)     constant off-diagonal per mode


def precomp_banded_proj_3d(problem, vareps: float) -> BandedProj3D:
    """Precompute the Thomas forward sweep for all (kx,ky,kz) modes but the DC one.

    Per-mode system T = M0 + lxyz*diag(M1d) + lxyz^2*M2, where lxyz is the
    only place the three spatial axes couple -- M0, M1d, M2 are the same
    time-direction building blocks as the 1D case, so the closed-form entries
    carry over with lx -> lxyz:

      diagonal:
        d[0]    = 1/dt^2 + lxyz*(1+eps/dt) + lxyz^2*eps^2/4
        d[1:-1] = 2/dt^2 + lxyz            + lxyz^2*eps^2/2
        d[-1]   = 1/dt^2 + lxyz*(1-eps/dt) + lxyz^2*eps^2/4
      off-diagonal (constant per mode):
        e = -1/dt^2 + lxyz^2*eps^2/4

    The (kx,ky,kz)=(0,0,0) DC mode has lxyz=0 and a singular T; its diagonal
    is set to 1 (off-diagonal 0) so a zeroed RHS there solves to zero, and
    proj_fokker_planck_banded_3d overwrites that mode via a 1D DCT-in-t --
    exactly as in the 1D and 2D banded projections.
    """
    nt, dt = problem.nt, problem.dt
    lx, ly, lz = problem.lambda_x, problem.lambda_y, problem.lambda_z
    # (nx, ny, nz); lxyz[0, 0, 0] = 0 since lx[0] = ly[0] = lz[0] = 0
    lxyz = lx[:, None, None] + ly[None, :, None] + lz[None, None, :]

    mid = (2 / dt**2 + lxyz + lxyz**2 * vareps**2 / 2)[None, ...]
    top = (1 / dt**2 + lxyz * (1 + vareps / dt) + lxyz**2 * vareps**2 / 4)[None, ...]
    bot = (1 / dt**2 + lxyz * (1 - vareps / dt) + lxyz**2 * vareps**2 / 4)[None, ...]
    D = jnp.concatenate([top, jnp.broadcast_to(mid, (nt - 2, *lxyz.shape)), bot], axis=0)

    e = -1 / dt**2 + lxyz**2 * vareps**2 / 4  # (nx, ny, nz)

    D = D.at[:, 0, 0, 0].set(1.0)
    e = e.at[0, 0, 0].set(0.0)

    D_mod = thomas_batch_precomp(D, e)
    return BandedProj3D(D_mod=D_mod, e=e)


def proj_fokker_planck_banded_3d(x_in: State3D, problem, vareps: float, bp: BandedProj3D) -> State3D:
    """Project (rho, mx, my, mz) onto the FP constraint

        d_t rho + d_x mx + d_y my + d_z mz = eps * (d_xx + d_yy + d_zz) rho

    BCs: rho(0,...)=rho0, rho(1,...)=rho1, mx=my=mz=0 on the domain walls
    (baked into `problem.ops`).
    """
    ops = problem.ops
    rho0, rho1 = problem.rho0, problem.rho1
    nt, nx, ny, nz = problem.nt, problem.nx, problem.ny, problem.nz

    mu, psi_x, psi_y, psi_z = x_in.rho, x_in.mx, x_in.my, x_in.mz
    zeros_x = jnp.zeros((nt, ny, nz))
    zeros_y = jnp.zeros((nt, nx, nz))
    zeros_z = jnp.zeros((nt, nx, ny))

    # --- FP residual  f = d_t mu + div psi - eps*Delta mu ---
    mu_phi = ops.interp_t_at_phi(mu, rho0, rho1)
    laplacian_mu = (
        ops.deriv_x_at_phi(ops.deriv_x_at_m(mu_phi), zeros_x, zeros_x)
        + ops.deriv_y_at_phi(ops.deriv_y_at_m(mu_phi), zeros_y, zeros_y)
        + ops.deriv_z_at_phi(ops.deriv_z_at_m(mu_phi), zeros_z, zeros_z)
    )
    f = (
        ops.deriv_t_at_phi(mu, rho0, rho1)
        + ops.deriv_x_at_phi(psi_x, zeros_x, zeros_x)
        + ops.deriv_y_at_phi(psi_y, zeros_y, zeros_y)
        + ops.deriv_z_at_phi(psi_z, zeros_z, zeros_z)
        - vareps * laplacian_mu
    )

    # No early-exit near-zero-residual check, for projection_2d.py's reason
    # (it would force a device->host sync every iteration).

    # --- 3D DCT in x, y and z ---
    f_hat = dct3_xyz(f)  # (nt, nx, ny, nz)

    # --- Batched Thomas solve; zero RHS at the (kx,ky,kz)=(0,0,0) DC mode ---
    rhs = f_hat.at[:, 0, 0, 0].set(0.0)
    phi_hat = thomas_batch_solve(bp.D_mod, bp.e, rhs)

    # --- DC mode: lambda_xyz=0 -> T singular, invert via 1D DCT-in-t ---
    f1_t = _dct_axis(f_hat[:, 0, 0, 0], axis=0)
    lambda_t = problem.lambda_t
    phi1_t = jnp.concatenate([jnp.zeros(1), f1_t[1:] / lambda_t[1:]])
    phi_hat = phi_hat.at[:, 0, 0, 0].set(_idct_axis(phi1_t, axis=0))

    # --- 3D IDCT back to physical space ---
    phi = idct3_xyz(phi_hat)

    # --- Apply A* corrections ---
    dphi_dx = ops.deriv_x_at_m(phi)
    dphi_dy = ops.deriv_y_at_m(phi)
    dphi_dz = ops.deriv_z_at_m(phi)
    nabla_phi = ops.interp_t_at_rho(
        ops.deriv_x_at_phi(dphi_dx, zeros_x, zeros_x)
        + ops.deriv_y_at_phi(dphi_dy, zeros_y, zeros_y)
        + ops.deriv_z_at_phi(dphi_dz, zeros_z, zeros_z)
    )

    rho_out = mu + ops.deriv_t_at_rho(phi) + vareps * nabla_phi
    mx_out = psi_x + dphi_dx
    my_out = psi_y + dphi_dy
    mz_out = psi_z + dphi_dz

    return State3D(rho_out, mx_out, my_out, mz_out)
