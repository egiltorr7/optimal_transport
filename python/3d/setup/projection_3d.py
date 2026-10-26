"""Projection onto the 3D Fokker-Planck constraint set via DCT-in-(x,y,z) +
per-mode solve in t.

3D analogue of projection_2d.py, GPU version. Under a DCT in every spatial
axis the FP operator decouples into nx*ny*nz independent tridiagonal-in-time
systems indexed by the combined eigenvalue

    lambda_xyz = lambda_x[kx] + lambda_y[ky] + lambda_z[kz]

-- each per-mode system is exactly the 1D one with lambda_x[k] replaced by
lambda_xyz; only the number of terms in that sum changes from 2D.

What changed from the first (eager) 3D port, mirroring 2D:

  1. Thomas sweeps are jax.lax.scan loops (one traced step run nt times)
     instead of Python lists stacked at the end (3*nt separate dispatches).
  2. A second per-mode solver, method="dct": every per-mode matrix is a
     Neumann Laplacian in t plus two end-row deviations, so T^{-1} is a
     DCT-in-t division plus a rank-2 Woodbury correction with a precomputed
     2x2 capacitance matrix per mode. Fully parallel over all entries; the
     DC mode (0,0,0) needs no special case (a0 = a1 = 0, 1/mu_0 := 0).
     2-3x over the scan-Thomas on the GPU at 2D sizes.
  3. Three DCT kernels behind set_dct_method: "fft2n" (mirror + 2N FFT),
     "fftn" (Makhoul N-point), "matmul" (dense cosine matrix, one tensordot).
     matmul is the fastest on the GH200 for N <= 512 (XLA's FFT along a
     non-last axis inserts transposes; the GEMM is one cuBLAS call). In
     float32 pin jax_default_matmul_precision to "highest".
  4. The precompute dataclasses are pytrees, so they pass into the compiled
     LADMM chunk as arguments (ladmm_jit.py) rather than baked-in constants.

All DCT / Thomas / DctModes code is rank-agnostic (trailing dims are
whatever the spatial shape is); it is copied rather than imported from
2d/setup because both trees are top-level packages named `setup`.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import lax

from .state_3d import State3D

# ---------------------------------------------------------------------------
# DCT-II / DCT-III along one axis: three interchangeable implementations,
# all matching scipy.fft.dct(type=2, norm='ortho').
# ---------------------------------------------------------------------------
_DCT_METHOD = "matmul"


def set_dct_method(method: str) -> None:
    global _DCT_METHOD
    if method not in ("fft2n", "fftn", "matmul"):
        raise ValueError(f"unknown DCT method {method!r}")
    _DCT_METHOD = method


def get_dct_method() -> str:
    return _DCT_METHOD


def _ortho_weights(n: int, shape) -> jnp.ndarray:
    return jnp.concatenate([jnp.ones(1) / jnp.sqrt(n), jnp.sqrt(2.0 / n) * jnp.ones(n - 1)]).reshape(shape)


def _dct_axis_fft2n(f: jnp.ndarray, axis: int) -> jnp.ndarray:
    n = f.shape[axis]
    xe = jnp.concatenate([f, jnp.flip(f, axis=axis)], axis=axis)
    V = jnp.fft.fft(xe, axis=axis)
    shape = [1] * f.ndim
    shape[axis] = n
    k = jnp.arange(n)
    tw = jnp.exp(-1j * jnp.pi * k / (2 * n)).reshape(shape)
    w = _ortho_weights(n, shape)
    V_trunc = lax.slice_in_dim(V, 0, n, axis=axis)
    return jnp.real(V_trunc * tw) * (w / 2)


def _idct_axis_fft2n(X: jnp.ndarray, axis: int) -> jnp.ndarray:
    n = X.shape[axis]
    shape = [1] * X.ndim
    shape[axis] = n
    k = jnp.arange(n)
    w = _ortho_weights(n, shape)
    tw = jnp.exp(1j * jnp.pi * k / (2 * n)).reshape(shape)
    U = (X * w).astype(complex) * tw
    pad_shape = list(X.shape)
    pad_shape[axis] = n
    xe = jnp.concatenate([U, jnp.zeros(pad_shape, dtype=U.dtype)], axis=axis)
    x = jnp.real(jnp.fft.ifft(xe, axis=axis)) * (2 * n)
    return lax.slice_in_dim(x, 0, n, axis=axis)


def _dct_axis_fftn(f: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Makhoul (1980): v = [x_0, x_2, ..., x_3, x_1], X_k = Re(exp(-i pi k/2N) FFT(v)_k)."""
    n = f.shape[axis]
    ev = lax.slice_in_dim(f, 0, n, stride=2, axis=axis)
    od = lax.slice_in_dim(f, 1, n, stride=2, axis=axis)
    v = jnp.concatenate([ev, jnp.flip(od, axis=axis)], axis=axis)
    V = jnp.fft.fft(v, axis=axis)
    shape = [1] * f.ndim
    shape[axis] = n
    k = jnp.arange(n)
    tw = jnp.exp(-1j * jnp.pi * k / (2 * n)).reshape(shape)
    w = _ortho_weights(n, shape)
    return jnp.real(V * tw) * (w / 2) * 2


def _put_along_axis(out, idx, vals, axis):
    sl = [slice(None)] * out.ndim
    sl[axis] = idx
    return out.at[tuple(sl)].set(vals)


def _idct_axis_fftn(X: jnp.ndarray, axis: int) -> jnp.ndarray:
    n = X.shape[axis]
    shape = [1] * X.ndim
    shape[axis] = n
    k = jnp.arange(n)
    w = _ortho_weights(n, shape)
    tw = jnp.exp(1j * jnp.pi * k / (2 * n)).reshape(shape)
    Y = X / w
    Yr = jnp.flip(lax.slice_in_dim(Y, 1, n, axis=axis), axis=axis)
    pad_shape = list(X.shape)
    pad_shape[axis] = 1
    Y_rev = jnp.concatenate([jnp.zeros(pad_shape, dtype=Y.dtype), Yr], axis=axis)
    V = tw * (Y - 1j * Y_rev)
    v = jnp.real(jnp.fft.ifft(V, axis=axis))
    n_ev = (n + 1) // 2
    ev = lax.slice_in_dim(v, 0, n_ev, axis=axis)
    od = jnp.flip(lax.slice_in_dim(v, n_ev, n, axis=axis), axis=axis)
    out = jnp.zeros(list(X.shape), dtype=v.dtype)
    out = _put_along_axis(out, jnp.arange(0, n, 2), ev, axis)
    out = _put_along_axis(out, jnp.arange(1, n, 2), od, axis)
    return out


def _dct_matrix(n: int) -> jnp.ndarray:
    """(n, n) orthonormal DCT-II matrix C, X = C @ x."""
    k = jnp.arange(n)[:, None]
    j = jnp.arange(n)[None, :]
    C = jnp.cos(jnp.pi * (2 * j + 1) * k / (2 * n))
    return C * _ortho_weights(n, (n, 1))


def _apply_matrix_axis(C: jnp.ndarray, f: jnp.ndarray, axis: int) -> jnp.ndarray:
    out = jnp.tensordot(C.astype(f.dtype), f, axes=([1], [axis]))   # new axis lands in front
    return jnp.moveaxis(out, 0, axis)


def _dct_axis_matmul(f, axis):
    return _apply_matrix_axis(_dct_matrix(f.shape[axis]), f, axis)


def _idct_axis_matmul(X, axis):
    return _apply_matrix_axis(_dct_matrix(X.shape[axis]).T, X, axis)


def _dct_axis(f: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Orthonormal DCT-II along one axis of an n-D array (see set_dct_method)."""
    if _DCT_METHOD == "fft2n":
        return _dct_axis_fft2n(f, axis)
    if _DCT_METHOD == "fftn":
        return _dct_axis_fftn(f, axis)
    return _dct_axis_matmul(f, axis)


def _idct_axis(X: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Inverse of _dct_axis (orthonormal DCT-III) along one axis."""
    if _DCT_METHOD == "fft2n":
        return _idct_axis_fft2n(X, axis)
    if _DCT_METHOD == "fftn":
        return _idct_axis_fftn(X, axis)
    return _idct_axis_matmul(X, axis)


def dct3_xyz(f: jnp.ndarray) -> jnp.ndarray:
    """3D DCT-II along the x (1), y (2) and z (3) axes of an (nt,nx,ny,nz) array."""
    return _dct_axis(_dct_axis(_dct_axis(f, axis=1), axis=2), axis=3)


def idct3_xyz(f_hat: jnp.ndarray) -> jnp.ndarray:
    """Inverse of dct3_xyz."""
    return _idct_axis(_idct_axis(_idct_axis(f_hat, axis=3), axis=2), axis=1)


# ---------------------------------------------------------------------------
# Batched Thomas (constant off-diagonal), as lax.scan loops.
# ---------------------------------------------------------------------------
def thomas_batch_precomp(D: jnp.ndarray, e: jnp.ndarray) -> jnp.ndarray:
    """Forward sweep for K tridiagonal systems with constant off-diagonal e.
    D: (nt, ...) diagonals; e: (...,) broadcastable against D's trailing dims."""
    def body(prev, Dt):
        new = Dt - (e / prev) * e
        return new, new

    _, rest = lax.scan(body, D[0], D[1:])
    return jnp.concatenate([D[:1], rest], axis=0)


def thomas_batch_solve(D_mod: jnp.ndarray, e: jnp.ndarray, F: jnp.ndarray) -> jnp.ndarray:
    """Back-substitute K tridiagonal systems precomputed by thomas_batch_precomp."""
    def fwd(prev, inp):
        Dm_prev, Ft = inp
        new = Ft - (e / Dm_prev) * prev
        return new, new

    _, f_rest = lax.scan(fwd, F[0], (D_mod[:-1], F[1:]))
    f = jnp.concatenate([F[:1], f_rest], axis=0)

    def bwd(nxt, inp):
        ft, Dm = inp
        new = (ft - e * nxt) / Dm
        return new, new

    x_last = f[-1] / D_mod[-1]
    _, x_rest = lax.scan(bwd, x_last, (f[:-1], D_mod[:-1]), reverse=True)
    return jnp.concatenate([x_rest, x_last[None]], axis=0)


# ---------------------------------------------------------------------------
# Per-mode time systems: constant interior + two end deviations, and the
# fully parallel DCT-in-t solver (rank-agnostic in the spatial dims).
# ---------------------------------------------------------------------------
@dataclass
class DctModes:
    """T0^{-1} via DCT-in-t plus the rank-2 end correction.

    inv_mu:  (nt, *sp)  1/mu_k per mode, 0 where mu_k == 0 (the DC mode)
    g0, g1:  (nt, *sp)  T0^{-1} e_0, T0^{-1} e_{nt-1}
    ca0..cb1: (*sp)     rows of C (I + G C)^{-1}: (y_0, y_{nt-1}) -> (w0, w1)
    """
    inv_mu: jnp.ndarray
    g0: jnp.ndarray
    g1: jnp.ndarray
    ca0: jnp.ndarray
    ca1: jnp.ndarray
    cb0: jnp.ndarray
    cb1: jnp.ndarray


jax.tree_util.register_dataclass(
    DctModes, data_fields=["inv_mu", "g0", "g1", "ca0", "ca1", "cb0", "cb1"], meta_fields=[])


def _dct_t(v):
    return _dct_axis(v, axis=0)


def _idct_t(v):
    return _idct_axis(v, axis=0)


def _dc_index(sp_ndim: int):
    return (slice(None),) + (0,) * sp_ndim


def precomp_dct_modes(nt: int, top, mid, bot, e, lambda_t=None) -> DctModes:
    """top/mid/bot/e: (*sp) per-mode tridiagonal entries (first row, interior
    rows, last row diagonals; constant off-diagonal)."""
    sp = mid.shape
    k = jnp.arange(nt).reshape((nt,) + (1,) * len(sp))
    mu = mid[None] + 2.0 * e[None] * jnp.cos(jnp.pi * k / nt)      # (nt, *sp)
    inv_mu = jnp.where(mu == 0.0, 0.0, 1.0 / jnp.where(mu == 0.0, 1.0, mu))
    if lambda_t is not None:
        # At the DC mode mu_k == lambda_t[k] analytically; use lambda_t's
        # closed form there so the DC inverse matches the Thomas path's
        # DCT-in-t pseudo-inverse to the last bit.
        inv_lt = jnp.concatenate([jnp.zeros(1), 1.0 / lambda_t[1:]])
        inv_mu = inv_mu.at[_dc_index(len(sp))].set(inv_lt)

    a0 = top - (mid + e)                                             # (*sp)
    a1 = bot - (mid + e)

    def T0_inv(rhs):
        return _idct_t(_dct_t(rhs) * inv_mu)

    unit_shape = (nt,) + (1,) * len(sp)
    e0 = jnp.zeros(unit_shape).at[0].set(1.0)
    e1 = jnp.zeros(unit_shape).at[nt - 1].set(1.0)
    g0 = T0_inv(jnp.broadcast_to(e0, (nt, *sp)))
    g1 = T0_inv(jnp.broadcast_to(e1, (nt, *sp)))
    G00, G01, G10, G11 = g0[0], g1[0], g0[-1], g1[-1]
    K00 = 1.0 + G00 * a0
    K01 = G01 * a1
    K10 = G10 * a0
    K11 = 1.0 + G11 * a1
    det = K00 * K11 - K01 * K10
    Ki00, Ki01, Ki10, Ki11 = K11 / det, -K01 / det, -K10 / det, K00 / det
    ca0, ca1 = a0 * Ki00, a0 * Ki01     # w0 = ca0*v0 + ca1*v1
    cb0, cb1 = a1 * Ki10, a1 * Ki11     # w1 = cb0*v0 + cb1*v1
    return DctModes(inv_mu=inv_mu, g0=g0, g1=g1, ca0=ca0, ca1=ca1, cb0=cb0, cb1=cb1)


def solve_dct_modes(dm: DctModes, rhs: jnp.ndarray) -> jnp.ndarray:
    """T^{-1} rhs for every mode at once, rhs: (nt, *sp) in DCT-space."""
    y = _idct_t(_dct_t(rhs) * dm.inv_mu)
    v0, v1 = y[0], y[-1]
    w0 = dm.ca0 * v0 + dm.ca1 * v1
    w1 = dm.cb0 * v0 + dm.cb1 * v1
    return y - dm.g0 * w0[None] - dm.g1 * w1[None]


def _stack_diag(nt, top, mid, bot):
    return jnp.concatenate([top[None], jnp.broadcast_to(mid[None], (nt - 2, *mid.shape)), bot[None]], axis=0)


def solve_time_modes(bp, f_hat: jnp.ndarray, lambda_t: jnp.ndarray) -> jnp.ndarray:
    """phi_hat = T^{-1} f_hat for every mode, including the DC mode.

    Works for any precomp object carrying (D_mod, e, dct): BandedProj3D and
    projection_expsemi_3d.ExpsemiProj3D.
    """
    if bp.dct is not None:
        return solve_dct_modes(bp.dct, f_hat)
    dc = _dc_index(f_hat.ndim - 1)
    rhs = f_hat.at[dc].set(0.0)
    phi_hat = thomas_batch_solve(bp.D_mod, bp.e, rhs)
    f1_t = _dct_axis(f_hat[dc], axis=0)
    phi1_t = jnp.concatenate([jnp.zeros(1), f1_t[1:] / lambda_t[1:]])
    return phi_hat.at[dc].set(_idct_axis(phi1_t, axis=0))


# ---------------------------------------------------------------------------
# CN / banded projection
# ---------------------------------------------------------------------------
@dataclass
class BandedProj3D:
    D_mod: jnp.ndarray  # (nt, nx, ny, nz) Thomas-modified diagonal, all modes
    e: jnp.ndarray      # (nx, ny, nz)     constant off-diagonal per mode
    dct: DctModes | None = None   # set when method == "dct"


jax.tree_util.register_dataclass(BandedProj3D, data_fields=["D_mod", "e", "dct"], meta_fields=[])


def lambda_xyz(problem) -> jnp.ndarray:
    lx, ly, lz = problem.lambda_x, problem.lambda_y, problem.lambda_z
    return lx[:, None, None] + ly[None, :, None] + lz[None, None, :]   # (nx,ny,nz), [0,0,0] = 0


def _banded_entries_3d(problem, vareps: float):
    dt = problem.dt
    lxyz = lambda_xyz(problem)
    mid = 2 / dt**2 + lxyz + lxyz**2 * vareps**2 / 2
    top = 1 / dt**2 + lxyz * (1 + vareps / dt) + lxyz**2 * vareps**2 / 4
    bot = 1 / dt**2 + lxyz * (1 - vareps / dt) + lxyz**2 * vareps**2 / 4
    e = -1 / dt**2 + lxyz**2 * vareps**2 / 4
    return top, mid, bot, e


def precomp_banded_proj_3d(problem, vareps: float, method: str = "thomas") -> BandedProj3D:
    """Precompute the per-mode time systems for all (kx,ky,kz) modes.

    Per-mode entries are the 1D closed forms with lx -> lxyz:
      d[0]    = 1/dt^2 + lxyz*(1+eps/dt) + lxyz^2*eps^2/4
      d[1:-1] = 2/dt^2 + lxyz            + lxyz^2*eps^2/2
      d[-1]   = 1/dt^2 + lxyz*(1-eps/dt) + lxyz^2*eps^2/4
      e       = -1/dt^2 + lxyz^2*eps^2/4
    method="thomas": the DC mode (singular) is neutralised (diag 1, off 0),
    zero RHS, and overwritten via DCT-in-t in solve_time_modes.
    method="dct": the parallel DCT-in-t solver (module docstring).
    """
    if method not in ("thomas", "dct"):
        raise ValueError(f"unknown method {method!r}")
    nt = problem.nt
    top, mid, bot, e = _banded_entries_3d(problem, vareps)
    D = _stack_diag(nt, top, mid, bot)
    D = D.at[:, 0, 0, 0].set(1.0)
    e_th = e.at[0, 0, 0].set(0.0)
    D_mod = thomas_batch_precomp(D, e_th)
    dct = precomp_dct_modes(nt, top, mid, bot, e, problem.lambda_t) if method == "dct" else None
    return BandedProj3D(D_mod=D_mod, e=e_th, dct=dct)


def proj_fokker_planck_banded_3d(x_in: State3D, problem, vareps: float, bp: BandedProj3D) -> State3D:
    """Project (rho, mx, my, mz) onto the FP constraint

        d_t rho + d_x mx + d_y my + d_z mz = eps * (d_xx + d_yy + d_zz) rho

    BCs: rho(0,...)=rho0, rho(1,...)=rho1, mx=my=mz=0 on the walls (in problem.ops).
    """
    ops = problem.ops
    rho0, rho1 = problem.rho0, problem.rho1
    nt, nx, ny, nz = problem.nt, problem.nx, problem.ny, problem.nz
    dtype = x_in.rho.dtype

    mu, psi_x, psi_y, psi_z = x_in.rho, x_in.mx, x_in.my, x_in.mz
    zeros_x = jnp.zeros((nt, ny, nz), dtype)
    zeros_y = jnp.zeros((nt, nx, nz), dtype)
    zeros_z = jnp.zeros((nt, nx, ny), dtype)

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
    # No early-exit residual check: it would force a device->host sync per iteration.

    f_hat = dct3_xyz(f)
    phi_hat = solve_time_modes(bp, f_hat, problem.lambda_t)
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

    return State3D(mu + ops.deriv_t_at_rho(phi) + vareps * nabla_phi,
                   psi_x + dphi_dx, psi_y + dphi_dy, psi_z + dphi_dz)
