"""Projection onto the 2D Fokker-Planck constraint set via DCT-in-(x,y) +
per-mode tridiagonal solve in t.

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
     implemented via FFTs (see `set_dct_method` for the three variants),
     applied along an `axis=` argument directly rather than
     proj_fokker_planck_banded.m's permute+reshape+dct2_xy, since
     jnp.fft.fft(..., axis=k) needs no permute to hit any axis and
     dispatches to cuFFT on GPU.
  2. JAX arrays are immutable, so the Thomas forward/backward sweeps are
     written as jax.lax.scan loops (one traced step, executed nt times)
     rather than writing into a preallocated array slice-by-slice as
     projection.py's 1D thomas_batch_precomp/solve (or MATLAB's D_mod(k,:)
     = ...) do. The earlier version accumulated a Python list and stacked
     it, which cost 3*nt separate op dispatches per projection.

Two per-mode solvers are available (precomp_banded_proj_2d's `method`):

  "thomas"  the batched Thomas sweep above: sequential in t, O(nt) tiny
            kernels per solve when the loop runs on a GPU.
  "dct"     fully parallel. Every per-mode matrix here has the form
                T = T0 + a0*e_0 e_0^T + a1*e_{nt-1} e_{nt-1}^T
            with T0 the constant-diagonal, constant-off-diagonal symmetric
            tridiagonal whose end entries equal mid+e -- i.e. a Neumann
            Laplacian in t, diagonalized by the DCT-II in t with eigenvalues
            mu_k = mid + 2e*cos(pi k/nt) -- and (a0, a1) the two end-row
            deviations. T^{-1} is then T0^{-1} plus a rank-2 Woodbury
            correction whose per-mode 2x2 capacitance matrix is precomputed,
            so a solve is one DCT-in-t, a division, one IDCT-in-t and an
            elementwise correction: no sequential dependency at all. The
            (0,0) DC mode needs no special case in this form: there
            a0 = a1 = 0 and mu_0 = 0, and zeroing 1/mu_0 IS the DCT-in-t
            pseudo-inverse projection_2d used for it all along.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import lax

from .state_2d import State2D

# ---------------------------------------------------------------------------
# DCT-II / DCT-III along one axis. Three interchangeable implementations,
# all matching scipy.fft.dct(type=2, norm='ortho'); pick with set_dct_method.
#
#   fft2n   mirror the input to length 2N and take a complex FFT
#           (matlab/shared/1d/gpu/dct_rows.m's trick). Simplest; 2N-point
#           complex FFT plus a 2N-length temporary.
#   fftn    Makhoul's even/odd reordering: N-point complex FFT, no padding.
#   matmul  dense N x N cosine matrix applied with einsum. One GEMM per
#           transform, no FFT plan, exact to ~1e-15 like the others; for
#           N <= a few hundred this is often the fastest on a GPU and the
#           most predictable across JAX/XLA versions.
# ---------------------------------------------------------------------------
_DCT_METHOD = "fft2n"


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
    """Makhoul (1980): v = [x_0, x_2, ..., x_{N-1 or N-2}, ..., x_3, x_1],
    X_k = Re(exp(-i pi k/2N) FFT(v)_k)."""
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


def _idct_axis_fftn(X: jnp.ndarray, axis: int) -> jnp.ndarray:
    n = X.shape[axis]
    shape = [1] * X.ndim
    shape[axis] = n
    k = jnp.arange(n)
    w = _ortho_weights(n, shape)
    tw = jnp.exp(1j * jnp.pi * k / (2 * n)).reshape(shape)
    Y = X / w                                       # undo the orthonormal scaling
    # V_k = exp(i pi k/2N) (Y_k - i Y_{N-k}),  Y_N := 0  (v real => V_{N-k} = conj V_k)
    Yr = jnp.flip(lax.slice_in_dim(Y, 1, n, axis=axis), axis=axis)     # Y_{N-1} .. Y_1
    pad_shape = list(X.shape)
    pad_shape[axis] = 1
    Y_rev = jnp.concatenate([jnp.zeros(pad_shape, dtype=Y.dtype), Yr], axis=axis)  # Y_{N-k}
    V = tw * (Y - 1j * Y_rev)
    v = jnp.real(jnp.fft.ifft(V, axis=axis))
    n_ev = (n + 1) // 2
    ev = lax.slice_in_dim(v, 0, n_ev, axis=axis)
    od = jnp.flip(lax.slice_in_dim(v, n_ev, n, axis=axis), axis=axis)
    # interleave: x[0::2] = ev, x[1::2] = od
    out_shape = list(X.shape)
    out = jnp.zeros(out_shape, dtype=v.dtype)
    idx_ev = jnp.arange(0, n, 2)
    idx_od = jnp.arange(1, n, 2)
    out = _put_along_axis(out, idx_ev, ev, axis)
    out = _put_along_axis(out, idx_od, od, axis)
    return out


def _put_along_axis(out, idx, vals, axis):
    sl = [slice(None)] * out.ndim
    sl[axis] = idx
    return out.at[tuple(sl)].set(vals)


def _dct_matrix(n: int) -> jnp.ndarray:
    """(n, n) orthonormal DCT-II matrix C, X = C @ x."""
    k = jnp.arange(n)[:, None]
    j = jnp.arange(n)[None, :]
    C = jnp.cos(jnp.pi * (2 * j + 1) * k / (2 * n))
    return C * _ortho_weights(n, (n, 1))


def _apply_matrix_axis(C: jnp.ndarray, f: jnp.ndarray, axis: int) -> jnp.ndarray:
    out = jnp.tensordot(C, f, axes=([1], [axis]))   # new axis lands in front
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


def dct2_xy(f: jnp.ndarray) -> jnp.ndarray:
    """2D DCT-II along the x axis (1) then y axis (2) of an (nt, nx, ny) array."""
    return _dct_axis(_dct_axis(f, axis=1), axis=2)


def idct2_xy(f_hat: jnp.ndarray) -> jnp.ndarray:
    """Inverse of dct2_xy."""
    return _idct_axis(_idct_axis(f_hat, axis=2), axis=1)


# ---------------------------------------------------------------------------
# Batched Thomas (constant off-diagonal), as lax.scan loops.
# ---------------------------------------------------------------------------
def thomas_batch_precomp(D: jnp.ndarray, e: jnp.ndarray) -> jnp.ndarray:
    """Forward sweep for K tridiagonal systems with constant off-diagonal e.

    D: (nt, ...) diagonals. e: (...,) constant off-diagonal, broadcastable
    against D's trailing dims (here (nx, ny)). Returns D_mod: (nt, ...)
    such that thomas_batch_solve(D_mod, e, F) solves every system at once.
    """
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
# Per-mode time systems: the shared "constant interior, two end deviations"
# structure, and its two solvers.
# ---------------------------------------------------------------------------
@dataclass
class DctModes:
    """T0^{-1} via DCT-in-t plus the rank-2 end correction (module docstring).

    inv_mu:  (nt, nx, ny)  1/mu_k per mode, 0 where mu_k == 0 (the DC mode)
    g0, g1:  (nt, nx, ny)  T0^{-1} e_0, T0^{-1} e_{nt-1}
    ca0..cb1: (nx, ny)     rows of C (I + G C)^{-1}: the 2x2 map from
                           (y_0, y_{nt-1}) to the correction weights (w0, w1)
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


def precomp_dct_modes(nt: int, top, mid, bot, e, lambda_t=None) -> DctModes:
    """top/mid/bot/e: (nx, ny) per-mode tridiagonal entries (first row,
    interior rows, last row diagonals; constant off-diagonal)."""
    k = jnp.arange(nt)[:, None, None]
    mu = mid[None] + 2.0 * e[None] * jnp.cos(jnp.pi * k / nt)      # (nt, nx, ny)
    # The only zero eigenvalue is k=0 of the DC mode (mid + 2e = 0 there).
    # Guard exact zeros; a mode that is merely tiny is inverted as-is.
    inv_mu = jnp.where(mu == 0.0, 0.0, 1.0 / jnp.where(mu == 0.0, 1.0, mu))
    if lambda_t is not None:
        # At the DC mode mu_k == lambda_t[k] analytically; use lambda_t's
        # closed form there so the DC inverse matches the reference
        # projection's DCT-in-t path to the last bit.
        inv_lt = jnp.concatenate([jnp.zeros(1), 1.0 / lambda_t[1:]])
        inv_mu = inv_mu.at[:, 0, 0].set(inv_lt)

    a0 = top - (mid + e)                                             # (nx, ny)
    a1 = bot - (mid + e)

    def T0_inv(rhs):
        return _idct_t(_dct_t(rhs) * inv_mu)

    nx, ny = mid.shape
    e0 = jnp.zeros((nt, 1, 1)).at[0].set(1.0)
    e1 = jnp.zeros((nt, 1, 1)).at[nt - 1].set(1.0)
    g0 = T0_inv(jnp.broadcast_to(e0, (nt, nx, ny)))
    g1 = T0_inv(jnp.broadcast_to(e1, (nt, nx, ny)))
    # G = V^T T0^{-1} U = [[g0[0], g1[0]], [g0[-1], g1[-1]]]
    G00, G01, G10, G11 = g0[0], g1[0], g0[-1], g1[-1]
    # K = I + G C,  C = diag(a0, a1);  w = C K^{-1} v,  v = (y_0, y_{nt-1})
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
    """T^{-1} rhs for every mode at once, rhs: (nt, nx, ny) in DCT-xy space."""
    y = _idct_t(_dct_t(rhs) * dm.inv_mu)
    v0, v1 = y[0], y[-1]
    w0 = dm.ca0 * v0 + dm.ca1 * v1
    w1 = dm.cb0 * v0 + dm.cb1 * v1
    return y - dm.g0 * w0[None] - dm.g1 * w1[None]


@dataclass
class BandedProj2D:
    D_mod: jnp.ndarray  # (nt, nx, ny) Thomas-modified diagonal, all modes
    e: jnp.ndarray       # (nx, ny)     constant off-diagonal per mode
    dct: DctModes | None = None   # set when method == "dct"


jax.tree_util.register_dataclass(BandedProj2D, data_fields=["D_mod", "e", "dct"], meta_fields=[])


def _banded_entries_2d(problem, vareps: float):
    nt, dt = problem.nt, problem.dt
    lx, ly = problem.lambda_x, problem.lambda_y  # (nx,), (ny,); lx[0]=ly[0]=0
    lxy = lx[:, None] + ly[None, :]              # (nx, ny), lxy[0, 0] = 0
    mid = 2 / dt**2 + lxy + lxy**2 * vareps**2 / 2
    top = 1 / dt**2 + lxy * (1 + vareps / dt) + lxy**2 * vareps**2 / 4
    bot = 1 / dt**2 + lxy * (1 - vareps / dt) + lxy**2 * vareps**2 / 4
    e = -1 / dt**2 + lxy**2 * vareps**2 / 4  # (nx, ny)
    return top, mid, bot, e


def _stack_diag(nt, top, mid, bot):
    return jnp.concatenate([top[None], jnp.broadcast_to(mid[None], (nt - 2, *mid.shape)), bot[None]], axis=0)


def precomp_banded_proj_2d(problem, vareps: float, method: str = "thomas") -> BandedProj2D:
    """Precompute the per-mode time systems for all (kx,ky) modes.

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

    method="thomas": the (kx,ky)=(0,0) DC mode has lxy=0 and a singular T;
    its diagonal is set to 1 (off-diagonal 0) so a zeroed RHS there solves
    to zero and proj_fokker_planck_banded_2d overwrites that mode via 1D
    DCT-in-t, exactly as in the 1D banded projection and
    matlab/shared/2d/utils/precomp_banded_proj.m.
    method="dct": see the module docstring; the DC mode is handled inside.
    """
    nt = problem.nt
    top, mid, bot, e = _banded_entries_2d(problem, vareps)
    D = _stack_diag(nt, top, mid, bot)
    D = D.at[:, 0, 0].set(1.0)
    e_th = e.at[0, 0].set(0.0)
    D_mod = thomas_batch_precomp(D, e_th)
    dct = precomp_dct_modes(nt, top, mid, bot, e, problem.lambda_t) if method == "dct" else None
    if method not in ("thomas", "dct"):
        raise ValueError(f"unknown method {method!r}")
    return BandedProj2D(D_mod=D_mod, e=e_th, dct=dct)


def solve_time_modes(bp, f_hat: jnp.ndarray, lambda_t: jnp.ndarray) -> jnp.ndarray:
    """phi_hat = T^{-1} f_hat for every (kx,ky) mode, including the DC mode
    (inverted by DCT-in-t on its own Neumann time-Laplacian).

    Works for any precomp object carrying (D_mod, e, dct): BandedProj2D and
    projection_expsemi_2d.ExpsemiProj2D.
    """
    if bp.dct is not None:
        return solve_dct_modes(bp.dct, f_hat)
    rhs = f_hat.at[:, 0, 0].set(0.0)
    phi_hat = thomas_batch_solve(bp.D_mod, bp.e, rhs)
    f1_t = _dct_axis(f_hat[:, 0, 0], axis=0)
    phi1_t = jnp.concatenate([jnp.zeros(1), f1_t[1:] / lambda_t[1:]])
    return phi_hat.at[:, 0, 0].set(_idct_axis(phi1_t, axis=0))


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

    f_hat = dct2_xy(f)  # (nt, nx, ny)
    phi_hat = solve_time_modes(bp, f_hat, problem.lambda_t)
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
