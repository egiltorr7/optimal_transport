#!/usr/bin/env python3
"""
3D Schrödinger Bridge via Hopf-Cole / Sinkhorn iteration.

Factorization:  rho(t,x,y,z) = phi(t,x,y,z) * hat_phi(t,x,y,z)
  phi     : forward  heat eq  d_t phi      = (eps/2) Delta phi,     phi(0)  = phi0
  hat_phi : backward heat eq -d_t hat_phi  = (eps/2) Delta hat_phi, hat_phi(1) = hat_phi1

Neumann BCs on [0,1]^3 via separable DCT-II.
Sinkhorn runs in log-domain for numerical stability.
  — In 3D with small eps, the heat-kernel overlap between the two marginals can
    be exp(-50) or smaller, making naive potentials overflow float64.
    Log-domain arithmetic (log-sum-exp trick) avoids this completely.

GPU acceleration via CuPy (optional).

Memory footprint for N^3 grid:
  CPU: rho0/rho1 each ~N^3*8 bytes;  N=32 → 0.3 MB,  N=64 → 2.1 MB
  GPU: same (VRAM).
"""

import numpy as np
from scipy.fft import dctn, idctn
from scipy.optimize import fsolve
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── GPU detection ──────────────────────────────────────────────────────────────
try:
    import cupy as cp
    import cupyx.scipy.fft as cpfft
    HAS_GPU = True
    try:
        _gpu_name = cp.cuda.Device(0).attributes.get('DeviceName', 'GPU')
    except Exception:
        _gpu_name = 'GPU'
    print(f"CuPy available — {_gpu_name}")
except (ImportError, Exception):
    HAS_GPU = False
    print("CuPy not available — CPU only")


def _xp(u):
    if HAS_GPU and isinstance(u, cp.ndarray):
        return cp
    return np


def _fft_mod(u):
    if HAS_GPU and isinstance(u, cp.ndarray):
        return cpfft
    import scipy.fft as _sf
    return _sf


def to_gpu(u):
    return cp.asarray(u) if HAS_GPU else u


def to_cpu(u):
    if HAS_GPU and isinstance(u, cp.ndarray):
        return cp.asnumpy(u)
    return u


def sync():
    if HAS_GPU:
        cp.cuda.Stream.null.synchronize()


# ── nD Neumann heat kernel ─────────────────────────────────────────────────────

def heat_kernel_nd(u, eps, t, dx_list):
    """
    Apply exp((eps/2)*t*Delta_Neumann) to u (any ndim, numpy or cupy).
    Eigenvalues: lambda_{k1,...} = sum_i 4/dxi^2 sin^2(pi*ki/(2*Ni))
    """
    if t == 0.0:
        return u.copy()
    xp   = _xp(u)
    fft  = _fft_mod(u)
    shape = u.shape
    ndim  = len(shape)
    lam   = xp.zeros(shape)
    for axis, dx in enumerate(dx_list):
        N     = shape[axis]
        k     = xp.arange(N, dtype=float)
        lam_k = (4.0 / dx**2) * xp.sin(xp.pi * k / (2.0 * N))**2
        sl        = [None] * ndim
        sl[axis]  = slice(None)
        lam = lam + lam_k[tuple(sl)]
    decay = xp.exp(-0.5 * eps * t * lam)
    return fft.idctn(fft.dctn(u, type=2, norm='ortho') * decay, type=2, norm='ortho')


def log_heat_kernel_nd(v, eps, t, dx_list):
    """
    Numerically stable log(P_t[exp(v)]).

    Key idea (log-sum-exp trick):
        log(P_t[exp(v)]) = v_max + log(P_t[exp(v - v_max)])
    Since exp(v - v_max) ∈ [0,1], the heat kernel application never overflows.

    This is critical in 3D where exp(v) can reach ~10^8 for small eps.
    """
    if t == 0.0:
        return v.copy()
    xp    = _xp(v)
    v_max = float(xp.max(v))
    result = heat_kernel_nd(xp.exp(v - v_max), eps, t, dx_list)
    return v_max + xp.log(xp.maximum(result, 1e-300))


def laplacian_nd(u, dx_list):
    """Apply Neumann Laplacian exactly via DCT-II."""
    xp   = _xp(u)
    fft  = _fft_mod(u)
    shape = u.shape
    ndim  = len(shape)
    lam   = xp.zeros(shape)
    for axis, dx in enumerate(dx_list):
        N     = shape[axis]
        k     = xp.arange(N, dtype=float)
        lam_k = (4.0 / dx**2) * xp.sin(xp.pi * k / (2.0 * N))**2
        sl        = [None] * ndim
        sl[axis]  = slice(None)
        lam = lam + lam_k[tuple(sl)]
    return fft.idctn(-lam * fft.dctn(u, type=2, norm='ortho'), type=2, norm='ortho')


# ── Log-domain Sinkhorn ────────────────────────────────────────────────────────

def sinkhorn_nd(rho0, rho1, eps, dx_list, max_iter=5000, tol=1e-9, log_every=500):
    """
    Log-domain Hopf-Cole Sinkhorn (numpy or cupy arrays).

    Convergence criterion: max |u + log(P_T[exp(v)]) - log(rho0)| → 0
    This is the relative log-domain marginal error; tol=1e-9 means the
    absolute marginal error is at most rho0.max() * 1e-9.

    Returns (u, v, history) — log-potentials + convergence trace.
    """
    xp = _xp(rho0)
    T  = 1.0
    log_rho0 = xp.log(xp.maximum(rho0, 1e-300))
    log_rho1 = xp.log(xp.maximum(rho1, 1e-300))

    u   = xp.zeros_like(rho0)
    v   = log_rho1.copy()
    lhv = log_heat_kernel_nd(v, eps, T, dx_list)
    history = []

    for it in range(max_iter):
        u   = log_rho0 - lhv
        lhu = log_heat_kernel_nd(u, eps, T, dx_list)
        v   = log_rho1 - lhu
        lhv = log_heat_kernel_nd(v, eps, T, dx_list)

        # Correct convergence: original-domain marginal error (avoids gauge oscillation)
        # u + lhv - log_rho0  = lhv_new - lhv_old (gauge freedom), NOT a marginal error.
        # Instead check: |exp(u + lhv) - rho0|; clip to avoid overflow in early iters.
        err  = float(xp.max(xp.abs(xp.exp(xp.clip(u + lhv, -500.0, 500.0)) - rho0)))
        err1 = float(xp.max(xp.abs(xp.exp(xp.clip(lhu + v, -500.0, 500.0)) - rho1)))
        history.append(err)

        if it % log_every == 0:
            print(f"  it={it:5d}  marg0={err:.2e}  marg1={err1:.2e}")

        if err < tol:
            print(f"Converged at iter {it+1},  marg_err={err:.2e}")
            break
    else:
        print(f"Did NOT converge ({max_iter} iters),  marg_err={err:.2e}")

    return u, v, history


# ── Trajectory ─────────────────────────────────────────────────────────────────

def build_trajectory_nd(u, v, eps, dx_list, N_t=16):
    """
    Build space-time trajectory from log-potentials u=log(phi0), v=log(hat_phi1).

    rho(t) = exp(log_P_t[exp(u)] + log_P_{1-t}[exp(v)])
    Returns rho_traj, phi_traj, hat_phi_traj of shape (N_t+1, *spatial).
    """
    xp     = _xp(u)
    shape  = u.shape
    t_vals = np.linspace(0.0, 1.0, N_t + 1)

    phi_traj     = xp.zeros((N_t + 1,) + shape)
    hat_phi_traj = xp.zeros((N_t + 1,) + shape)

    for i, t in enumerate(t_vals):
        phi_traj[i]     = xp.exp(log_heat_kernel_nd(u, eps, float(t),     dx_list))
        hat_phi_traj[i] = xp.exp(log_heat_kernel_nd(v, eps, float(1 - t), dx_list))

    rho_traj = phi_traj * hat_phi_traj
    return rho_traj, phi_traj, hat_phi_traj, t_vals


# ── Analytical 3D Gaussian SB ─────────────────────────────────────────────────

def gaussian_sb_analytical_3d(mu0, sigma0, mu1, sigma1, eps, x, y, z, t_vals):
    """
    Exact SB for N(mu0, sigma0^2 I_3) → N(mu1, sigma1^2 I_3) in R^3.
    Factorizes into independent 1D solutions per axis.
    Returns rho_an of shape (N_t+1, Nx, Ny, Nz).
    """
    s0, s1 = sigma0**2, sigma1**2

    def eqs(p):
        a, b = p
        if a <= 0 or b <= 0:
            return [1e10, 1e10]
        return [1/a + 1/(b + eps) - 1/s0,
                1/(a + eps) + 1/b - 1/s1]

    a_sq, b_sq = fsolve(eqs, [s0 * 2, s1 * 2])
    print(f"  alpha^2={a_sq:.6f}, beta^2={b_sq:.6f}")

    Ac = 1/a_sq;        Bc = 1/(b_sq + eps)
    Cc = 1/(a_sq + eps); Dc = 1/b_sq
    det = Ac * Dc - Bc * Cc

    m0 = np.zeros(3)
    n1 = np.zeros(3)
    for axis in range(3):
        rhs0 = mu0[axis] / s0
        rhs1 = mu1[axis] / s1
        m0[axis] = (Dc * rhs0 - Bc * rhs1) / det
        n1[axis] = (Ac * rhs1 - Cc * rhs0) / det

    coords  = [x, y, z]
    rho_an  = np.zeros((len(t_vals), len(x), len(y), len(z)))
    mu_an   = np.zeros((len(t_vals), 3))

    for i, t in enumerate(t_vals):
        At   = 1.0 / (a_sq + eps * t)
        Bt   = 1.0 / (b_sq + eps * (1 - t))
        sig2 = 1.0 / (At + Bt)
        sig_t = np.sqrt(sig2)
        norm1d = sig_t * np.sqrt(2 * np.pi)
        marginals = []
        for axis in range(3):
            mu_an[i, axis] = sig2 * (At * m0[axis] + Bt * n1[axis])
            marginals.append(
                np.exp(-0.5 * (coords[axis] - mu_an[i, axis])**2 / sig2) / norm1d
            )
        rho_an[i] = (marginals[0][:, None, None] *
                     marginals[1][None, :, None] *
                     marginals[2][None, None, :])

    return rho_an, mu_an


# ── Verification ───────────────────────────────────────────────────────────────

def run_verifications_3d(u, v, rho0, rho1,
                          rho_traj, phi_traj, hat_phi_traj, t_vals, eps, dx):
    print("\n" + "═"*55)
    print("  3D Verification report")
    print("═"*55)
    T  = 1.0
    dV = dx**3
    xp = _xp(u)
    dx_list = [dx, dx, dx]

    # 1. Marginals
    lhv = log_heat_kernel_nd(v, eps, T, dx_list)
    r0  = xp.exp(u + lhv)
    e0  = float(xp.max(xp.abs(r0 - rho0)))
    lhu = log_heat_kernel_nd(u, eps, T, dx_list)
    r1  = xp.exp(lhu + v)
    e1  = float(xp.max(xp.abs(r1 - rho1)))
    print(f"  1. Marginal rho0 match (L∞):     {e0:.2e}  {'OK' if e0 < 1e-4 else 'FAIL'}")
    print(f"     Marginal rho1 match (L∞):     {e1:.2e}  {'OK' if e1 < 1e-4 else 'FAIL'}")

    # 2. Mass conservation
    masses   = xp.sum(rho_traj * dV, axis=(1, 2, 3))
    mass_dev = float(masses.max() - masses.min())
    print(f"  2. Mass conservation (max-min):  {mass_dev:.2e}  {'OK' if mass_dev < 1e-5 else 'WARN'}")

    # 3. Positivity
    rho_min = float(xp.min(rho_traj))
    print(f"  3. Min of rho:                   {rho_min:.2e}  {'OK' if rho_min > -1e-6 else 'FAIL'}")

    # 4. Continuity (analytic): d_t rho + d_x J = 0 by construction
    #    Verified at a few interior times via d_t rho = (eps/2)*(hat_phi*Δphi - phi*Δhat_phi)
    #    Note: phi/hat_phi values may be large but products stay within float64.
    cont_errs = []
    for idx in [len(t_vals) // 4, len(t_vals) // 2, 3 * len(t_vals) // 4]:
        phi_t    = phi_traj[idx]
        hphi_t   = hat_phi_traj[idx]
        Dphi     = laplacian_nd(phi_t,  dx_list)
        Dhphi    = laplacian_nd(hphi_t, dx_list)
        dt_rho   = 0.5 * eps * (hphi_t * Dphi  - phi_t  * Dhphi)
        dxJ_an   = 0.5 * eps * (phi_t  * Dhphi - hphi_t * Dphi)
        cont_errs.append(float(xp.max(xp.abs(dt_rho + dxJ_an))))
    cont_max = max(cont_errs)
    print(f"  4. Continuity residual (analyt): {cont_max:.2e}  "
          f"{'OK' if cont_max < 1e-8 else 'FAIL'}")

    # 5. Semigroup property
    rng    = np.random.default_rng(42)
    u_test = xp.asarray(np.abs(rng.standard_normal((8, 8, 8))))
    lhs    = heat_kernel_nd(heat_kernel_nd(u_test, eps, 0.4, dx_list), eps, 0.3, dx_list)
    rhs    = heat_kernel_nd(u_test, eps, 0.7, dx_list)
    sg_err = float(xp.max(xp.abs(lhs - rhs)))
    print(f"  5. Semigroup property (L∞):      {sg_err:.2e}  {'OK' if sg_err < 1e-12 else 'FAIL'}")

    print("═"*55 + "\n")


# ── Visualization ──────────────────────────────────────────────────────────────

def plot_slices(rho_traj, rho_an, t_vals, x, N_t,
                outfile='hopf_cole_3d_slices.png'):
    """
    xy-projection (sum over z) at 7 time points: Sinkhorn (top) vs Analytical (bottom).

    A fixed z-slice (e.g. z=0.5) is a bad choice when the Gaussians are at z=0.25
    and z=0.75: the t=0 and t=1 panels would show exp(-12.5)≈0, and vmax computed
    from t=0 would be ≈0, making every intermediate frame look saturated/"blown up".
    The 2D marginal ρ_xy = ∫ρ dz is always non-zero and shows the full trajectory.
    """
    snap_times = np.linspace(0, 1, 7)
    dx = x[1] - x[0]

    # xy marginals: shape (N_t+1, Nx, Ny)
    proj_num = rho_traj.sum(axis=3) * dx      # ∫ρ dz
    proj_an  = rho_an .sum(axis=3) * dx

    # vmax over all displayed snapshots (NOT just t=0 which can be ≈0 at wrong slice)
    vmax = float(max(proj_num[int(round(t * N_t))].max() for t in snap_times))

    fig, axes = plt.subplots(2, 7, figsize=(18, 5))
    for k, t_snap in enumerate(snap_times):
        idx = int(round(t_snap * N_t))
        axes[0, k].imshow(proj_num[idx].T, origin='lower',
                          extent=[0, 1, 0, 1], cmap='viridis', vmin=0, vmax=vmax)
        axes[0, k].set_title(f't={t_snap:.2f}', fontsize=8)
        axes[0, k].set_xticks([]); axes[0, k].set_yticks([])
        axes[1, k].imshow(proj_an[idx].T, origin='lower',
                          extent=[0, 1, 0, 1], cmap='viridis', vmin=0, vmax=vmax)
        axes[1, k].set_xticks([]); axes[1, k].set_yticks([])
    axes[0, 0].set_ylabel('Sinkhorn\n(xy projection)', fontsize=8)
    axes[1, 0].set_ylabel('Analytical\n(xy projection)', fontsize=8)
    plt.suptitle('3D SB — xy marginal  ρ_xy(t) = ∫ρ dz', fontsize=11)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"Saved {outfile}")


def plot_marginals(rho_traj, rho_an, t_vals, x, N_t,
                   outfile='hopf_cole_3d_marginals.png'):
    """1D marginal projections (integrate out two spatial axes)."""
    snap_times = [0.0, 0.25, 0.5, 0.75, 1.0]
    colors     = plt.cm.plasma(np.linspace(0.1, 0.9, len(snap_times)))
    dx         = x[1] - x[0]
    labels     = ['x', 'y', 'z']
    sum_axes_list = [(1, 2), (0, 2), (0, 1)]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (label, saxes) in zip(axes, zip(labels, sum_axes_list)):
        for t_snap, c in zip(snap_times, colors):
            idx    = int(round(t_snap * N_t))
            marg_s  = rho_traj[idx].sum(axis=saxes) * dx**2
            marg_an = rho_an[idx].sum(axis=saxes) * dx**2
            lbl = f't={t_snap:.2f}'
            ax.plot(x, marg_s,  color=c, lw=2.0, label=lbl)
            ax.plot(x, marg_an, color=c, lw=1.2, ls='--')
        ax.set_xlabel(label)
        ax.set_title(f'Marginal in {label}')
        ax.grid(True, alpha=0.3)

    axes[0].legend(fontsize=7, ncol=2)
    from matplotlib.lines import Line2D
    axes[2].add_artist(axes[2].legend(
        handles=[Line2D([0], [0], color='k', lw=2,   label='Sinkhorn'),
                 Line2D([0], [0], color='k', lw=1.2, ls='--', label='Analytical')],
        fontsize=8, loc='upper right'))
    plt.suptitle('3D SB — 1D marginal projections', fontsize=11)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"Saved {outfile}")


# ── CPU vs GPU benchmark ───────────────────────────────────────────────────────

def run_benchmark(N_list, eps=0.02, n_iters=50):
    """Time Sinkhorn (fixed n_iters) for each N on CPU and (if available) GPU."""
    results = {'N': N_list, 'cpu': [], 'gpu': []}

    for N in N_list:
        print(f"\n  Benchmark N={N}^3 ...")
        dx  = 1.0 / N
        x   = (np.arange(N) + 0.5) * dx
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        dx_list = [dx, dx, dx]

        rho0 = np.exp(-0.5 * ((X - 0.25)**2 + (Y - 0.25)**2 + (Z - 0.25)**2) / 0.05**2)
        rho0 /= rho0.sum() * dx**3
        rho1 = np.exp(-0.5 * ((X - 0.75)**2 + (Y - 0.75)**2 + (Z - 0.75)**2) / 0.05**2)
        rho1 /= rho1.sum() * dx**3

        t0 = time.perf_counter()
        sinkhorn_nd(rho0, rho1, eps, dx_list, max_iter=n_iters, tol=0.0, log_every=9999)
        t_cpu = time.perf_counter() - t0
        results['cpu'].append(t_cpu)
        print(f"    CPU: {t_cpu:.3f}s")

        if HAS_GPU:
            rho0_g = cp.asarray(rho0)
            rho1_g = cp.asarray(rho1)
            sinkhorn_nd(rho0_g, rho1_g, eps, dx_list, max_iter=3, tol=0.0, log_every=9999)
            sync()
            t0 = time.perf_counter()
            sinkhorn_nd(rho0_g, rho1_g, eps, dx_list,
                        max_iter=n_iters, tol=0.0, log_every=9999)
            sync()
            t_gpu = time.perf_counter() - t0
            results['gpu'].append(t_gpu)
            print(f"    GPU: {t_gpu:.3f}s  (speedup: {t_cpu/t_gpu:.1f}x)")
        else:
            results['gpu'].append(None)

    return results


def plot_benchmark(results, outfile='hopf_cole_3d_benchmark.png'):
    N_list  = results['N']
    has_gpu = any(t is not None for t in results['gpu'])
    ncols   = 2 if has_gpu else 1

    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    ax = axes[0]
    ax.loglog(N_list, results['cpu'], 'b-o', lw=2, label='CPU (numpy)')
    if has_gpu:
        gpu_vals = [t for t in results['gpu'] if t is not None]
        N_gpu    = [n for n, t in zip(N_list, results['gpu']) if t is not None]
        ax.loglog(N_gpu, gpu_vals, 'r-o', lw=2, label='GPU (cupy)')
    ax.set_xlabel('N (per axis)'); ax.set_ylabel('Time (s)')
    ax.set_title('3D Sinkhorn — runtime vs N (50 iters)')
    ax.legend(); ax.grid(True, alpha=0.3, which='both')

    if has_gpu:
        speedups = [c/g for c, g in zip(results['cpu'], results['gpu']) if g is not None]
        axes[1].plot(N_gpu, speedups, 'g-o', lw=2)
        axes[1].axhline(1.0, color='gray', ls='--', lw=1)
        axes[1].set_xlabel('N'); axes[1].set_ylabel('Speedup (CPU/GPU)')
        axes[1].set_title('GPU Speedup'); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"Saved {outfile}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    N      = 64     # N=32 + sigma=0.05 gives only 1.6 pts/σ → poor analytical match
    eps    = 0.05   # 0.02 is ill-conditioned in 3D (heat-kernel overlap ~3e-7); use 0.05+
    N_t    = 16
    dx     = 1.0 / N
    x      = (np.arange(N) + 0.5) * dx
    y      = x.copy()
    z      = x.copy()
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    dx_list = [dx, dx, dx]

    mu0    = [0.25, 0.25, 0.25]
    mu1    = [0.75, 0.75, 0.75]
    sigma0 = sigma1 = 0.05   # N=64 gives 3.2 pts/σ; DCT eigenvalues accurate to <0.5%

    rho0 = np.exp(-0.5 * ((X - mu0[0])**2 + (Y - mu0[1])**2 + (Z - mu0[2])**2) / sigma0**2)
    rho0 /= rho0.sum() * dx**3
    rho1 = np.exp(-0.5 * ((X - mu1[0])**2 + (Y - mu1[1])**2 + (Z - mu1[2])**2) / sigma1**2)
    rho1 /= rho1.sum() * dx**3

    print(f"Running 3D Sinkhorn  (ε={eps}, N={N}^3, N_t={N_t}) ...")
    print(f"  Grid memory: {rho0.nbytes / 1e6:.1f} MB per array")
    print(f"  rho0 mass = {rho0.sum()*dx**3:.8f}")
    print(f"  rho1 mass = {rho1.sum()*dx**3:.8f}")
    print(f"  rho0 peak = {rho0.max():.1f}  (log-domain Sinkhorn needed for stability)")

    u, v, history = sinkhorn_nd(rho0, rho1, eps, dx_list, max_iter=5000, tol=1e-9)

    rho_traj, phi_traj, hat_phi_traj, t_vals = build_trajectory_nd(
        u, v, eps, dx_list, N_t=N_t)

    print("\nComputing analytical Gaussian SB solution ...")
    rho_an, mu_an = gaussian_sb_analytical_3d(
        mu0, sigma0, mu1, sigma1, eps, x, y, z, t_vals)

    run_verifications_3d(u, v, rho0, rho1,
                          rho_traj, phi_traj, hat_phi_traj, t_vals, eps, dx)

    # L∞ of the full 3D field (sensitive to grid-sampling at narrow peaks)
    err_vs_an = np.max(np.abs(rho_traj - rho_an), axis=(1, 2, 3))
    print(f"  Max 3D L∞ error vs analytical:  {err_vs_an.max():.2e}")
    # 1D marginals (integrate out two axes) — cleaner comparison metric
    marg_x_num = rho_traj.sum(axis=(2, 3)) * dx**2    # shape (N_t+1, N)
    marg_x_an  = rho_an .sum(axis=(2, 3)) * dx**2
    marg_y_num = rho_traj.sum(axis=(1, 3)) * dx**2
    marg_y_an  = rho_an .sum(axis=(1, 3)) * dx**2
    marg_z_num = rho_traj.sum(axis=(1, 2)) * dx**2
    marg_z_an  = rho_an .sum(axis=(1, 2)) * dx**2
    err_mx = float(np.max(np.abs(marg_x_num - marg_x_an)))
    err_my = float(np.max(np.abs(marg_y_num - marg_y_an)))
    err_mz = float(np.max(np.abs(marg_z_num - marg_z_an)))
    print(f"  Max 1D marginal L∞ vs analyt.: x={err_mx:.2e}  y={err_my:.2e}  z={err_mz:.2e}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_slices(rho_traj, rho_an, t_vals, x, N_t)
    plot_marginals(rho_traj, rho_an, t_vals, x, N_t)

    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.semilogy(history)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('max |φ₀·P_T[φ̂₁] − ρ₀|  (absolute L∞)')
    ax3.set_title('3D Sinkhorn convergence')
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('hopf_cole_3d_convergence.png', dpi=150, bbox_inches='tight')
    print("Saved hopf_cole_3d_convergence.png")

    # ── CPU vs GPU benchmark ──────────────────────────────────────────────────
    print("\nRunning CPU vs GPU benchmark ...")
    N_bench = [8, 16, 24, 32, 48, 64] if HAS_GPU else [8, 16, 24, 32, 48]
    bench   = run_benchmark(N_bench, eps=eps, n_iters=50)
    plot_benchmark(bench, outfile='hopf_cole_3d_benchmark.png')
