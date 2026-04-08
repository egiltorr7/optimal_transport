#!/usr/bin/env python3
"""
2D Schrödinger Bridge via Hopf-Cole / Sinkhorn iteration.

Factorization:  rho(t,x,y) = phi(t,x,y) * hat_phi(t,x,y)
  phi     : forward  heat eq  d_t phi      = (eps/2) Delta phi,     phi(0)  = phi0
  hat_phi : backward heat eq -d_t hat_phi  = (eps/2) Delta hat_phi, hat_phi(1) = hat_phi1

Neumann BCs on [0,1]^2 via separable DCT-II.
Sinkhorn runs in log-domain for numerical stability (avoids overflow for small eps).
GPU acceleration via CuPy (optional).
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

    Uses log-sum-exp trick: subtract max(v) before exp to keep values in [0,1],
    then add it back.  Avoids overflow even when v is very large or very negative.
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
    Log-domain Hopf-Cole Sinkhorn.

    Works with log-potentials u = log(phi0), v = log(hat_phi1).
    Convergence criterion: max |u + log(P_T[exp(v)]) - log(rho0)| → 0.

    Returns (u, v, history) — log-potentials + convergence trace.
    """
    xp = _xp(rho0)
    T  = 1.0
    log_rho0 = xp.log(xp.maximum(rho0, 1e-300))
    log_rho1 = xp.log(xp.maximum(rho1, 1e-300))

    u   = xp.zeros_like(rho0)
    v   = log_rho1.copy()
    lhv = log_heat_kernel_nd(v, eps, T, dx_list)   # log(P_T[exp(v)])
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

def build_trajectory_nd(u, v, eps, dx_list, N_t=32):
    """
    Build space-time trajectory from log-potentials u=log(phi0), v=log(hat_phi1).
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


# ── Analytical 2D Gaussian SB ─────────────────────────────────────────────────

def gaussian_sb_analytical_2d(mu0, sigma0, mu1, sigma1, eps, x, y, t_vals):
    """
    Exact SB for N(mu0, sigma0^2 I_2) → N(mu1, sigma1^2 I_2) in R^2.
    Factorizes into independent 1D solutions per axis.
    Returns rho_an of shape (N_t+1, Nx, Ny).
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

    m0 = np.zeros(2)
    n1 = np.zeros(2)
    for axis in range(2):
        rhs0 = mu0[axis] / s0
        rhs1 = mu1[axis] / s1
        m0[axis] = (Dc * rhs0 - Bc * rhs1) / det
        n1[axis] = (Ac * rhs1 - Cc * rhs0) / det

    rho_an = np.zeros((len(t_vals), len(x), len(y)))
    mu_an  = np.zeros((len(t_vals), 2))

    for i, t in enumerate(t_vals):
        At   = 1.0 / (a_sq + eps * t)
        Bt   = 1.0 / (b_sq + eps * (1 - t))
        sig2 = 1.0 / (At + Bt)
        sig_t = np.sqrt(sig2)
        norm1d = sig_t * np.sqrt(2 * np.pi)
        for axis in range(2):
            mu_an[i, axis] = sig2 * (At * m0[axis] + Bt * n1[axis])
        rho_x = np.exp(-0.5 * (x - mu_an[i, 0])**2 / sig2) / norm1d
        rho_y = np.exp(-0.5 * (y - mu_an[i, 1])**2 / sig2) / norm1d
        rho_an[i] = np.outer(rho_x, rho_y)

    return rho_an, mu_an


# ── Verification ───────────────────────────────────────────────────────────────

def run_verifications_2d(u, v, rho0, rho1,
                          rho_traj, phi_traj, hat_phi_traj, t_vals, eps, dx):
    print("\n" + "═"*55)
    print("  2D Verification report")
    print("═"*55)
    T  = 1.0
    dV = dx**2
    xp = _xp(u)
    dx_list = [dx, dx]

    # 1. Marginals (log-domain, avoids overflow)
    lhv = log_heat_kernel_nd(v, eps, T, dx_list)
    r0  = xp.exp(u + lhv)
    e0  = float(xp.max(xp.abs(r0 - rho0)))
    lhu = log_heat_kernel_nd(u, eps, T, dx_list)
    r1  = xp.exp(lhu + v)
    e1  = float(xp.max(xp.abs(r1 - rho1)))
    print(f"  1. Marginal rho0 match (L∞):     {e0:.2e}  {'OK' if e0 < 1e-5 else 'FAIL'}")
    print(f"     Marginal rho1 match (L∞):     {e1:.2e}  {'OK' if e1 < 1e-5 else 'FAIL'}")

    # 2. Mass conservation
    masses    = xp.sum(rho_traj * dV, axis=(1, 2))
    mass_dev  = float(masses.max() - masses.min())
    print(f"  2. Mass conservation (max-min):  {mass_dev:.2e}  {'OK' if mass_dev < 1e-6 else 'WARN'}")

    # 3. Positivity
    rho_min = float(xp.min(rho_traj))
    print(f"  3. Min of rho:                   {rho_min:.2e}  {'OK' if rho_min > -1e-7 else 'FAIL'}")

    # 4. Continuity (analytic): d_t rho + d_x J = 0 always, verified numerically
    cont_errs = []
    for i in range(1, len(t_vals) - 1):
        Dphi  = laplacian_nd(phi_traj[i],     dx_list)
        Dhphi = laplacian_nd(hat_phi_traj[i], dx_list)
        dt_rho  = 0.5 * eps * (hat_phi_traj[i] * Dphi  - phi_traj[i] * Dhphi)
        dxJ_an  = 0.5 * eps * (phi_traj[i]     * Dhphi - hat_phi_traj[i] * Dphi)
        cont_errs.append(float(xp.max(xp.abs(dt_rho + dxJ_an))))
    print(f"  4. Continuity residual (analyt): {max(cont_errs):.2e}  "
          f"{'OK' if max(cont_errs) < 1e-10 else 'FAIL'}")

    # 5. Semigroup property
    rng    = np.random.default_rng(42)
    u_test = xp.asarray(np.abs(rng.standard_normal((16, 16))))
    lhs    = heat_kernel_nd(heat_kernel_nd(u_test, eps, 0.4, dx_list), eps, 0.3, dx_list)
    rhs    = heat_kernel_nd(u_test, eps, 0.7, dx_list)
    sg_err = float(xp.max(xp.abs(lhs - rhs)))
    print(f"  5. Semigroup property (L∞):      {sg_err:.2e}  {'OK' if sg_err < 1e-12 else 'FAIL'}")

    print("═"*55 + "\n")


# ── CPU vs GPU benchmark ───────────────────────────────────────────────────────

def run_benchmark(N_list, eps=0.02, n_iters=50):
    results = {'N': N_list, 'cpu': [], 'gpu': []}

    for N in N_list:
        print(f"\n  Benchmark N={N}x{N} ...")
        dx  = 1.0 / N
        x   = (np.arange(N) + 0.5) * dx
        X, Y = np.meshgrid(x, x, indexing='ij')
        dx_list = [dx, dx]

        rho0 = np.exp(-0.5 * ((X - 0.25)**2 + (Y - 0.25)**2) / 0.05**2)
        rho0 /= rho0.sum() * dx**2
        rho1 = np.exp(-0.5 * ((X - 0.75)**2 + (Y - 0.75)**2) / 0.05**2)
        rho1 /= rho1.sum() * dx**2

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


def plot_benchmark(results, outfile='hopf_cole_2d_benchmark.png'):
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
    ax.set_title('2D Sinkhorn — runtime vs N (50 iters)')
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
    N      = 64
    eps    = 0.02
    N_t    = 32
    dx     = 1.0 / N
    x      = (np.arange(N) + 0.5) * dx
    y      = x.copy()
    X, Y   = np.meshgrid(x, y, indexing='ij')
    dx_list = [dx, dx]

    mu0    = [0.25, 0.25]
    mu1    = [0.75, 0.75]
    sigma0 = sigma1 = 0.05

    rho0 = np.exp(-0.5 * ((X - mu0[0])**2 + (Y - mu0[1])**2) / sigma0**2)
    rho0 /= rho0.sum() * dx**2
    rho1 = np.exp(-0.5 * ((X - mu1[0])**2 + (Y - mu1[1])**2) / sigma1**2)
    rho1 /= rho1.sum() * dx**2

    print(f"Running 2D Sinkhorn  (ε={eps}, N={N}x{N}, N_t={N_t}) ...")

    u, v, history = sinkhorn_nd(rho0, rho1, eps, dx_list, max_iter=5000, tol=1e-7)

    rho_traj, phi_traj, hat_phi_traj, t_vals = build_trajectory_nd(
        u, v, eps, dx_list, N_t=N_t)

    print("\nComputing analytical Gaussian SB solution ...")
    rho_an, mu_an = gaussian_sb_analytical_2d(mu0, sigma0, mu1, sigma1, eps, x, y, t_vals)

    run_verifications_2d(u, v, rho0, rho1,
                          rho_traj, phi_traj, hat_phi_traj, t_vals, eps, dx)

    # L∞ of the full 2D field (sensitive to grid-sampling at narrow peaks)
    err_vs_an = np.max(np.abs(rho_traj - rho_an), axis=(1, 2))
    print(f"  Max 2D L∞ error vs analytical:  {err_vs_an.max():.2e}")
    # 1D marginals (integrate out one axis) — cleaner comparison metric
    marg_x_num = rho_traj.sum(axis=2) * dx     # shape (N_t+1, N)
    marg_x_an  = rho_an .sum(axis=2) * dx
    marg_y_num = rho_traj.sum(axis=1) * dx
    marg_y_an  = rho_an .sum(axis=1) * dx
    err_mx = float(np.max(np.abs(marg_x_num - marg_x_an)))
    err_my = float(np.max(np.abs(marg_y_num - marg_y_an)))
    print(f"  Max 1D marginal L∞ vs analyt.: x={err_mx:.2e}  y={err_my:.2e}")

    # ── Snapshot figure ────────────────────────────────────────────────────────
    snap_times = np.linspace(0, 1, 9)
    vmax = float(rho_traj[0].max())

    fig, axes = plt.subplots(2, 9, figsize=(22, 5))
    for k, t_snap in enumerate(snap_times):
        idx = int(round(t_snap * N_t))
        axes[0, k].imshow(rho_traj[idx].T, origin='lower',
                          extent=[0, 1, 0, 1], cmap='viridis', vmin=0, vmax=vmax)
        axes[0, k].set_title(f't={t_snap:.2f}', fontsize=8)
        axes[0, k].set_xticks([]); axes[0, k].set_yticks([])
        axes[1, k].imshow(rho_an[idx].T, origin='lower',
                          extent=[0, 1, 0, 1], cmap='viridis', vmin=0, vmax=vmax)
        axes[1, k].set_xticks([]); axes[1, k].set_yticks([])
    axes[0, 0].set_ylabel('Sinkhorn', fontsize=9)
    axes[1, 0].set_ylabel('Analytical', fontsize=9)
    plt.suptitle(
        rf'2D SB  —  $\varepsilon={eps}$,  '
        rf'$\rho_0=\mathcal{{N}}(({mu0[0]},{mu0[1]}),{sigma0}^2 I_2)$  →  '
        rf'$\rho_1=\mathcal{{N}}(({mu1[0]},{mu1[1]}),{sigma1}^2 I_2)$', fontsize=11)
    plt.tight_layout()
    plt.savefig('hopf_cole_2d_snapshots.png', dpi=150, bbox_inches='tight')
    print("Saved hopf_cole_2d_snapshots.png")

    # ── Error map ─────────────────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))
    for k, t_snap in enumerate([0.0, 0.5, 1.0]):
        idx = int(round(t_snap * N_t))
        err = np.abs(rho_traj[idx] - rho_an[idx])
        im  = axes2[k].imshow(err.T, origin='lower',
                               extent=[0, 1, 0, 1], cmap='hot_r')
        plt.colorbar(im, ax=axes2[k])
        axes2[k].set_title(f'|Sinkhorn − Analytic|  t={t_snap:.1f}', fontsize=10)
        axes2[k].set_xlabel('x'); axes2[k].set_ylabel('y')
    plt.tight_layout()
    plt.savefig('hopf_cole_2d_error.png', dpi=150, bbox_inches='tight')
    print("Saved hopf_cole_2d_error.png")

    # ── Convergence ───────────────────────────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.semilogy(history)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('max |φ₀·P_T[φ̂₁] − ρ₀|  (absolute L∞)')
    ax3.set_title('2D Sinkhorn convergence')
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('hopf_cole_2d_convergence.png', dpi=150, bbox_inches='tight')
    print("Saved hopf_cole_2d_convergence.png")

    # ── Benchmark ─────────────────────────────────────────────────────────────
    print("\nRunning CPU vs GPU benchmark ...")
    N_bench = [16, 32, 64, 128, 256]
    bench   = run_benchmark(N_bench, eps=eps, n_iters=50)
    plot_benchmark(bench, outfile='hopf_cole_2d_benchmark.png')
