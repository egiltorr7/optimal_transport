#!/usr/bin/env python3
"""
Epsilon-sweep study: at what ε does the Hopf-Cole Sinkhorn start to fail?
And how much does epsilon annealing extend the achievable range?

For each dimension d ∈ {1,2,3} and grid sizes N, sweeps ε and records
the number of iterations to convergence (relative marginal error < tol_rel).
Crosses in the plot mean "did not converge within max_iter".

Row 1: cold-start Sinkhorn
Row 2: epsilon annealing (geometric schedule from ε_start → ε_target)

Key question: does the critical ε depend on N (grid resolution)?
Expected answer: yes — under-resolved Gaussians (small N·σ) behave like
narrower marginals, which are harder to bridge for the same ε.
"""

import numpy as np
from scipy.fft import dctn, idctn
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── Neumann heat kernel (separable DCT-II, any dimension) ────────────────────

def heat_kernel_nd(u, eps, t, dx_list):
    if t == 0.0:
        return u.copy()
    ndim = len(u.shape)
    lam  = np.zeros(u.shape)
    for axis, dx in enumerate(dx_list):
        N     = u.shape[axis]
        k     = np.arange(N, dtype=float)
        lam_k = (4.0 / dx**2) * np.sin(np.pi * k / (2.0 * N))**2
        sl = [None] * ndim; sl[axis] = slice(None)
        lam = lam + lam_k[tuple(sl)]
    decay = np.exp(-0.5 * eps * t * lam)
    return idctn(dctn(u, type=2, norm='ortho') * decay, type=2, norm='ortho')


def log_heat_kernel_nd(v, eps, t, dx_list):
    """Numerically stable log(P_t[exp(v)]) via log-sum-exp trick."""
    if t == 0.0:
        return v.copy()
    v_max  = float(np.max(v))
    result = heat_kernel_nd(np.exp(v - v_max), eps, t, dx_list)
    return v_max + np.log(np.maximum(result, 1e-300))


# ── Sinkhorn cold-start (returns iteration count) ─────────────────────────────

def sinkhorn_count(rho0, rho1, eps, dx_list, max_iter, tol_rel=1e-6):
    """
    Run log-domain Sinkhorn; return (n_iters, final_rel_err).
    n_iters == max_iter means the method did NOT converge.
    Convergence criterion: max|φ₀·P_T[φ̂₁] − ρ₀| / ρ₀.max() < tol_rel.
    """
    tol      = tol_rel * float(rho0.max())
    log_rho0 = np.log(np.maximum(rho0, 1e-300))
    log_rho1 = np.log(np.maximum(rho1, 1e-300))

    u   = np.zeros_like(rho0)
    v   = log_rho1.copy()
    lhv = log_heat_kernel_nd(v, eps, 1.0, dx_list)
    rel_err = np.inf

    for it in range(max_iter):
        u   = log_rho0 - lhv
        lhu = log_heat_kernel_nd(u, eps, 1.0, dx_list)
        v   = log_rho1 - lhu
        lhv = log_heat_kernel_nd(v, eps, 1.0, dx_list)

        err     = float(np.max(np.abs(np.exp(np.clip(u + lhv, -500., 500.)) - rho0)))
        rel_err = err / float(rho0.max())
        if rel_err < tol_rel:
            return it + 1, rel_err

    return max_iter, rel_err


# ── Sinkhorn with epsilon annealing ──────────────────────────────────────────

def sinkhorn_annealed(rho0, rho1, eps_target, dx_list, max_iter,
                      eps_start=0.5, n_levels=8, tol_rel=1e-6):
    """
    Geometric epsilon schedule: eps_start → ... → eps_target.
    Warm-start (u, v) from previous level.
    Returns (total_iters, final_rel_err).
    total_iters == max_iter * n_levels means nothing converged.
    """
    log_rho0 = np.log(np.maximum(rho0, 1e-300))
    log_rho1 = np.log(np.maximum(rho1, 1e-300))

    # Geometric schedule
    eps_schedule = np.geomspace(eps_start, eps_target, n_levels)

    # Initialize potentials
    u   = np.zeros_like(rho0)
    v   = log_rho1.copy()

    total_iters = 0

    for eps in eps_schedule:
        # Re-compute lhv at the new eps
        lhv = log_heat_kernel_nd(v, eps, 1.0, dx_list)
        rel_err = np.inf

        for it in range(max_iter):
            u   = log_rho0 - lhv
            lhu = log_heat_kernel_nd(u, eps, 1.0, dx_list)
            v   = log_rho1 - lhu
            lhv = log_heat_kernel_nd(v, eps, 1.0, dx_list)

            total_iters += 1
            err     = float(np.max(np.abs(np.exp(np.clip(u + lhv, -500., 500.)) - rho0)))
            rel_err = err / float(rho0.max())
            if rel_err < tol_rel:
                break  # converged at this level, move to next eps

    return total_iters, rel_err


# ── Problem setup ─────────────────────────────────────────────────────────────

def build_problem(d, N, sigma=0.05, mu0_val=0.25, mu1_val=0.75):
    dx    = 1.0 / N
    x     = (np.arange(N) + 0.5) * dx
    grids = np.meshgrid(*([x] * d), indexing='ij')
    r0    = sum((g - mu0_val)**2 for g in grids)
    r1    = sum((g - mu1_val)**2 for g in grids)
    rho0  = np.exp(-0.5 * r0 / sigma**2); rho0 /= rho0.sum() * dx**d
    rho1  = np.exp(-0.5 * r1 / sigma**2); rho1 /= rho1.sum() * dx**d
    return rho0, rho1, [dx] * d


# ── Sweep (both cold-start and annealed) ──────────────────────────────────────

def sweep(d, N_list, eps_list, sigma=0.05, max_iter=200, tol_rel=1e-6,
          anneal_levels=8, eps_anneal_start=0.5):
    results_cold   = {}
    results_anneal = {}

    for N in N_list:
        rho0, rho1, dx_list = build_problem(d, N, sigma)
        peak          = float(rho0.max())
        pts_per_sigma = sigma * N
        print(f"  d={d} N={N:4d}  peak={peak:8.1f}  pts/σ={pts_per_sigma:.1f}", end='', flush=True)

        n_cold, err_cold     = [], []
        n_anneal, err_anneal = [], []
        t0 = time.perf_counter()

        for eps in eps_list:
            # Cold start
            n_it, rel_e = sinkhorn_count(rho0, rho1, eps, dx_list, max_iter, tol_rel)
            n_cold.append(n_it); err_cold.append(rel_e)

            # Annealed
            n_it2, rel_e2 = sinkhorn_annealed(
                rho0, rho1, eps, dx_list,
                max_iter     = max_iter,
                eps_start    = eps_anneal_start,
                n_levels     = anneal_levels,
                tol_rel      = tol_rel,
            )
            n_anneal.append(n_it2); err_anneal.append(rel_e2)

        elapsed = time.perf_counter() - t0
        n_conv_cold   = sum(n < max_iter                       for n in n_cold)
        n_conv_anneal = sum(n < max_iter * anneal_levels       for n in n_anneal)
        print(f"  {elapsed:.1f}s  cold={n_conv_cold}/{len(eps_list)}  "
              f"anneal={n_conv_anneal}/{len(eps_list)}")

        results_cold[N]   = np.array(n_cold),   np.array(err_cold)
        results_anneal[N] = np.array(n_anneal),  np.array(err_anneal)

    return results_cold, results_anneal


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_study(all_cold, all_anneal, N_lists, eps_list, max_iters,
               anneal_levels, tol_rel, sigma, outfile='eps_study.png'):
    dims = [1, 2, 3]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    row_labels = ['Cold start', f'Annealing ({anneal_levels} levels, ε₀=0.5)']

    for col, (d, N_list, max_iter) in enumerate(zip(dims, N_lists, max_iters)):
        colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(N_list)))
        budget = max_iter * anneal_levels  # total budget for annealed

        for row, (results, cap, label_cap) in enumerate([
            (all_cold[d],   max_iter, max_iter),
            (all_anneal[d], budget,   budget),
        ]):
            ax = axes[row, col]

            for c, N in zip(colors, N_list):
                n_iters, _ = results[N]
                conv = n_iters < cap

                if conv.any():
                    ax.loglog(eps_list[conv], n_iters[conv], 'o-',
                              color=c, label=f'N={N}', lw=1.8, ms=5, zorder=3)
                if (~conv).any():
                    ax.loglog(eps_list[~conv], np.full((~conv).sum(), cap), 'x',
                              color=c, ms=10, mew=2.5, zorder=3,
                              label=f'N={N}' if not conv.any() else None)

            ax.axhline(cap, color='k', ls='--', lw=1.2, alpha=0.6,
                       label=f'budget={cap}')
            ax.set_xlabel('ε', fontsize=12)
            if col == 0:
                ax.set_ylabel('Total iterations', fontsize=10)
            ax.set_title(f'{d}D  —  {row_labels[row]}', fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, which='both')
            ax.set_ylim(0.7, cap * 5)

    fig.suptitle(
        f'Hopf-Cole Sinkhorn: cold-start vs epsilon annealing  '
        f'(σ=0.05,  tol={tol_rel:.0e},  μ₀=0.25→μ₁=0.75)\n'
        '●─● converged,   ✕ = did not converge within budget',
        fontsize=11)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"\nSaved {outfile}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    sigma         = 0.05
    tol_rel       = 1e-6
    anneal_levels = 8
    eps_ann_start = 0.5

    # ε from 0.003 to 0.5, log-spaced
    eps_list = np.logspace(np.log10(0.003), np.log10(0.5), 22)

    cfg = {
        1: dict(N_list=[32, 64, 128, 256], max_iter=500),
        2: dict(N_list=[32, 64, 128, 256], max_iter=300),
        3: dict(N_list=[16, 32, 64],       max_iter=100),
    }

    all_cold   = {}
    all_anneal = {}

    for d, kw in cfg.items():
        print(f"\n=== {d}D sweep ===")
        t0 = time.perf_counter()
        cold, anneal = sweep(
            d, kw['N_list'], eps_list, sigma,
            kw['max_iter'], tol_rel,
            anneal_levels = anneal_levels,
            eps_anneal_start = eps_ann_start,
        )
        all_cold[d]   = cold
        all_anneal[d] = anneal
        print(f"  total {d}D: {time.perf_counter()-t0:.1f}s")

    plot_study(
        all_cold, all_anneal,
        N_lists      = [cfg[d]['N_list']  for d in [1, 2, 3]],
        eps_list     = eps_list,
        max_iters    = [cfg[d]['max_iter'] for d in [1, 2, 3]],
        anneal_levels= anneal_levels,
        tol_rel      = tol_rel,
        sigma        = sigma,
    )
