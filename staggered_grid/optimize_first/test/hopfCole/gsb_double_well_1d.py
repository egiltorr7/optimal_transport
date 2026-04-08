#!/usr/bin/env python3
"""
Generalized Schrödinger Bridge (gSB) in 1D with external potential.

Implements Section 5 of schrodinger_bridge_hopf_cole.tex:

  Density factorization:
    rho(t,x) = eta_fw(t,x) * eta_bw(t,x) * exp(U(x)/gamma + 1)

  where eta_fw, eta_bw satisfy the Fokker-Planck PDE:
    partial_t eta = gamma * eta_xx + (eta * U_x)_x     (no-flux BCs)

  whose semigroup is  P_t^U  (replaces the plain heat semigroup when U=0).

  Generalized Sinkhorn updates:
    eta_fw0 <- rho0 / (P_1^U[eta_bw1] * exp(U/gamma+1))
    eta_bw1 <- rho1 / (P_1^U[eta_fw0] * exp(U/gamma+1))

Potential: double-well  U(x) = kappa * ((x-0.5)^2 - a^2)^2
  Minima at x = 0.5 ± a.  With a=0.25: wells at x=0.25 and x=0.75.
  Stationary measure  pi(x) ~ exp(-U/gamma)  is bimodal.

Physical picture: particles start in the left well (rho0 at x=0.25) and
end in the right well (rho1 at x=0.75), with a diffusive barrier at x=0.5.
The gSB gives the most likely stochastic path under this energy landscape,
which "hugs" the wells and crosses the barrier as a coherent lump.
"""

import numpy as np
from scipy.sparse import diags
from scipy.linalg import expm
from scipy.fft import dct, idct
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ── Potential ──────────────────────────────────────────────────────────────────

def double_well(x, kappa=30.0, a=0.25):
    """
    U(x) = kappa * ((x-0.5)^2 - a^2)^2.
    Minima at x = 0.5 ± a (value 0); barrier at x=0.5 (height kappa*a^4).
    """
    z  = (x - 0.5)**2 - a**2
    U  = kappa * z**2
    dU = 4.0 * kappa * (x - 0.5) * z       # d U / d x
    return U, dU


# ── FP matrix: Scharfetter-Gummel scheme, no-flux BCs ────────────────────────

def _bernoulli(z):
    """B(z) = z / (exp(z) - 1), numerically stable for all z."""
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(np.abs(z) < 1e-10, 1.0, z / np.expm1(z))


def build_fp_matrix(dx, gamma, U):
    """
    Scharfetter-Gummel (exponential-fitting) discretization of the FP operator:
        L_U eta = gamma * eta_xx + (eta * U_x)_x,   no-flux BCs.

    Face flux:  F_{j+1/2} = (gamma/dx) * [B(-alpha_j)*eta_{j+1} - B(alpha_j)*eta_j]
    where  alpha_j = (U_{j+1} - U_j) / gamma  and  B(z) = z / (e^z - 1).

    Properties guaranteed by this scheme:
      - All off-diagonals > 0, diagonal < 0  (M-matrix => eigenvalues <= 0)
      - Column sums = 0  (exact mass conservation)
      - Discrete stationary state:  pi_j = C * exp(-U_j/gamma)  (L_U pi = 0 exactly)
      - Stable for all grid Péclet numbers  |dU*dx/gamma|
    """
    N     = len(U)
    g     = gamma / dx**2
    alpha = (U[1:] - U[:-1]) / gamma    # shape N-1, signed potential step per face

    bm = _bernoulli( alpha)    # B( alpha_j), length N-1
    bp = _bernoulli(-alpha)    # B(-alpha_j) = exp(alpha_j)*B(alpha_j), length N-1

    lo   = g * bm              # lower diag, length N-1  (all > 0)
    up   = g * bp              # upper diag, length N-1  (all > 0)

    diag = np.empty(N)
    diag[0]    = -g * bm[0]
    diag[1:-1] = -g * (bm[1:] + bp[:-1])
    diag[-1]   = -g * bp[-1]

    return diags([lo, diag, up], [-1, 0, 1], shape=(N, N), format='csr')


# ── Standard heat semigroup (U=0) via DCT-II ─────────────────────────────────

def heat_neumann(u, gamma, t, dx):
    """Apply exp(t * gamma * Delta_Neumann) to u via DCT-II."""
    if t == 0.0:
        return u.copy()
    N   = len(u)
    k   = np.arange(N, dtype=float)
    lam = 4.0 / dx**2 * np.sin(np.pi * k / (2.0 * N))**2
    return idct(dct(u, type=2, norm='ortho') * np.exp(-gamma * t * lam),
                type=2, norm='ortho')


# ── Sinkhorn: standard SB (U=0) ───────────────────────────────────────────────

def sinkhorn_standard(rho0, rho1, gamma, dx, max_iter=3000, tol=1e-10):
    """Log-domain Sinkhorn for the plain SB (no potential)."""
    log_rho0 = np.log(np.maximum(rho0, 1e-300))
    log_rho1 = np.log(np.maximum(rho1, 1e-300))

    def log_heat(w):
        c = float(w.max())
        return c + np.log(np.maximum(heat_neumann(np.exp(w - c), gamma, 1.0, dx), 1e-300))

    u, v = np.zeros_like(rho0), log_rho1.copy()
    lhv  = log_heat(v)

    for it in range(max_iter):
        u_new = log_rho0 - lhv
        lhu   = log_heat(u_new)
        v_new = log_rho1 - lhu
        lhv   = log_heat(v_new)
        err   = np.max(np.abs(u_new - u)) + np.max(np.abs(v_new - v))
        u, v  = u_new, v_new
        if err < tol:
            print(f"  Standard SB converged at iter {it+1},  Δpot={err:.2e}")
            break
    else:
        print(f"  Standard SB: did not converge ({max_iter} iters), Δpot={err:.2e}")

    return np.exp(u), np.exp(v)


# ── Sinkhorn: generalized SB ──────────────────────────────────────────────────

def sinkhorn_gsb(rho0, rho1, gamma, U, prop_full, dx,
                 max_iter=3000, tol=1e-10, log_every=200):
    """
    Generalized Hopf-Cole Sinkhorn (Algorithm 2 in the tex).

    prop_full : dense N×N matrix  expm(L_U)  (precomputed P_1^U).
    Returns   : eta_fw0, eta_bw1, convergence history.
    """
    log_rho0 = np.log(np.maximum(rho0, 1e-300))
    log_rho1 = np.log(np.maximum(rho1, 1e-300))
    log_eUg1 = U / gamma + 1.0              # log(exp(U/gamma+1))

    def log_fp(w):
        """log( P_1^U[ exp(w) ] ) with log-sum-exp shift for stability."""
        c = float(w.max())
        return c + np.log(np.maximum(prop_full @ np.exp(w - c), 1e-300))

    # Initialisation: eta_fw0=1, eta_bw1 = rho1/exp(U/gamma+1)
    u   = np.zeros_like(rho0)
    v   = log_rho1 - log_eUg1
    lhv = log_fp(v)
    history = []

    for it in range(max_iter):
        u_new = log_rho0 - lhv - log_eUg1
        lhu   = log_fp(u_new)
        v_new = log_rho1 - lhu - log_eUg1
        lhv   = log_fp(v_new)

        err = np.max(np.abs(u_new - u)) + np.max(np.abs(v_new - v))
        history.append(float(err))
        u, v = u_new, v_new

        if it % log_every == 0:
            r0 = np.exp(u + lhv + log_eUg1)
            r1 = np.exp(lhu + v + log_eUg1)
            e0 = np.max(np.abs(r0 - rho0))
            e1 = np.max(np.abs(r1 - rho1))
            print(f"  it={it:5d}  Δpot={err:.2e}  marg0={e0:.2e}  marg1={e1:.2e}")

        if err < tol:
            print(f"  gSB converged at iter {it+1},  Δpot={err:.2e}")
            break
    else:
        print(f"  gSB: did not converge ({max_iter} iters),  Δpot={err:.2e}")

    return np.exp(u), np.exp(v), history


# ── Trajectories ──────────────────────────────────────────────────────────────

def build_trajectory_gsb(eta_fw0, eta_bw1, gamma, U, prop_step, N_t=64):
    """
    rho(t_i, x) = P_{t_i}^U[eta_fw0] * P_{1-t_i}^U[eta_bw1] * exp(U/gamma+1)
    Uses prop_step = expm((1/N_t)*L_U) applied iteratively.
    """
    N    = len(eta_fw0)
    eUg1 = np.exp(U / gamma + 1.0)

    eta_fw      = np.zeros((N_t + 1, N))
    eta_bw      = np.zeros((N_t + 1, N))
    eta_fw[0]   = eta_fw0.copy()
    eta_bw[-1]  = eta_bw1.copy()
    for i in range(1, N_t + 1):
        eta_fw[i] = prop_step @ eta_fw[i - 1]
    for i in range(N_t - 1, -1, -1):
        eta_bw[i] = prop_step @ eta_bw[i + 1]

    rho    = eta_fw * eta_bw * eUg1[np.newaxis, :]
    t_vals = np.linspace(0.0, 1.0, N_t + 1)
    return rho, t_vals


def build_trajectory_std(phi0, hat_phi1, gamma, dx, N_t=64):
    """Standard SB trajectory (U=0)."""
    N      = len(phi0)
    t_vals = np.linspace(0.0, 1.0, N_t + 1)
    rho    = np.zeros((N_t + 1, N))
    for i, t in enumerate(t_vals):
        rho[i] = (heat_neumann(phi0,      gamma, t,       dx) *
                  heat_neumann(hat_phi1,  gamma, 1.0 - t, dx))
    return rho, t_vals


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Grid and parameters ───────────────────────────────────────────────────
    N      = 128
    dx     = 1.0 / N
    x      = (np.arange(N) + 0.5) * dx

    gamma  = 0.02    # diffusion coefficient (= eps/2 in tex sections 1–4)
    kappa  = 30.0    # double-well steepness
    a_well = 0.25    # half-separation of wells; minima at x=0.25 and x=0.75
    sigma  = 0.05    # marginal std dev
    N_t    = 64

    mu0, mu1 = 0.5 - a_well, 0.5 + a_well   # 0.25, 0.75

    # ── Marginals ─────────────────────────────────────────────────────────────
    rho0 = np.exp(-0.5 * (x - mu0)**2 / sigma**2)
    rho1 = np.exp(-0.5 * (x - mu1)**2 / sigma**2)
    rho0 /= rho0.sum() * dx
    rho1 /= rho1.sum() * dx

    # ── Potential ─────────────────────────────────────────────────────────────
    U, dU    = double_well(x, kappa=kappa, a=a_well)
    barrier  = kappa * a_well**4            # height of the central barrier
    pi_stat  = np.exp(-U / gamma)
    pi_stat /= pi_stat.sum() * dx

    print("=" * 60)
    print(f"Double-well potential: kappa={kappa}, a={a_well}")
    print(f"  Barrier height: {barrier:.4f}  =  {barrier/gamma:.1f} x gamma")
    print(f"  Stationary measure: bimodal, peaks at x=0.25 and x=0.75")
    print("=" * 60)

    # ── Build FP operator and precompute propagators ──────────────────────────
    # Pass U (not dU): SG scheme uses finite differences of U at cell centers
    L_U       = build_fp_matrix(dx, gamma, U)

    print("Building propagators (matrix exponentials)...")
    prop_full = expm(L_U.toarray())                     # P_1^U  (N×N)
    prop_step = expm((1.0 / N_t) * L_U.toarray())      # P_{dt}^U for trajectory

    # Sanity: discrete stationary state should satisfy L_U pi = 0 exactly
    pi_disc   = np.exp(-U / gamma)           # unnormalised, exact discrete stationary
    res_pi    = np.linalg.norm(L_U @ pi_disc)
    print(f"  ||L_U pi_disc|| = {res_pi:.2e}  (SG scheme: should be ~machine epsilon)")

    # Sanity: prop_full should preserve total mass (1^T prop_full = 1^T)
    rng      = np.random.default_rng(0)
    eta_test = np.abs(rng.standard_normal(N)) + 0.1
    mass_err = abs((prop_full @ eta_test).sum() - eta_test.sum()) / eta_test.sum()
    print(f"  Relative mass conservation of prop_full: {mass_err:.2e}")

    # ── gSB Sinkhorn ──────────────────────────────────────────────────────────
    print("\nRunning generalized SB Sinkhorn...")
    eta_fw0, eta_bw1, history_gsb = sinkhorn_gsb(
        rho0, rho1, gamma, U, prop_full, dx,
        max_iter=3000, tol=1e-10, log_every=200)

    # ── Standard SB (U=0) ────────────────────────────────────────────────────
    print("\nRunning standard SB Sinkhorn (U=0)...")
    phi0, hat_phi1 = sinkhorn_standard(rho0, rho1, gamma, dx)

    # ── Build trajectories ────────────────────────────────────────────────────
    print("\nBuilding trajectories...")
    rho_gsb, t_vals = build_trajectory_gsb(eta_fw0, eta_bw1, gamma, U, prop_step, N_t)
    rho_std, _      = build_trajectory_std(phi0, hat_phi1, gamma, dx, N_t)

    # ── Verification ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Verification report")
    print("=" * 60)

    eUg1 = np.exp(U / gamma + 1.0)
    r0   = eta_fw0 * (prop_full @ eta_bw1) * eUg1
    r1   = (prop_full @ eta_fw0) * eta_bw1 * eUg1
    e0   = np.max(np.abs(r0 - rho0))
    e1   = np.max(np.abs(r1 - rho1))
    print(f"  gSB marginal rho0 error (L∞): {e0:.2e}  {'OK' if e0 < 1e-6 else 'WARN'}")
    print(f"  gSB marginal rho1 error (L∞): {e1:.2e}  {'OK' if e1 < 1e-6 else 'WARN'}")

    masses_gsb = rho_gsb.sum(axis=1) * dx
    masses_std = rho_std.sum(axis=1) * dx
    print(f"  gSB mass deviation:  {masses_gsb.max() - masses_gsb.min():.2e}")
    print(f"  SB  mass deviation:  {masses_std.max() - masses_std.min():.2e}")
    print(f"  Min rho_gsb: {rho_gsb.min():.2e}  "
          f"({'OK' if rho_gsb.min() > -1e-12 else 'negative!'})")
    print("=" * 60)

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # 1. gSB heatmap
    ax = axes[0, 0]
    im = ax.imshow(rho_gsb, aspect='auto', origin='lower',
                   extent=[0, 1, 0, 1], cmap='viridis')
    plt.colorbar(im, ax=ax)
    ax.set_title(r'$\rho_\mathrm{gSB}(t,x)$ — double-well $U$', fontsize=12)
    ax.set_xlabel('x'); ax.set_ylabel('t')

    # 2. Standard SB heatmap
    ax = axes[0, 1]
    im = ax.imshow(rho_std, aspect='auto', origin='lower',
                   extent=[0, 1, 0, 1], cmap='viridis')
    plt.colorbar(im, ax=ax)
    ax.set_title(r'$\rho_\mathrm{SB}(t,x)$ — standard ($U=0$)', fontsize=12)
    ax.set_xlabel('x'); ax.set_ylabel('t')

    # 3. Difference
    ax = axes[0, 2]
    diff = rho_gsb - rho_std
    vm   = np.abs(diff).max()
    im   = ax.imshow(diff, aspect='auto', origin='lower',
                     extent=[0, 1, 0, 1], cmap='RdBu_r', vmin=-vm, vmax=vm)
    plt.colorbar(im, ax=ax)
    ax.set_title(r'$\rho_\mathrm{gSB} - \rho_\mathrm{SB}$', fontsize=12)
    ax.set_xlabel('x'); ax.set_ylabel('t')

    # 4. Potential + stationary measure + marginals
    ax   = axes[1, 0]
    ax2  = ax.twinx()
    ax.plot(x, U / U.max(), 'b-', lw=2, label=r'$U(x)$ (norm.)')
    ax2.fill_between(x, pi_stat, alpha=0.2, color='r')
    ax2.plot(x, pi_stat, 'r-',  lw=2, label=r'$\pi \propto e^{-U/\gamma}$')
    ax2.plot(x, rho0,    'g--', lw=1.5, label=r'$\rho_0$')
    ax2.plot(x, rho1,    'm--', lw=1.5, label=r'$\rho_1$')
    ax.set_xlabel('x')
    ax.set_ylabel('$U(x)$ (normalised)', color='b')
    ax2.set_ylabel('density', color='gray')
    ax.set_title(f'Potential, $\\pi$, and marginals\n'
                 f'barrier = {barrier:.3f} = {barrier/gamma:.1f}$\\gamma$', fontsize=10)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc='upper center')

    # 5. Snapshots: gSB vs standard SB
    ax = axes[1, 1]
    snap_times = [0.0, 0.25, 0.5, 0.75, 1.0]
    colors     = plt.cm.plasma(np.linspace(0.1, 0.9, len(snap_times)))
    for t_snap, c in zip(snap_times, colors):
        idx = int(round(t_snap * N_t))
        ax.plot(x, rho_gsb[idx], color=c, lw=2.0, label=f't={t_snap:.2f}')
        ax.plot(x, rho_std[idx], color=c, lw=1.2, ls='--', alpha=0.7)
    leg1 = ax.legend(fontsize=7, loc='upper center', title='time', ncol=3)
    ax.add_artist(leg1)
    ax.legend(handles=[Line2D([0], [0], color='k', lw=2.0, label='gSB (double-well $U$)'),
                        Line2D([0], [0], color='k', lw=1.2, ls='--', label='SB ($U=0$)')],
              fontsize=9, loc='upper right')
    ax.set_title('Snapshots: gSB (solid) vs standard SB (dashed)', fontsize=11)
    ax.set_xlabel('x')

    # 6. Convergence history
    ax = axes[1, 2]
    ax.semilogy(history_gsb, lw=1.5)
    ax.axhline(1e-10, color='k', ls='--', lw=1, alpha=0.5, label='tol=$10^{-10}$')
    ax.set_title('gSB Sinkhorn convergence', fontsize=11)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Potential change ($L^\\infty$)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        rf'Generalized Schrodinger Bridge — double-well $U(x)=\kappa[(x-0.5)^2-a^2]^2$'
        rf'  ($\kappa={kappa}$, $a={a_well}$, $\gamma={gamma}$, $\sigma={sigma}$)',
        fontsize=11)
    plt.tight_layout()
    plt.savefig('gsb_double_well_1d.png', dpi=150, bbox_inches='tight')
    print('\nSaved gsb_double_well_1d.png')
