#!/usr/bin/env python3
"""
Schrödinger Bridge Problem — 1D, Gaussian marginals.

Dynamic formulation (Benamou-Brenier for SBP):
    min_{ρ≥0, u}   ∫₀¹ ∫ ½|u|² ρ dx dt
    s.t.  ∂ₜρ + ∂ₓ(ρu) = (σ²/2) ∂ₓₓρ,   ρ(0)=ρ₀,  ρ(1)=ρ₁

is EQUIVALENT to the static entropic OT problem (proved by Chen, Georgiou, Pavon 2016):
    min_{P≥0}   Σᵢⱼ Pᵢⱼ Cᵢⱼ + ε Σᵢⱼ Pᵢⱼ log Pᵢⱼ
    s.t.  P1 = ρ₀ dx,   Pᵀ1 = ρ₁ dx
where  C = (xᵢ−xⱼ)²/2  and  ε = σ²/2.

Solved by:
  STEP 1 — Sinkhorn iterations (log-domain, exponential convergence)
  STEP 2 — Bridge density reconstruction:
             ρ(t,x) = Σᵢⱼ P*ᵢⱼ · N(x; (1−s)xᵢ+s·xⱼ, σ²s(1−s)T)
             where s = t/T  (Brownian bridge interpolation)
  STEP 3 — Drift reconstruction from continuity equation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import logsumexp

# ── 1. PARAMETERS ────────────────────────────────────────────────────────────
sigma = 0.5          # diffusion of reference  dX = u dt + σ dW
eps   = sigma**2 / 2 # entropic regularisation  ε = σ²/2
T     = 1.0

Nx   = 128           # spatial grid
x    = np.linspace(-4.0, 4.0, Nx)
dx   = x[1] - x[0]

mu0, s0 = -1.5, 0.5  # ρ₀ = N(μ₀, s₀²)
mu1, s1 =  1.5, 0.5  # ρ₁ = N(μ₁, s₁²)

p0 = norm.pdf(x, mu0, s0);  p0 /= p0.sum() * dx
p1 = norm.pdf(x, mu1, s1);  p1 /= p1.sum() * dx

# Discrete measures (probability mass):  a_i = p0(x_i)·dx,  Σ a_i = 1
a = p0 * dx
b = p1 * dx

print(f"sigma={sigma}, eps={eps:.4f}, Nx={Nx}, dx={dx:.4f}")
print(f"Mass check: Σa={a.sum():.6f}, Σb={b.sum():.6f}")

# ── 2. SINKHORN (log-domain) ─────────────────────────────────────────────────
# Kernel: Kᵢⱼ = exp(−(xᵢ−xⱼ)² / (2εT))  (BM heat kernel at time T)
log_K = -(x[:, None] - x[None, :])**2 / (2 * eps * T)   # (Nx, Nx)

n_sink  = 3000
tol     = 1e-10

log_a = np.log(a)
log_b = np.log(b)
log_v = np.zeros(Nx)          # initialise log v = 0

err_hist = []

for k in range(n_sink):
    # u-step:  log u ← log a − log(K @ v)
    log_Kv  = logsumexp(log_K + log_v[None, :], axis=1)
    log_u   = log_a - log_Kv

    # v-step:  log v ← log b − log(Kᵀ @ u)
    log_KTu = logsumexp(log_K + log_u[:, None], axis=0)
    log_v   = log_b - log_KTu

    # Marginal error (check every 50 iters)
    if k % 50 == 0:
        log_row = log_u + logsumexp(log_K + log_v[None, :], axis=1)
        log_col = log_v + logsumexp(log_K + log_u[:, None], axis=0)
        err = max(np.abs(np.exp(log_row) - a).max(),
                  np.abs(np.exp(log_col) - b).max())
        err_hist.append((k, err))
        if err < tol:
            print(f"Sinkhorn converged at iter {k+1}, err={err:.2e}")
            break

iters_hist = [e[0] for e in err_hist]
err_vals   = [e[1] for e in err_hist]
print(f"Final marginal error: {err_vals[-1]:.2e}")

# Optimal coupling  P*ᵢⱼ = uᵢ Kᵢⱼ vⱼ   (probability mass, Σᵢⱼ P*ᵢⱼ = 1)
log_P = log_u[:, None] + log_K + log_v[None, :]
P     = np.exp(log_P)          # (Nx, Nx)

print(f"Coupling check: Σ P = {P.sum():.6f}")
print(f"  row-marginal err: {np.abs(P.sum(1) - a).max():.2e}")
print(f"  col-marginal err: {np.abs(P.sum(0) - b).max():.2e}")

# ── 3. BRIDGE DENSITY RECONSTRUCTION ────────────────────────────────────────
# ρ(t, xₖ) = Σᵢⱼ P*ᵢⱼ · N(xₖ; (1−s)xᵢ + s·xⱼ, σ²s(1−s)T)
#
# Vectorised: G[k, (i,j)] = N(xₖ; Mᵢⱼ, v_bridge)   shape (Nx, Nx²)
#             ρ[t] = G @ P.ravel()
#
# This exactly integrates the Brownian bridge kernel over the coupling P.

Nt  = 60
t   = np.linspace(0, T, Nt)
rho = np.zeros((Nt, Nx))

P_flat = P.ravel()                              # (Nx²,)
x_flat = x                                      # (Nx,)

for ti, tv in enumerate(t):
    s = tv / T

    if s < 1e-8:          # t = 0
        rho[ti] = p0
        continue
    if s > 1 - 1e-8:      # t = T
        rho[ti] = p1
        continue

    v_br  = sigma**2 * s * (1 - s) * T          # bridge variance
    std   = np.sqrt(v_br)
    M_flat = ((1-s)*x[:, None] + s*x[None, :]).ravel()   # (Nx²,)

    # G[k, n] = N(x[k]; M_flat[n], std)   shape (Nx, Nx²)
    diff  = x_flat[:, None] - M_flat[None, :]   # (Nx, Nx²)
    G     = np.exp(-diff**2 / (2*v_br)) / (std * np.sqrt(2*np.pi))

    rho[ti] = G @ P_flat                         # (Nx,)
    # Renormalise (the integral over the finite grid is ≈1 but not exact)
    mass = rho[ti].sum() * dx
    rho[ti] /= mass

# ── 4. DRIFT RECONSTRUCTION from continuity equation ────────────────────────
# ∂ₜρ + ∂ₓ(ρu) = (σ²/2) ∂ₓₓρ   →   ∂ₓm = (σ²/2)∂ₓₓρ − ∂ₜρ
# Integrate: m(t,x) = ∫_{x_min}^x [(σ²/2)∂yyρ − ∂ₜρ] dy   (zero-flux BC)

dt_num   = t[1] - t[0]
drho_dt  = np.gradient(rho, dt_num, axis=0)
drho_dxx = np.gradient(np.gradient(rho, dx, axis=1), dx, axis=1)

rhs  = (sigma**2 / 2) * drho_dxx - drho_dt   # ∂ₓ(ρu)
m    = np.cumsum(rhs, axis=1) * dx            # ρ·u  (momentum density)
u_dr = m / np.maximum(rho, 1e-6)             # drift velocity u = m/ρ

# ── 5. PLOTS ─────────────────────────────────────────────────────────────────
T_grid, X_grid = np.meshgrid(t, x, indexing='ij')

# ─── Figure 1: overview ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

ax = axes[0, 0]
im = ax.contourf(X_grid, T_grid, rho, levels=50, cmap='viridis')
plt.colorbar(im, ax=ax)
ax.set_title(r'Density  $\rho(t,x)$', fontsize=13)
ax.set_xlabel('x');  ax.set_ylabel('t')

ax = axes[0, 1]
vc = np.percentile(np.abs(u_dr), 97)
im2 = ax.contourf(X_grid, T_grid, u_dr, levels=50, cmap='RdBu_r',
                  vmin=-vc, vmax=vc)
plt.colorbar(im2, ax=ax)
ax.set_title(r'Drift field  $u(t,x)$', fontsize=13)
ax.set_xlabel('x');  ax.set_ylabel('t')

ax = axes[1, 0]
ax.plot(x, rho[0],   lw=2.5, c='steelblue', label=r'$\rho(0,x)$ computed')
ax.plot(x, p0, '--', lw=2,   c='steelblue', alpha=0.5, label=r'$\rho_0$ target')
ax.plot(x, rho[-1],  lw=2.5, c='tomato',    label=r'$\rho(1,x)$ computed')
ax.plot(x, p1, '--', lw=2,   c='tomato',    alpha=0.5, label=r'$\rho_1$ target')
ax.legend(fontsize=9);  ax.set_xlabel('x')
ax.set_title('Boundary marginals (should overlap)', fontsize=12)

ax = axes[1, 1]
ax.semilogy(iters_hist, err_vals, 'b-o', ms=3, lw=1.5, label='max marginal error')
ax.axhline(tol, ls='--', c='gray', label=f'tol={tol:.0e}')
ax.legend(fontsize=9);  ax.set_xlabel('Sinkhorn iteration')
ax.set_title('Convergence', fontsize=12)
ax.set_ylabel(r'$\max(\|\hat\rho_0 - \rho_0\|, \|\hat\rho_1 - \rho_1\|)$')

plt.suptitle(
    rf'Schrödinger Bridge  ($\sigma={sigma}$, $\varepsilon=\sigma^2/2={eps:.3f}$,'
    rf' $\rho_0=\mathcal{{N}}({mu0},{s0}^2)$, $\rho_1=\mathcal{{N}}({mu1},{s1}^2)$)',
    fontsize=11)
plt.tight_layout()
plt.savefig('sbp.png', dpi=150, bbox_inches='tight')
print("Saved sbp.png")

# ─── Figure 2: density evolution ─────────────────────────────────────────────
n_show = 9
t_idx  = np.round(np.linspace(0, Nt-1, n_show)).astype(int)
colors = plt.cm.plasma(np.linspace(0.05, 0.92, n_show))

fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))

ax = axes2[0]
for k, i in enumerate(t_idx):
    ax.plot(x, rho[i], color=colors[k], lw=2, label=rf'$t={t[i]:.2f}$')
ax.set_xlabel('x', fontsize=12);  ax.set_ylabel(r'$\rho(t,x)$', fontsize=12)
ax.set_title(r'Density slices  $\rho(t,x)$', fontsize=13)
ax.legend(fontsize=8, ncol=2)

ax = axes2[1]
im3 = ax.pcolormesh(x, t, rho, cmap='viridis', shading='auto')
plt.colorbar(im3, ax=ax, label=r'$\rho$')
ax.set_xlabel('x', fontsize=12);  ax.set_ylabel('t', fontsize=12)
ax.set_title(r'Density heatmap  $\rho(t,x)$', fontsize=13)

plt.suptitle(
    rf'Density evolution — Schrödinger Bridge  ($\sigma={sigma}$)',
    fontsize=12)
plt.tight_layout()
plt.savefig('sbp_evolution.png', dpi=150, bbox_inches='tight')
print("Saved sbp_evolution.png")

plt.show()

# ── 6. SUMMARY ───────────────────────────────────────────────────────────────
ke = float(np.sum(m**2 / np.maximum(rho, 1e-8)) * dx * (t[1]-t[0]))
print(f"\nKinetic energy  KE = ∫∫ m²/(2ρ) dx dt ≈ {ke:.4f}")
print(f"OT cost (σ→0 limit): (μ₁−μ₀)²/2 = {(mu1-mu0)**2/2:.4f}")
print(f"(KE < OT cost: diffusion reduces transport cost)")
