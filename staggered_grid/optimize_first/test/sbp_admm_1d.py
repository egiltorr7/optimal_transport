#!/usr/bin/env python3
"""
1D Schrödinger Bridge — ADMM on the dynamic (Benamou-Brenier) formulation.
Structured explicitly to generalise to 3D (see NOTE comments).

Dynamic SBP:
    min  Σᵢⱼ  mᵢⱼ² / (2 ρᵢⱼ)          [normalised kinetic energy]
    s.t. ρᵢ₊₁ − A ρᵢ + B mᵢ = 0        [explicit-Euler continuity, i=0…Nt-2]
         ρ₀    = rho0                     [left BC]
         ρ_{Nt-1} = rho1                  [right BC]

ADMM (ρₓ,mₓ = prox variable; ρ_z,m_z = constraint variable):
    Step 1  prox_kinetic   :  (ρₓ,mₓ) = argmin m²/(2ρ) + r/2‖(ρ,m)−(ρ_z−u)‖²
                               → pointwise cubic,  solved by Newton  (ρₓ ≥ 0 always)
    Step 2  proj_cont      :  (ρ_z,m_z) = proj onto  {Cz = b}
                               → 1D: sparse KKT factorisation (pre-computed once)
                               → 3D: replace with preconditioned CG (see NOTE)
    Step 3  dual update    :  u ← u + (ρₓ,mₓ) − (ρ_z,m_z)
    + adaptive penalty (Boyd et al. 2010)

Reference solution u_* = (ρ_*, m_*) from Sinkhorn (exact for SBP):
    ρ_* : Brownian-bridge interpolation of the optimal coupling P*
    m_* : momentum ρ_* u_* reconstructed from the continuity equation
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')           # non-interactive backend — no display needed
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.sparse import diags, eye as speye, lil_matrix
from scipy.sparse.linalg import factorized
from scipy.special import logsumexp

# ─────────────────────────────────────────────────────────────────────────────
# 1.  PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
sigma = 0.3          # reference SDE diffusion  dX = u dt + σ dW
mu0, s0 = -1.5, 0.5  # ρ₀ = N(μ₀, s₀²)
mu1, s1 =  1.5, 0.5  # ρ₁ = N(μ₁, s₁²)
T     = 1.0

# Grid — choose Nt so that CFL = |v_max|·Δt/Δx < 1
# For this problem |v_max| ≈ |μ₁−μ₀|/T = 3;  need Δt/Δx < 1/3.
Nt = 48       # time slices
Nx = 64       # spatial points
x  = np.linspace(-4.0, 4.0, Nx)
t  = np.linspace(0.0, T, Nt)
dx = x[1] - x[0]
dt = t[1] - t[0]
print(f"Grid  Nt={Nt}, Nx={Nx},  dx={dx:.4f}, dt={dt:.4f}")
print(f"CFL   {abs(mu1-mu0)*dt/dx:.3f}  (must be < 1)")

# ADMM hyper-parameters
r0        = 5.0     # initial penalty
n_iter    = 3000    # max iterations
mu_ada    = 5.0     # adaptive-penalty trigger ratio
tau_ada   = 2.0     # adaptive-penalty scale factor
tol_rel   = 1e-5    # relative stopping tolerance (tight — ensures ADMM ≈ u_*)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  BOUNDARY CONDITIONS
# ─────────────────────────────────────────────────────────────────────────────
rho0 = norm.pdf(x, mu0, s0);  rho0 /= rho0.sum() * dx
rho1 = norm.pdf(x, mu1, s1);  rho1 /= rho1.sum() * dx

# ─────────────────────────────────────────────────────────────────────────────
# 3.  DISCRETE SPATIAL OPERATORS
#     NOTE (3D): replace with Kronecker products  Dₓ⊗Iy⊗Iz,  Iy⊗Dy⊗Iz, etc.
# ─────────────────────────────────────────────────────────────────────────────
def make_Dx(N, h):
    D = diags([-np.ones(N-1), np.ones(N-1)], [-1, 1],
              shape=(N, N), format='lil') / (2*h)
    D[0, 0] = -1/h;   D[0, 1]  =  1/h
    D[-1,-2] = -1/h;  D[-1,-1] =  1/h
    return D.tocsr()

def make_Dxx(N, h):
    D = diags([np.ones(N-1), -2*np.ones(N), np.ones(N-1)],
              [-1, 0, 1], shape=(N, N), format='lil') / h**2
    D[0, 0] = -1/h**2;   D[0, 1]  =  1/h**2
    D[-1,-2] =  1/h**2;  D[-1,-1] = -1/h**2
    return D.tocsr()

Dx  = make_Dx(Nx, dx)
Dxx = make_Dxx(Nx, dx)

# Explicit-Euler operators:   ρᵢ₊₁ = A_fwd ρᵢ  −  B_div mᵢ
A_fwd = (speye(Nx) + dt * (sigma**2 / 2) * Dxx).tocsr()   # diffusion
B_div = (dt * Dx).tocsr()                                   # divergence

# ─────────────────────────────────────────────────────────────────────────────
# 4.  CONSTRAINT MATRIX  C z = b_c
#     z = [ρ₀,…,ρ_{Nt-1},  m₀,…,m_{Nt-1}]  ∈ ℝ^{2·Nt·Nx}
#
#     NOTE (3D): C has the same block structure but with Kronecker spatial ops.
#               For large 3D grids, replace the direct KKT solve below with
#               a preconditioned CG on the normal equations  CCᵀ λ = C a − b.
#               A good preconditioner is a block-diagonal approximation of CCᵀ.
# ─────────────────────────────────────────────────────────────────────────────
# No-flux BCs for m at spatial boundaries (2*Nt constraints):
#   m[i, 0] = 0  and  m[i, Nx-1] = 0  for all i.
# Without these, mass can enter/exit through the domain walls, and the
# objective is trivially "minimised" by spreading ρ to ∞ — wrong physics.
n_con = (Nt + 1) * Nx + 2 * Nt
n_var = 2 * Nt * Nx

A_a = A_fwd.toarray();  B_a = B_div.toarray();  I_a = speye(Nx).toarray()

C   = lil_matrix((n_con, n_var))
b_c = np.zeros(n_con)

# Density BC rows
C[0:Nx,          0:Nx]             = I_a;  b_c[0:Nx]    = rho0
C[Nx:2*Nx, (Nt-1)*Nx:Nt*Nx]       = I_a;  b_c[Nx:2*Nx] = rho1

# Continuity rows
for i in range(Nt - 1):
    row = (2 + i) * Nx
    C[row:row+Nx, (i+1)*Nx:(i+2)*Nx]  =  I_a   # +ρᵢ₊₁
    C[row:row+Nx,     i*Nx:(i+1)*Nx]  = -A_a   # −A_fwd ρᵢ
    cm = Nt*Nx + i*Nx
    C[row:row+Nx, cm:cm+Nx]           =  B_a   # +B_div mᵢ

# No-flux momentum BCs: m[i, 0] = 0,  m[i, Nx-1] = 0
row0 = (Nt + 1) * Nx
for i in range(Nt):
    C[row0 + i,          Nt*Nx + i*Nx]        = 1.0   # m[i,   0  ] = 0
    C[row0 + Nt + i,     Nt*Nx + i*Nx + Nx-1] = 1.0   # m[i, Nx-1 ] = 0

C = C.tocsr()

# Pre-factored sparse solve for CCᵀ  (done ONCE, reused every ADMM iteration)
print("Factorising CCᵀ (sparse) …", end=" ", flush=True)
CCT      = (C @ C.T).tocsc()
CCT_reg  = CCT + 1e-10 * speye(n_con, format='csc')
_CCT_inv = factorized(CCT_reg)     # NOTE (3D): replace with CG
print(f"done  (shape {CCT.shape}, nnz={CCT.nnz})")


def proj_cont(a_rho, a_m):
    """Project (a_rho, a_m) onto {Cz = b_c} via pre-factored KKT."""
    a   = np.r_[a_rho.ravel(), a_m.ravel()]
    lam = _CCT_inv(C @ a - b_c)
    z   = a - C.T @ lam
    return z[:Nt*Nx].reshape(Nt, Nx), z[Nt*Nx:].reshape(Nt, Nx)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PROXIMAL OF KINETIC ENERGY
#
#     f(ρ, m) = Σᵢⱼ  mᵢⱼ² / (2 ρᵢⱼ)       (c = 1, normalised)
#
#     Pointwise sub-problem (given a, b; penalty r):
#         min_{ρ>0, m}  m²/(2ρ)  +  r/2·(ρ−a)²  +  r/2·(m−b)²
#
#     Optimality → m* = r·b·ρ* / (1 + r·ρ*)
#                → cubic:  (ρ−a)·(1 + r·ρ)² = r·b²/2
#
#     Solved by Newton (vectorised over all grid points).
#
#     NOTE (3D): identical — just |m|² = mₓ²+mᵧ²+m_z², same cubic for ρ*,
#               then each component mₖ* = r·bₖ·ρ* / (1+r·ρ*).
# ─────────────────────────────────────────────────────────────────────────────
def prox_kinetic(a_rho, a_m, r, n_newton=30, eps=1e-14):
    a = a_rho.ravel()
    b = a_m.ravel()

    rho_out = np.zeros_like(a)
    m_out   = np.zeros_like(b)

    # b ≈ 0  →  ρ* = max(a, 0),  m* = 0
    zm = np.abs(b) < eps
    rho_out[zm] = np.maximum(a[zm], 0.0)

    # General case: Newton on  g(ρ) = (ρ−a)(1+rρ)² − rb²/2 = 0
    idx = ~zm
    if idx.any():
        ai, bi = a[idx], b[idx]
        rhs = r * bi**2 / 2.0
        rho = np.maximum(ai, eps) + np.abs(bi) / np.sqrt(r + eps)
        for _ in range(n_newton):
            q   = 1.0 + r * rho
            g   = (rho - ai) * q**2 - rhs
            gp  = q**2 + 2.0 * r * (rho - ai) * q
            rho = rho - g / (gp + eps)
            rho = np.maximum(rho, eps)
        rho_out[idx] = rho
        m_out[idx]   = r * bi * rho / (1.0 + r * rho)

    return rho_out.reshape(Nt, Nx), m_out.reshape(Nt, Nx)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  REFERENCE DENSITY  ρ_*  via Sinkhorn (exact for SBP)
#
#     ρ_* : Brownian-bridge interpolation of the optimal coupling P*
#     Used to track  ‖ρ_k − ρ_*‖  during ADMM.
# ─────────────────────────────────────────────────────────────────────────────
print("\nRunning Sinkhorn for reference density ρ_* …")
eps_sink = sigma**2 / 2
log_K    = -(x[:, None] - x[None, :])**2 / (2 * eps_sink * T)
log_a_s  = np.log(rho0 * dx)
log_b_s  = np.log(rho1 * dx)
log_v_s  = np.zeros(Nx)

for _ in range(2000):
    log_u_s = log_a_s - logsumexp(log_K + log_v_s[None, :], axis=1)
    log_v_s = log_b_s - logsumexp(log_K + log_u_s[:, None], axis=0)

log_P_s = log_u_s[:, None] + log_K + log_v_s[None, :]
P_s     = np.exp(log_P_s)
P_flat  = P_s.ravel()

# ρ_* on the ADMM time grid (Brownian-bridge interpolation)
rho_star = np.zeros((Nt, Nx))
for ti, tv in enumerate(t):
    s = tv / T
    if s < 1e-8:   rho_star[ti] = rho0; continue
    if s > 1-1e-8: rho_star[ti] = rho1; continue
    v_br = sigma**2 * s * (1 - s) * T
    M_f  = ((1-s)*x[:, None] + s*x[None, :]).ravel()
    diff = x[:, None] - M_f[None, :]
    G    = np.exp(-diff**2 / (2*v_br)) / np.sqrt(2*np.pi*v_br)
    rho_star[ti] = G @ P_flat
    rho_star[ti] /= rho_star[ti].sum() * dx

print("Sinkhorn reference done.")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  ADMM LOOP
# ─────────────────────────────────────────────────────────────────────────────
# Warm-start: linear density interpolation → project onto constraint
rho_z = np.stack([(1-s)*rho0 + s*rho1 for s in t])
m_z   = np.zeros((Nt, Nx))
rho_z, m_z = proj_cont(rho_z, m_z)

rho_x = rho_z.copy()
m_x   = m_z.copy()
u_rho = np.zeros((Nt, Nx))
u_m   = np.zeros((Nt, Nx))
r     = r0

pres_hist      = []
dres_hist      = []
ke_hist        = []
delta_u_hist   = []   # ‖u_{k+1} − u_k‖  = ‖(Δρ, Δm)‖  consecutive iterate distance
dist_rho_hist  = []   # ‖ρ_k − ρ_*‖       density distance to Sinkhorn reference

rho_x_prev = rho_x.copy()
m_x_prev   = m_x.copy()

print(f"\nADMM  (r₀={r0}, tol={tol_rel}, max_iter={n_iter})")
for it in range(n_iter):

    # Save previous prox iterate for delta_u
    rho_x_prev[:] = rho_x
    m_x_prev[:]   = m_x

    # ── Step 1: prox of kinetic energy  (ρₓ ≥ 0 always) ─────────────────────
    rho_x, m_x = prox_kinetic(rho_z - u_rho, m_z - u_m, r)

    # ── Step 2: project onto continuity constraint ────────────────────────────
    rho_z_old, m_z_old = rho_z.copy(), m_z.copy()
    rho_z, m_z = proj_cont(rho_x + u_rho, m_x + u_m)

    # ── Step 3: dual update ───────────────────────────────────────────────────
    u_rho += rho_x - rho_z
    u_m   += m_x   - m_z

    # ── Diagnostics ───────────────────────────────────────────────────────────
    pres = float(np.linalg.norm(
               np.r_[(rho_x - rho_z).ravel(), (m_x - m_z).ravel()]))
    dres = r * float(np.linalg.norm(
               np.r_[(rho_z - rho_z_old).ravel(), (m_z - m_z_old).ravel()]))

    safe = np.maximum(rho_x, 1e-12)
    ke   = float(np.sum(m_x**2 / (2 * safe)) * dx * dt)

    # ‖u_{k+1} − u_k‖  (full iterate change on prox variable)
    delta_u = float(np.linalg.norm(
                  np.r_[(rho_x - rho_x_prev).ravel(),
                        (m_x   - m_x_prev  ).ravel()]))

    # ‖ρ_k − ρ_*‖  (density distance to Sinkhorn reference)
    dist_rho = float(np.linalg.norm((rho_x - rho_star).ravel()))

    pres_hist.append(pres)
    dres_hist.append(dres)
    ke_hist.append(ke)
    delta_u_hist.append(delta_u)
    dist_rho_hist.append(dist_rho)

    # ── Adaptive penalty (Boyd et al. 2010, §3.4.1) ───────────────────────────
    if pres > mu_ada * dres:
        u_rho /= tau_ada;  u_m /= tau_ada;  r *= tau_ada
    elif dres > mu_ada * pres:
        u_rho *= tau_ada;  u_m *= tau_ada;  r /= tau_ada

    # ── Relative stopping criterion ───────────────────────────────────────────
    norm_x = np.linalg.norm(np.r_[rho_x.ravel(), m_x.ravel()])
    norm_z = np.linalg.norm(np.r_[rho_z.ravel(), m_z.ravel()])
    norm_u = r * np.linalg.norm(np.r_[u_rho.ravel(), u_m.ravel()])
    scale  = max(norm_x, norm_z)
    tol_p  = tol_rel * scale
    tol_d  = tol_rel * norm_u

    if (it + 1) % 200 == 0:
        print(f"  iter {it+1:4d} | pres {pres:.2e} | dres {dres:.2e} "
              f"| KE {ke:.4f} | r {r:.2f} | ‖ρ−ρ*‖ {dist_rho:.4f}")

    if pres < tol_p and dres < tol_d:
        print(f"  Converged at iter {it+1}  (pres {pres:.1e}, dres {dres:.1e})")
        break

n_iters = len(ke_hist)
iters   = np.arange(1, n_iters + 1)

print(f"\nFinal  KE_ADMM = {ke_hist[-1]:.5f}")
print(f"  OT lower bound (σ→0): (μ₁−μ₀)²/2 = {(mu1-mu0)**2/2:.4f}")
print(f"  Final ‖ρ − ρ*‖ = {dist_rho_hist[-1]:.5f}")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  PLOTS
# ─────────────────────────────────────────────────────────────────────────────
T_grid, X_grid = np.meshgrid(t, x, indexing='ij')
v_field = m_x / np.maximum(rho_x, 1e-6)

n_show = 9
t_idx  = np.round(np.linspace(0, Nt-1, n_show)).astype(int)
colors = plt.cm.plasma(np.linspace(0.05, 0.93, n_show))

# ── Figure 1: ADMM overview ───────────────────────────────────────────────────
fig1, axes1 = plt.subplots(2, 2, figsize=(13, 9))

ax = axes1[0, 0]
im = ax.contourf(X_grid, T_grid, rho_x, levels=40, cmap='viridis')
plt.colorbar(im, ax=ax)
ax.set_title(r'ADMM: Density $\rho(t,x)$', fontsize=12)
ax.set_xlabel('x');  ax.set_ylabel('t')

ax = axes1[0, 1]
vc = np.percentile(np.abs(v_field), 97)
im2 = ax.contourf(X_grid, T_grid, v_field, levels=40, cmap='RdBu_r',
                  vmin=-vc, vmax=vc)
plt.colorbar(im2, ax=ax)
ax.set_title(r'ADMM: Drift field $u(t,x)$', fontsize=12)
ax.set_xlabel('x');  ax.set_ylabel('t')

ax = axes1[1, 0]
ax.plot(x, rho_x[0],   lw=2.5, c='steelblue', label=r'ADMM  $\rho(0,x)$')
ax.plot(x, rho0,  '--', lw=2,  c='steelblue', alpha=.5, label=r'target $\rho_0$')
ax.plot(x, rho_x[-1],  lw=2.5, c='tomato',    label=r'ADMM  $\rho(1,x)$')
ax.plot(x, rho1,  '--', lw=2,  c='tomato',    alpha=.5, label=r'target $\rho_1$')
ax.legend(fontsize=8);  ax.set_xlabel('x')
ax.set_title('Boundary marginals (ADMM vs target)', fontsize=12)

ax = axes1[1, 1]
ax.semilogy(iters, pres_hist, lw=1.5, label='primal residual  ‖ρₓ−ρ_z, mₓ−m_z‖')
ax.semilogy(iters, dres_hist, lw=1.5, label='dual residual  r‖Δρ_z, Δm_z‖')
ax.legend(fontsize=8);  ax.set_xlabel('ADMM iteration')
ax.set_title(f'ADMM residuals  (final KE={ke_hist[-1]:.3f})', fontsize=12)

plt.suptitle(
    rf'1D Schrödinger Bridge — ADMM  '
    rf'($\sigma={sigma}$, $N_t={Nt}$, $N_x={Nx}$)',
    fontsize=12)
plt.tight_layout()
plt.savefig('sbp_admm_1d.png', dpi=150, bbox_inches='tight')
print("Saved: sbp_admm_1d.png")

# ── Figure 2: convergence + density evolution (4 panels) ─────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(13, 9))

# Panel (0,0): ‖u_{k+1} − u_k‖ vs iterations
ax = axes2[0, 0]
ax.semilogy(iters, delta_u_hist, lw=1.5, c='darkorange')
ax.set_xlabel('ADMM iteration', fontsize=11)
ax.set_ylabel(r'$\|u_{k+1} - u_k\|$', fontsize=11)
ax.set_title(r'Consecutive iterate distance  $\|u_{k+1} - u_k\|$', fontsize=12)
ax.grid(True, which='both', alpha=0.35)

# Panel (0,1): ‖ρ_k − ρ_*‖ vs iterations
ax = axes2[0, 1]
ax.semilogy(iters, dist_rho_hist, lw=1.5, c='purple')
ax.set_xlabel('ADMM iteration', fontsize=11)
ax.set_ylabel(r'$\|\rho_k - \rho_*\|$', fontsize=11)
ax.set_title(r'Density distance to Sinkhorn  $\|\rho_k - \rho_*\|$',
             fontsize=12)
ax.grid(True, which='both', alpha=0.35)

# Panel (1,0): ADMM density evolution
ax = axes2[1, 0]
for k, i in enumerate(t_idx):
    ax.plot(x, rho_x[i], color=colors[k], lw=2.0,
            label=rf'$t={t[i]:.2f}$')
ax.set_xlabel('x', fontsize=11);  ax.set_ylabel(r'$\rho(t,x)$', fontsize=11)
ax.set_title('ADMM density evolution', fontsize=12)
ax.legend(fontsize=7, ncol=2, loc='upper center')

# Panel (1,1): Sinkhorn (exact) density evolution
ax = axes2[1, 1]
for k, i in enumerate(t_idx):
    ax.plot(x, rho_star[i], color=colors[k], lw=2.0,
            label=rf'$t={t[i]:.2f}$')
ax.set_xlabel('x', fontsize=11);  ax.set_ylabel(r'$\rho(t,x)$', fontsize=11)
ax.set_title('Sinkhorn (exact) density evolution', fontsize=12)
ax.legend(fontsize=7, ncol=2, loc='upper center')

plt.suptitle(
    rf'SBP convergence analysis — ADMM vs Sinkhorn reference  ($\sigma={sigma}$)',
    fontsize=12)
plt.tight_layout()
plt.savefig('sbp_admm_convergence.png', dpi=150, bbox_inches='tight')
print("Saved: sbp_admm_convergence.png")

plt.close('all')
