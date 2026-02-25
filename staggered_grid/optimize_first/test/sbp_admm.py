#!/usr/bin/env python3
"""
ADMM solver for the 1D Schrödinger Bridge Problem — Benamou-Brenier formulation.

Problem:
    min_{ρ≥0, m}   ∫₀¹ ∫ m²/(2ρ) dx dt
    s.t.  ∂ₜρ + ∂ₓm = (σ²/2) ∂ₓₓρ   (Fokker-Planck)
          ρ(0,·) = ρ₀  (Gaussian, μ=-1.5, σ=0.5)
          ρ(1,·) = ρ₁  (Gaussian, μ=+1.5, σ=0.5)

Benamou-Brenier ADMM splitting:
    Lift to epigraph: introduce E ≥ m²/(2ρ).
    min  Σ E
    s.t. (ρ,m,E) ∈ K   (rotated second-order cone: 2ρE ≥ m², ρ≥0, E≥0)
         C(ρ,m) = b     (continuity equation + BCs, linear)

    ADMM (scaled dual u = (u_ρ, u_m, u_E)):
      1. cone-step  :  (ρₓ,mₓ,Eₓ) = proj_K(ρ_z − u_ρ, m_z − u_m, E_z − u_E)
      2a. cont-step :  (ρ_z,m_z)   = proj_continuity(ρₓ + u_ρ, mₓ + u_m)
      2b. E-step    :  E_z          = Eₓ + u_E − 1/r   [argmin (1/r)E + ½(E−a)²]
      3. dual update:  u ← u + (ρₓ,mₓ,Eₓ) − (ρ_z,m_z,E_z)

The cone-step always produces ρₓ ≥ 0 and Eₓ ≥ 0, so the solution is physical.
At convergence  Eₓ = mₓ²/(2ρₓ).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.sparse import diags, eye as speye, lil_matrix
from scipy.sparse.linalg import factorized as sparse_factorized

# ─────────────────────────────────────────────────────────────────
# 1.  PARAMETERS
# ─────────────────────────────────────────────────────────────────
Nt    = 30       # time slices
Nx    = 60       # spatial grid points
sigma = 0.3      # diffusion of reference SDE  dX = u dt + σ dW
r     = 1.0      # ADMM penalty
n_iter = 2000    # max iterations
tol   = 1e-4     # stopping tolerance (absolute primal+dual residual)

x_min, x_max = -4.0, 4.0
T = 1.0

x  = np.linspace(x_min, x_max, Nx)
t  = np.linspace(0.0, T, Nt)
dx = x[1] - x[0]
dt = t[1] - t[0]
print(f"Grid: Nt={Nt}, Nx={Nx},  dx={dx:.4f}, dt={dt:.4f}")
print(f"CFL (v=3): {3*dt/dx:.3f}  (< 1 for stability)")

# Gaussian boundary conditions (∫ρ dx = 1 on the grid)
rho0 = norm.pdf(x, -1.5, 0.5);  rho0 /= rho0.sum() * dx
rho1 = norm.pdf(x,  1.5, 0.5);  rho1 /= rho1.sum() * dx

# ─────────────────────────────────────────────────────────────────
# 2.  DISCRETE SPATIAL OPERATORS
# ─────────────────────────────────────────────────────────────────
def make_Dx(N, h):
    D = diags([-np.ones(N-1), np.ones(N-1)], [-1, 1],
              shape=(N, N), format='lil') / (2*h)
    D[0, 0] = -1/h;  D[0, 1] =  1/h
    D[-1,-2] = -1/h;  D[-1,-1] =  1/h
    return D.tocsr()

def make_Dxx(N, h):
    D = diags([np.ones(N-1), -2*np.ones(N), np.ones(N-1)],
              [-1, 0, 1], shape=(N, N), format='lil') / h**2
    D[0,  0] = -1/h**2;  D[0,  1] =  1/h**2
    D[-1,-2] =  1/h**2;  D[-1,-1] = -1/h**2
    return D.tocsr()

Dx  = make_Dx(Nx, dx)
Dxx = make_Dxx(Nx, dx)

# Explicit-Euler step:  ρ_{i+1} = A_fwd ρ_i  −  B_flux m_i
A_fwd  = (speye(Nx) + dt * (sigma**2 / 2) * Dxx).tocsr()
B_flux = (dt * Dx).tocsr()

# ─────────────────────────────────────────────────────────────────
# 3.  CONTINUITY CONSTRAINT MATRIX  C [ρ; m] = b_c
#
#  Constraints  ((Nt+1)·Nx rows):
#    ρ₀        = rho0               [rows 0       : Nx    ]
#    ρ_{Nt-1}  = rho1               [rows Nx      : 2·Nx  ]
#    ρ_{i+1} − A_fwd ρ_i + B_flux m_i = 0   [rows 2Nx + i·Nx]
# ─────────────────────────────────────────────────────────────────
n_con = (Nt + 1) * Nx
n_rho = Nt * Nx           # size of the ρ block
n_m   = Nt * Nx           # size of the m block
n_var = n_rho + n_m

A_arr  = A_fwd.toarray()
B_arr  = B_flux.toarray()
Ix_arr = speye(Nx).toarray()

C   = lil_matrix((n_con, n_var))
b_c = np.zeros(n_con)

C[0:Nx,   0:Nx]                              = Ix_arr;   b_c[0:Nx]   = rho0
C[Nx:2*Nx, (Nt-1)*Nx:Nt*Nx]                 = Ix_arr;   b_c[Nx:2*Nx] = rho1
for i in range(Nt - 1):
    row = (2 + i) * Nx
    C[row:row+Nx, (i+1)*Nx:(i+2)*Nx] =  Ix_arr    # +ρ_{i+1}
    C[row:row+Nx,     i*Nx:(i+1)*Nx] = -A_arr     # −A_fwd ρ_i
    cm = n_rho + i*Nx
    C[row:row+Nx, cm:cm+Nx]          =  B_arr     # +B_flux m_i
C = C.tocsr()

print("Factorising CCᵀ (sparse) …", end=" ", flush=True)
CCT        = (C @ C.T).tocsc()
CCT_reg    = CCT + 1e-10 * speye(n_con, format='csc')
_solve_CCT = sparse_factorized(CCT_reg)
print(f"done  ({CCT.shape}, nnz={CCT.nnz})")


def project_continuity(a_rho, a_m):
    """Project (a_rho, a_m) onto {C[ρ;m] = b_c}  via KKT (sparse LU)."""
    a   = np.concatenate([a_rho.ravel(), a_m.ravel()])
    lam = _solve_CCT(C @ a - b_c)
    z   = a - C.T @ lam
    return z[:n_rho].reshape(Nt, Nx), z[n_rho:].reshape(Nt, Nx)


# ─────────────────────────────────────────────────────────────────
# 4.  ROTATED SECOND-ORDER CONE PROJECTION
#
#  K = {(ρ, m, E) : 2ρE ≥ m², ρ≥0, E≥0}
#
#  Mapped to Lorentz cone  L = {(τ, u₁, u₂) : τ ≥ ||(u₁,u₂)||}  via:
#      τ  = (ρ + E) / √2
#      u₁ = (ρ − E) / √2
#      u₂ = m
#
#  proj_L(τ, u):
#    in L          → return (τ, u)
#    in -L         → return 0
#    otherwise     → ((τ+||u||)/2) · (1, u/||u||)
# ─────────────────────────────────────────────────────────────────
_SQRT2 = np.sqrt(2.0)

def proj_cone(a_rho, a_m, a_E):
    """Project (a_rho, a_m, a_E) onto the rotated SOC K (vectorised)."""
    rho = a_rho.ravel()
    m   = a_m.ravel()
    E   = a_E.ravel()

    tau  = (rho + E) / _SQRT2
    u1   = (rho - E) / _SQRT2
    u2   = m
    unorm = np.hypot(u1, u2)   # ||u||

    rho_out = np.zeros_like(rho)
    m_out   = np.zeros_like(m)
    E_out   = np.zeros_like(E)

    in_cone   = tau >= unorm              # already feasible
    anti_cone = tau <= -unorm             # project to origin
    proj_idx  = ~in_cone & ~anti_cone     # project onto cone boundary

    rho_out[in_cone] = rho[in_cone]
    m_out[in_cone]   = m[in_cone]
    E_out[in_cone]   = E[in_cone]
    # anti_cone → zeros (already initialised)

    if proj_idx.any():
        tp = tau[proj_idx];  u1p = u1[proj_idx]
        u2p = u2[proj_idx];  up  = unorm[proj_idx]
        tau_star = (tp + up) / 2.0          # positive (since tp > -up)
        rho_out[proj_idx] = (tau_star + tau_star * u1p / up) / _SQRT2
        E_out[proj_idx]   = (tau_star - tau_star * u1p / up) / _SQRT2
        m_out[proj_idx]   =  tau_star * u2p / up

    return (rho_out.reshape(a_rho.shape),
            m_out.reshape(a_m.shape),
            E_out.reshape(a_E.shape))


# ─────────────────────────────────────────────────────────────────
# 5.  INITIALISATION
# ─────────────────────────────────────────────────────────────────
# Warm-start: linear interpolation of density, then project onto continuity
rho_z = np.stack([(1.0-ti)*rho0 + ti*rho1 for ti in t])
m_z   = np.zeros((Nt, Nx))
rho_z, m_z = project_continuity(rho_z, m_z)

# Initial E: m²/(2*max(ρ,ε))
E_z = m_z**2 / (2 * np.maximum(rho_z, 1e-8))

rho_x = rho_z.copy()
m_x   = m_z.copy()
E_x   = E_z.copy()

u_rho = np.zeros((Nt, Nx))
u_m   = np.zeros((Nt, Nx))
u_E   = np.zeros((Nt, Nx))

primal_hist = []
dual_hist   = []
ke_hist     = []

# ─────────────────────────────────────────────────────────────────
# 6.  ADMM LOOP
# ─────────────────────────────────────────────────────────────────
print(f"\nRunning BB-ADMM (max {n_iter} iters, r={r}) …")

for it in range(n_iter):

    # ── Step 1: project onto rotated SOC (enforces ρ ≥ 0, E ≥ 0) ─────────────
    rho_x, m_x, E_x = proj_cone(
        rho_z - u_rho,
        m_z   - u_m,
        E_z   - u_E)

    # ── Step 2a: project (ρ, m) onto continuity constraint ───────────────────
    rho_z_prev = rho_z.copy()
    m_z_prev   = m_z.copy()
    E_z_prev   = E_z.copy()

    rho_z, m_z = project_continuity(rho_x + u_rho, m_x + u_m)

    # ── Step 2b: unconstrained E-update (minimise Σ E_z/r + ½‖E_z − a‖²) ────
    E_z = E_x + u_E - 1.0 / r

    # ── Step 3: dual update ───────────────────────────────────────────────────
    u_rho += rho_x - rho_z
    u_m   += m_x   - m_z
    u_E   += E_x   - E_z

    # ── Diagnostics ──────────────────────────────────────────────────────────
    pres = float(np.linalg.norm(
                 np.r_[(rho_x - rho_z).ravel(),
                       (m_x   - m_z).ravel(),
                       (E_x   - E_z).ravel()]))
    dres = r * float(np.linalg.norm(
                 np.r_[(rho_z - rho_z_prev).ravel(),
                       (m_z   - m_z_prev).ravel(),
                       (E_z   - E_z_prev).ravel()]))

    ke   = float(E_x.sum()) * dx * dt       # ≈ ∫∫ m²/(2ρ) at cone variable

    primal_hist.append(pres)
    dual_hist.append(dres)
    ke_hist.append(ke)

    if (it + 1) % 100 == 0:
        print(f"  iter {it+1:5d} | primal {pres:.3e} | dual {dres:.3e} | KE {ke:.4f}")

    if pres < tol and dres < tol:
        print(f"  Converged at iteration {it+1}  "
              f"(primal {pres:.2e}, dual {dres:.2e})")
        break

print(f"\nFinal KE = {ke_hist[-1]:.5f}  (OT lower bound ≈ {(1.5-(-1.5))**2/2:.4f})")

# ─────────────────────────────────────────────────────────────────
# 7.  EXTRACT SOLUTION  (use cone variable: ρ ≥ 0 by construction)
# ─────────────────────────────────────────────────────────────────
rho_sol = rho_x                          # always ≥ 0
m_sol   = m_x
v_sol   = m_x / np.maximum(rho_x, 1e-8) # drift velocity u = m/ρ

# ─────────────────────────────────────────────────────────────────
# 8.  VISUALISATION
# ─────────────────────────────────────────────────────────────────
T_grid, X_grid = np.meshgrid(t, x, indexing='ij')

# ── Figure 1: overview ───────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

ax = axes[0, 0]
im = ax.contourf(X_grid, T_grid, rho_sol, levels=40, cmap='viridis')
plt.colorbar(im, ax=ax)
ax.set_title(r'Density $\rho(t,x)$');  ax.set_xlabel('x');  ax.set_ylabel('t')

ax = axes[0, 1]
vmax = np.percentile(np.abs(v_sol), 97)
im2  = ax.contourf(X_grid, T_grid, v_sol, levels=40, cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax)
plt.colorbar(im2, ax=ax)
ax.set_title(r'Drift $v = m/\rho$');  ax.set_xlabel('x');  ax.set_ylabel('t')

ax = axes[1, 0]
ax.plot(x, rho_sol[0],  lw=2, color='steelblue', label=r'$\rho(0,\cdot)$ computed')
ax.plot(x, rho0,    '--',lw=2, color='steelblue', alpha=0.5, label=r'$\rho_0$ target')
ax.plot(x, rho_sol[-1], lw=2, color='tomato',    label=r'$\rho(1,\cdot)$ computed')
ax.plot(x, rho1,    '--',lw=2, color='tomato',    alpha=0.5, label=r'$\rho_1$ target')
ax.legend(fontsize=8);  ax.set_xlabel('x');  ax.set_title('Boundary marginals')

ax = axes[1, 1]
ax.semilogy(primal_hist, lw=1.5, label='primal residual')
ax.semilogy(dual_hist,   lw=1.5, label='dual residual')
ax.legend();  ax.set_xlabel('iteration');  ax.set_title('ADMM convergence')

plt.suptitle(
    rf'Schrödinger Bridge via BB-ADMM  ($\sigma={sigma}$, $N_t={Nt}$, $N_x={Nx}$)',
    fontsize=12)
plt.tight_layout()
plt.savefig('sbp_admm.png', dpi=150, bbox_inches='tight')

# ── Figure 2: density evolution (stacked slices) ─────────────────
n_show = 9
t_idx  = np.round(np.linspace(0, Nt-1, n_show)).astype(int)
colors = plt.cm.plasma(np.linspace(0, 1, n_show))

fig2, ax2 = plt.subplots(figsize=(9, 5))
for k, i in enumerate(t_idx):
    ax2.plot(x, rho_sol[i], color=colors[k], lw=2,
             label=rf'$t={t[i]:.2f}$')
ax2.set_xlabel('x');  ax2.set_ylabel(r'$\rho(t,x)$')
ax2.set_title(rf'Density evolution — Schrödinger Bridge  ($\sigma={sigma}$, KE={ke_hist[-1]:.3f})')
ax2.legend(fontsize=8, ncol=3, loc='upper center')
plt.tight_layout()
plt.savefig('sbp_admm_evolution.png', dpi=150, bbox_inches='tight')

plt.show()
print("Saved: sbp_admm.png  and  sbp_admm_evolution.png")
