"""Tests for the unbalanced (source-term) 1D solver, pipeline_unbalanced.py.

T1  uniform -> c*uniform, eps>0:    exact Hellinger path rho_t = ((1-t)+t sqrt c)^2, m = 0 (any eps).
T2  Gaussian -> c*Gaussian, eps=0:  same closed form (pure reaction is optimal for rho1 = c rho0 at eps=0).
T3  equal-mass Gaussians 1/3 -> 2/3, eps=0, small delta (pi*delta < distance): mass teleports,
    rho_t = (1-t)^2 rho0 + t^2 rho1 (destroy in place, create in place), m ~ 0.
T4  same Gaussians, eps=0.01, delta -> large: converges to the balanced solver's answer, difference ~ 1/delta^2.
T5  KKT relation on a generic unequal-mass problem: m/rho = d_x(delta * s/rho)  (from m = rho phi_x, s = rho phi/delta).
T6  discrete mass balance: d/dt mass = int s/delta exactly.
Also prints kinetic vs reaction cost and saves a figure.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optimal_transport.grid import setup_problem
from optimal_transport.operators import build_operators
from optimal_transport.pipeline import LadmmConfig, discretize_then_optimize
from optimal_transport.problems import ProblemDef, normal_pdf, prob_gaussian, analytical_sb_gaussian
from optimal_transport.projection import precomp_banded_proj, proj_fokker_planck_banded
from optimal_transport.pipeline_unbalanced import discretize_then_optimize_unbalanced

NT = NX = 64
def cfg(gamma=1.0, it=15000):
    return LadmmConfig(gamma=gamma, tau=gamma * 1.01, max_iter=it, eps_abs=1e-11, eps_rel=1e-8)

def make(prob_def, mass1=1.0, nt=NT, nx=NX):
    p = setup_problem(prob_def, nt=nt, nx=nx); p.ops = build_operators(p); p.rho1 = mass1 * p.rho1
    return p

def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))

def run(p, eps, delta, label, gamma=1.0):
    t0 = time.time(); r = discretize_then_optimize_unbalanced(p, eps, delta, cfg(gamma)); 
    print(f"[{label}] eps={eps:g} delta={delta:g}: iters={r.info.iters} conv={r.info.converged} "
          f"{time.time()-t0:.1f}s  KE={r.ke:.5f} reaction={r.reaction:.5f} total={r.ke+r.reaction:.5f}  "
          f"mass {r.mass_t[0]:.3f}->{r.mass_t[-1]:.3f}")
    return r

t_stag = (np.arange(1, NT) / NT)[:, None]
results = {}

# ---------- T1: uniform -> 2 uniform ----------
c = 2.0
p = make(ProblemDef("uniform", lambda x: np.ones_like(x), lambda x: np.ones_like(x), 0.5, 0.5, 1.0), mass1=c)
r = run(p, 0.01, 0.3, "T1 uniform->2*uniform")
lam = (1 - t_stag) + t_stag * np.sqrt(c)
rho_exact = lam**2 * p.rho0[None, :]
print(f"   T1: rel err rho vs Hellinger path = {rel(r.rho_stag, rho_exact):.2e}, max|m| = {np.abs(r.mx_stag).max():.2e}, "
      f"cost exact 2 delta^2 (sqrt c - 1)^2 = {2*0.3**2*(np.sqrt(c)-1)**2:.5f}")
results['T1'] = r

# ---------- T2: Gaussian -> 2 Gaussian at eps = 0 ----------
p = make(prob_gaussian(mu0=0.5, mu1=0.5, sigma=0.08), mass1=c)
r = run(p, 0.0, 0.3, "T2 gauss->2*gauss eps=0")
rho_exact = lam**2 * p.rho0[None, :]
print(f"   T2: rel err rho vs Hellinger path = {rel(r.rho_stag, rho_exact):.2e}, max|m|/max rho = {np.abs(r.mx_stag).max()/r.rho_stag.max():.2e}")
results['T2'] = r

# ---------- T3: teleport (small delta) at eps = 0 ----------
pg = make(prob_gaussian(mu0=1/3, mu1=2/3, sigma=0.05))
delta_small = 0.05
r = run(pg, 0.0, delta_small, "T3 teleport")
rho_tele = (1 - t_stag)**2 * pg.rho0[None, :] + t_stag**2 * pg.rho1[None, :]
print(f"   T3: pi*delta = {np.pi*delta_small:.3f} < distance 0.333. rel err rho vs (1-t)^2 rho0 + t^2 rho1 = {rel(r.rho_stag, rho_tele):.2e}; "
      f"exact cost 4 delta^2 = {4*delta_small**2:.5f}; transport-only cost would be >= {0.5*(1/3)**2:.4f}")
results['T3'] = r

# ---------- T4: large delta -> balanced ----------
eps = 0.01
bp = precomp_banded_proj(pg, eps)
t0 = time.time(); rb = discretize_then_optimize(pg, proj_fokker_planck_banded, bp, eps, cfg()); 
print(f"[T4 balanced] iters={rb.info.iters} conv={rb.info.converged} {time.time()-t0:.1f}s")
rho_ana, mx_ana = analytical_sb_gaussian(pg, eps)
print(f"   balanced solver vs analytic SB: rho {rel(rb.rho_stag, rho_ana):.2e}  m {rel(rb.mx_stag, mx_ana):.2e}")
for d in (0.5, 1.0, 2.0, 4.0):
    r = run(pg, eps, d, f"T4 delta={d}")
    print(f"   T4 delta={d}: rel diff to balanced solver: rho {rel(r.rho_stag, rb.rho_stag):.2e}  m {rel(r.mx_stag, rb.mx_stag):.2e}; "
          f"max mass deviation {np.abs(r.mass_t-1).max():.2e}")
    results[f'T4_{d}'] = r

# ---------- T5: KKT relation on an unequal-mass, shifted problem ----------
p = make(prob_gaussian(mu0=0.35, mu1=0.6, sigma=0.06), mass1=1.5)
r = run(p, 0.01, 0.4, "T5 generic")
rho, m, s = r.rho_cc, r.mx_cc, r.s_cc
mask = rho > 1e-2 * rho.max()
phi = np.where(mask, r.delta * s / np.where(mask, rho, 1.0), np.nan)      # phi = delta s / rho
dphi = np.gradient(phi, p.dx, axis=1)                                  # d_x phi on cell centres
v = np.where(mask, m / np.where(mask, rho, 1.0), np.nan)
inner = mask[:, 2:-2] & mask[:, 1:-3] & mask[:, 3:-1]
err = np.abs(v[:, 2:-2] - dphi[:, 2:-2])[inner]
print(f"   T5 KKT  m/rho vs d_x(delta s/rho): median rel err {np.median(err)/np.nanmax(np.abs(v)):.2e}, 90th pct {np.percentile(err,90)/np.nanmax(np.abs(v)):.2e}")
# T6 discrete mass balance using the FP-feasible x copy
dm = np.diff(r.mass_t) / p.dt
src = (r.s_stag / r.delta).sum(axis=1) * p.dx
print(f"   T6 mass balance: max |d mass/dt - int s/delta| = {np.abs(dm - src).max():.2e}  (mass {r.mass_t[0]:.3f} -> {r.mass_t[-1]:.3f})")
results['T5'] = r

# ---------- figure ----------
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), constrained_layout=True)
x = pg.xx; ks = [0, NT//4 - 1, NT//2 - 1, 3*NT//4 - 1, NT - 2]
cols = plt.cm.Blues(np.linspace(0.35, 1.0, len(ks)))
for ax, key, title in ((axes[0], 'T3', rf'$\delta={delta_small}$, $\varepsilon=0$: teleport ($\pi\delta<$ distance)'),
                       (axes[1], 'T4_4.0', r'$\delta=4$, $\varepsilon=0.01$: transport (balanced limit)')):
    r = results[key]
    for k, col in zip(ks, cols):
        ax.plot(x, r.rho_stag[k], color=col, lw=1.4, label=f't={(k+1)/NT:.2f}')
    ax.plot(x, pg.rho0, 'k:', lw=0.8); ax.plot(x, pg.rho1, 'k:', lw=0.8)
    ax.set_title(title); ax.set_xlabel('x'); ax.set_ylabel(r'$\rho_t$'); ax.legend(fontsize=7, frameon=False)
ax = axes[2]
tt = np.arange(NT + 1) / NT
for key, lab in (('T3', r'$\delta=0.05$'), ('T4_0.5', r'$\delta=0.5$'), ('T4_1.0', r'$\delta=1$'), ('T4_4.0', r'$\delta=4$')):
    ax.plot(tt, results[key].mass_t, lw=1.4, label=lab)
ax.plot(tt, (1-tt)**2 + tt**2, 'k--', lw=0.8, label=r'$(1-t)^2+t^2$ (pure teleport)')
ax.set_title('total mass along the bridge, equal-mass Gaussians'); ax.set_xlabel('t'); ax.set_ylabel('mass'); ax.legend(fontsize=7, frameon=False)
fig.suptitle(r'Unbalanced 1D solver: N(1/3, 0.05$^2$) $\to$ N(2/3, 0.05$^2$), $64\times64$, cost $\int|m|^2/2\rho+\delta^2\zeta^2/2\rho$', fontsize=10)
out = Path(__file__).resolve().parent.parent / 'docs' / 'figures' / 'unbalanced_1d.png'
fig.savefig(out, dpi=140); print('saved', out)
