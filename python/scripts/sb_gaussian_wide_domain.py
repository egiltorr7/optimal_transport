"""1D SB at large vareps on a WIDE spatial domain, vs the analytical solution.

The closed-form SB (analytical_sb_gaussian*) is the FREE-SPACE solution on all
of R, while the solver has no-flux walls. On [0,1] at vareps=1 that is hopeless:
the bridge's std at mid-time is ~0.71, so the domain is only +-0.71 sigma wide
and roughly half the mass would have to live outside it -- the reflection makes
the closed form simply the wrong target, which is why study_utils switches to a
Sinkhorn reference above 5e-3.

Widening the domain is the other way out. On [-2,3] the same bridge sits inside
+-3.5 sigma, so the mass the box truncates drops to ~4e-4 -- small enough that
the closed form becomes a usable target again. That truncated mass is the FLOOR
on agreement here, and it is printed below so the error can be read against it
rather than mistaken for discretization error.

Only the domain OFFSET is new in the solver (grid_nu.setup_problem_nu's x_min).
The discrete Neumann eigenvalues are translation-invariant -- they depend on nx
and dx alone -- so widening the box rescales them through dx (here dx = 5/256,
so lambda_max falls 25x versus L=1) and nothing else needed rederiving.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc

from optimal_transport.grid_nu import setup_problem_nu
from optimal_transport.operators_nu import build_operators_nu
from optimal_transport.pipeline_nu import LadmmConfig, discretize_then_optimize_nu
from optimal_transport.problems import prob_gaussian
from optimal_transport.problems_nu import analytical_sb_gaussian_nu
from optimal_transport.projection_expsemi_nu import (
    precomp_expsemi_proj_nu,
    proj_fokker_planck_expsemi_nu,
)
from optimal_transport.time_grid import clustered_time_grid

# ===========================================================================
# CONFIG
# ===========================================================================
NT = 64            # time cells (non-uniform: cosine-clustered at both ends)
NX = 256           # space cells
VAREPS = 1.0
X_MIN = -2.0       # domain [X_MIN, X_MIN + L]
L = 5.0
STRENGTH = 1.0     # clustered grid: 0 = uniform, 1 = full cosine clustering
GAMMA = 1.0
TAU = GAMMA * 101.0 / 100.0
MAX_ITER = 40_000
N_PROFILES = 6     # density profiles drawn across time
# ===========================================================================


def main():
    prob_def = prob_gaussian()
    tg = clustered_time_grid(NT, strength=STRENGTH)
    problem = setup_problem_nu(prob_def, tg, nx=NX, L=L, x_min=X_MIN)
    problem.ops = build_operators_nu(problem)

    # How much mass does the box truncate at the widest time? That is the floor
    # on any comparison against the free-space closed form.
    sigma = prob_def.sigma
    alpha = np.sqrt(sigma**4 + VAREPS**2) - sigma**2
    sig_mid = np.sqrt(sigma**2 + 2 * alpha * 0.25)          # widest, at t=1/2
    mu_mid = 0.5 * (prob_def.mu0 + prob_def.mu1)
    z_lo, z_hi = (mu_mid - X_MIN) / sig_mid, (X_MIN + L - mu_mid) / sig_mid
    truncated = 0.5 * (erfc(z_lo / np.sqrt(2)) + erfc(z_hi / np.sqrt(2)))

    print(f"domain [{X_MIN:g}, {X_MIN + L:g}]  nx={NX}  dx={problem.dx:.6f}  "
          f"lambda_max={problem.lambda_x.max():.4g}")
    print(f"time: clustered nt={problem.nt}  dt in "
          f"[{problem.dt_vec.min():.3e}, {problem.dt_vec.max():.3e}]  "
          f"ratio={problem.dt_vec.max()/problem.dt_vec.min():.1f}")
    print(f"bridge std at t=1/2: {sig_mid:.4f}  -> box is +-{z_lo:.2f} sigma")
    print(f"mass truncated by the box: {truncated:.3e}   <- FLOOR on the error below")

    bp = precomp_expsemi_proj_nu(problem, VAREPS)
    cfg = LadmmConfig(gamma=GAMMA, tau=TAU, max_iter=MAX_ITER,
                      eps_abs=1e-10, eps_rel=1e-15)
    res = discretize_then_optimize_nu(problem, proj_fokker_planck_expsemi_nu, bp,
                                      VAREPS, cfg)
    info = res.info
    print(f"solve: iters={info.iters} converged={info.converged} "
          f"wall={info.walltime:.1f}s  D={info.D[-1]:.2e} P={info.P[-1]:.2e}")

    rho_ana, mx_ana = analytical_sb_gaussian_nu(problem, VAREPS)
    t_stag = problem.t_edges[1:-1]
    err = np.sqrt(problem.dx * np.sum((res.rho_stag - rho_ana) ** 2, axis=1))
    rel = np.linalg.norm(res.rho_stag - rho_ana) / np.linalg.norm(rho_ana)
    print(f"L2(x) error vs analytical: max={err.max():.3e} at t={t_stag[np.argmax(err)]:.4f}"
          f"   mean={err.mean():.3e}   relative(whole slab)={rel:.3e}")

    # --- profiles ---
    idx = np.unique(np.linspace(0, len(t_stag) - 1, N_PROFILES).astype(int))
    colors = plt.cm.viridis(np.linspace(0, 0.92, len(idx)))
    stride = max(1, NX // 80)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for k, c in zip(idx, colors):
        axes[0].plot(problem.xx, rho_ana[k], "-", color=c, lw=1.6)
        axes[0].plot(problem.xx[::stride], res.rho_stag[k, ::stride], "o", color=c,
                     ms=4, mfc="none", label=f"t={t_stag[k]:.3f}")
    axes[0].set_xlabel("x"); axes[0].set_ylabel(r"$\rho$")
    axes[0].set_title(f"Density: analytical (line) vs LADMM (dots)\n"
                      f"domain [{X_MIN:g}, {X_MIN+L:g}], vareps={VAREPS:g}")
    axes[0].legend(fontsize=7.5); axes[0].grid(True, alpha=0.3)

    for k, c in zip(idx, colors):
        axes[1].semilogy(problem.xx, np.maximum(rho_ana[k], 1e-16), "-", color=c, lw=1.4)
        axes[1].semilogy(problem.xx[::stride], np.maximum(res.rho_stag[k, ::stride], 1e-16),
                         "o", color=c, ms=3.5, mfc="none")
    axes[1].set_ylim(1e-8, 5)
    axes[1].set_xlabel("x"); axes[1].set_ylabel(r"$\rho$ (log)")
    axes[1].set_title("same, log scale -- shows the tails the box truncates")
    axes[1].grid(True, alpha=0.3)

    axes[2].semilogy(t_stag, err, "-", color="tab:orange", lw=1.6, label="L2(x) error")
    axes[2].axhline(truncated, ls="--", c="k", lw=1.1,
                    label=f"truncated mass {truncated:.1e}")
    axes[2].set_xlabel("t"); axes[2].set_ylabel(r"$\|\rho-\rho_{ana}\|_{L^2(x)}$")
    axes[2].set_title("error vs the free-space closed form")
    axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "results"
    out.mkdir(exist_ok=True)
    stem = f"sb_wide_x{X_MIN:g}_{X_MIN+L:g}_nt{NT}_nx{NX}_eps{VAREPS:g}"
    fig.savefig(out / f"{stem}.png", dpi=140)
    print(f"figure: {out / f'{stem}.png'}")

    np.savez_compressed(
        out / f"{stem}.npz",
        rho_stag=res.rho_stag, mx_stag=res.mx_stag, rho_cc=res.rho_cc,
        rho_ana=rho_ana, mx_ana=mx_ana, err_t=err,
        t_edges=problem.t_edges, t_centers=problem.t_centers, dt_vec=problem.dt_vec,
        xx=problem.xx, dx=problem.dx, x_min=X_MIN, L=L, nt=problem.nt, nx=NX,
        vareps=VAREPS, truncated_mass=truncated,
        D=info.D, P=info.P, iters=info.iters, converged=info.converged,
    )
    print(f"data:   {out / f'{stem}.npz'}")


if __name__ == "__main__":
    main()
