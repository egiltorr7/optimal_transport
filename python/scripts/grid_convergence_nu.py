"""Grid convergence study for the non-uniform-time-grid pipeline
(pipeline_nu.py) -- sibling of grid_convergence.py / multi_eps_study.py,
generalized to a selectable time grid (uniform / graded_ends / clustered)
and FP-projection scheme (CN / ETD), swept across BOTH resolution (nt=nx
doubling) and vareps in one run.

Error metric: reuses sb_gaussian_refine_ends_vs_uniform.py's own err_cc/
err_stag (rho-only, per-time-row L2-in-space, then reduced by mean/max over
rows) -- NOT study_utils.l2_error, which is the uniform-grid scripts' single
dt*dx-weighted scalar combining both rho AND mx. That distinction matters:
the two are not directly comparable numbers, and this script does not
attempt to bridge that gap -- it uses whatever sb_gaussian_refine_ends_vs_
uniform.py (and, by extension, everything validated against it earlier this
session) already treats as ground truth, rather than inventing a third,
unvalidated metric.

Produces one overlay log-log plot (mean error left, max error right; one
curve per vareps) styled like study_utils.plot_grid_convergence (O(n^-1)/
O(n^-2) reference lines, same marker/legend/grid conventions), plus
print_orders-style empirical-order console output per vareps.

Usage:
    python3 scripts/grid_convergence_nu.py
    python3 scripts/grid_convergence_nu.py --grid graded_ends --scheme CN \
        --resolutions 32 64 128 --eps-list 1e-4 1e-2 1
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from optimal_transport.grid_nu import setup_problem_nu
from optimal_transport.ladmm_linesearch import MalitskyLinesearch
from optimal_transport.ladmm_reordered import GlyAdaptiveStepSize
from optimal_transport.pipeline_nu import LadmmConfig
from optimal_transport.problems import prob_gaussian
from optimal_transport.sinkhorn import sinkhorn_fit
from optimal_transport.time_grid import uniform_time_grid  # Sinkhorn fit grid only
from sb_gaussian_refine_ends_vs_uniform import run_one
from study_utils import (SINKHORN_MAX_ITER, SINKHORN_NX_FINE, SINKHORN_TOL, SINKHORN_VAREPS_THRESHOLD,
                         RESULTS_DIR, TIME_GRIDS, BooleanOptionalAction, build_time_grid,
                         print_orders)

# ============================================================================
# Edit these directly for repeated local tinkering -- every one is still
# overridable from the command line.
# ============================================================================
GRID = "clustered"            # "uniform" | "graded_ends" | "clustered"
SCHEME = "ETD"                 # "CN" | "ETD"
# SCHEME = "CN"                 
RESOLUTIONS = [32, 64, 128, 256, 512]   # nt = nx, doubling
EPS_LIST = [1e-4, 1e-2, 1e-1, 1.0, 10.0]
N_BOUNDARY = 5                  # graded_ends only
CHEB_STRENGTH = 1.0             # clustered only
GAMMA = 1
TAU = None                      # None -> gamma * 101/100
MAX_ITER = 40_000
EPS_ABS = 1e-10
EPS_REL = 1e-15
REORDERED = False             # False = ladmm.py's (r,x,y,delta) cycle (default).
                              # True  = ladmm_reordered.py's (y,r,delta,x) cycle
                              # -- see docs/adaptive_pdhg_goldstein.tex.
ADAPTIVE = False              # True (requires REORDERED) = Goldstein-Li-Yuan
                              # residual balancing. False = constant gamma/tau.
OMEGA0 = 0.95                 # adaptive only: their alpha_0
ETA = 0.95                    # adaptive only: adaptivity decay
THRESHOLD = 2.0               # adaptive only: their imbalance factor
LINESEARCH = False            # True = Malitsky-Pock linesearch on gamma
                              # (ladmm_linesearch.py). Runs ladmm.py's OWN
                              # unrotated cycle, so it is incompatible with
                              # REORDERED/ADAPTIVE; tau is then ignored, since
                              # the linesearch ties tau = 1/(BETA*gamma).
                              # See docs/malitsky_linesearch_ladmm.tex.
LS_BETA = 0.5                 # linesearch only: ties tau = 1/(beta*gamma), so
                              # gamma/tau = beta*gamma^2 -- beta sets the ratio
LS_MU = 0.7                   # linesearch only: backtracking factor in (0,1)
LS_KAPPA = 0.99               # linesearch only: tolerance in (0,1), near 1
LS_GROWTH = 0.2               # linesearch only: 0 = never grow gamma, 1 = always
                              # try the largest step (costs ~2 projections/iter)
# ============================================================================


def build_grid(name: str, n: int, n_boundary: int, cheb_strength: float):
    """Thin wrapper over study_utils.build_time_grid, dropping its label/tag
    (this script names its outputs by args.grid alone)."""
    return build_time_grid(name, n, strength=cheb_strength, n_boundary=n_boundary)[0]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid", choices=list(TIME_GRIDS), default=GRID)
    p.add_argument("--scheme", choices=["CN", "ETD"], default=SCHEME)
    p.add_argument("--resolutions", type=int, nargs="+", default=RESOLUTIONS)
    p.add_argument("--eps-list", type=float, nargs="+", default=EPS_LIST)
    p.add_argument("--n-boundary", type=int, default=N_BOUNDARY)
    p.add_argument("--cheb-strength", type=float, default=CHEB_STRENGTH)
    p.add_argument("--gamma", type=float, default=GAMMA)
    p.add_argument("--tau", type=float, default=TAU, help="default: gamma * 101/100")
    p.add_argument("--max-iter", type=int, default=MAX_ITER)
    p.add_argument("--eps-abs", type=float, default=EPS_ABS)
    p.add_argument("--eps-rel", type=float, default=EPS_REL)
    # BooleanOptionalAction so the module constants above can be overridden in
    # *either* direction from the command line (--reordered / --no-reordered).
    p.add_argument("--reordered", action=BooleanOptionalAction, default=REORDERED,
                   help="use ladmm_reordered.py's (y,r,delta,x) cycle instead of ladmm.py's")
    p.add_argument("--adaptive", action=BooleanOptionalAction, default=ADAPTIVE,
                   help="with --reordered: Goldstein-Li-Yuan Algorithm 1 residual balancing")
    p.add_argument("--omega0", type=float, default=OMEGA0)
    p.add_argument("--eta", type=float, default=ETA)
    p.add_argument("--threshold", type=float, default=THRESHOLD)
    p.add_argument("--linesearch", action=argparse.BooleanOptionalAction, default=LINESEARCH,
                   help="Malitsky-Pock linesearch on gamma (excludes --reordered/--adaptive)")
    p.add_argument("--ls-beta", type=float, default=LS_BETA)
    p.add_argument("--ls-mu", type=float, default=LS_MU)
    p.add_argument("--ls-kappa", type=float, default=LS_KAPPA)
    p.add_argument("--ls-growth", type=float, default=LS_GROWTH)
    return p.parse_args()


def main():
    args = parse_args()
    gamma = args.gamma
    tau = args.tau if args.tau is not None else gamma * (101.0 / 100.0)
    if args.adaptive and not args.reordered:
        raise SystemExit("--adaptive requires --reordered (the GLY rule is derived for that cycle)")
    ls_cfg = (MalitskyLinesearch(beta=args.ls_beta, mu=args.ls_mu,
                                 kappa=args.ls_kappa, growth=args.ls_growth)
              if args.linesearch else None)
    ladmm_cfg = LadmmConfig(gamma=gamma, tau=tau, max_iter=args.max_iter,
                             eps_abs=args.eps_abs, eps_rel=args.eps_rel,
                             linesearch=ls_cfg,
                             reordered=args.reordered,
                             step_size=GlyAdaptiveStepSize(
                                 omega0=args.omega0, eta=args.eta,
                                 threshold=args.threshold) if args.adaptive else None)
    ns = np.array(sorted(args.resolutions))

    print(f"grid={args.grid}  scheme={args.scheme}  resolutions={list(ns)}  eps_list={args.eps_list}\n"
          f"gamma={gamma:g}  tau={tau:.6f}  max_iter={args.max_iter}  "
          f"eps_rel={args.eps_rel:g}  eps_abs={args.eps_abs:g}\n"
          f"solver={'reordered' if args.reordered else 'ladmm'}"
          f"{f'  adaptive(omega0={args.omega0:g},eta={args.eta:g},thr={args.threshold:g})' if args.adaptive else '  fixed-step'}\n")

    results = []  # list of dicts: eps, n, mean_err, max_err, iters, converged, D, P, wall
    t_start = time.perf_counter()

    for vareps in args.eps_list:
        sink_fit = None
        if vareps > SINKHORN_VAREPS_THRESHOLD:
            fit_problem = setup_problem_nu(prob_gaussian(), uniform_time_grid(2), nx=SINKHORN_NX_FINE)
            t0 = time.perf_counter()
            sink_fit = sinkhorn_fit(fit_problem, vareps, SINKHORN_MAX_ITER, SINKHORN_TOL)
            print(f"[eps={vareps:g}] Sinkhorn fit: iters={sink_fit.iters} converged={sink_fit.converged} "
                  f"error={sink_fit.error:.2e} wall={time.perf_counter()-t0:.2f}s")

        for n in ns:
            tg = build_grid(args.grid, int(n), args.n_boundary, args.cheb_strength)
            t0 = time.perf_counter()
            r = run_one(tg, int(n), vareps, ladmm_cfg, args.scheme, sink_fit=sink_fit)
            wall = time.perf_counter() - t0
            info = r["result"].info
            row = dict(eps=vareps, n=int(n), mean_err=r["err_cc"].mean(), max_err=r["err_cc"].max(),
                       iters=info.iters, converged=info.converged, D=info.D[-1], P=info.P[-1], wall=wall)
            results.append(row)
            print(f"  n={n:4d}  mean_err={row['mean_err']:.4e}  max_err={row['max_err']:.4e}  "
                  f"iters={row['iters']}  converged={row['converged']}  "
                  f"D={row['D']:.2e}  P={row['P']:.2e}  wall={wall:.1f}s")

        errs = np.array([r["mean_err"] for r in results if r["eps"] == vareps])
        print_orders(f"eps={vareps:g}  cell-centre L2 (mean)", list(ns), errs)

    print(f"\ntotal wall time: {time.perf_counter()-t_start:.1f}s")

    # --- overlay log-log plot: mean/max error vs n, one curve per eps ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(args.eps_list)))

    for ax, key, title in zip(axes, ["mean_err", "max_err"], ["L2 error (mean over t)", "L2 error (max over t)"]):
        for vareps, color in zip(args.eps_list, colors):
            rows = sorted([r for r in results if r["eps"] == vareps], key=lambda r: r["n"])
            errs = np.array([r[key] for r in rows])
            ax.loglog(ns, errs, "o-", color=color, label=f"$\\varepsilon$={vareps:g}")

        anchor = min(r[key] for r in results if r["n"] == ns[-1])
        ax.loglog(ns, anchor * (ns / ns[-1]) ** -1.0, "k--", lw=1, label="O(n^-1) = O(h)")
        ax.loglog(ns, anchor * (ns / ns[-1]) ** -2.0, "k:", lw=1, label="O(n^-2) = O(h^2)")

        ax.set_xlabel("n = nx = nt")
        ax.set_ylabel("error")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, which="both")

    fig.suptitle(f"Grid refinement study (non-uniform pipeline): grid={args.grid}  scheme={args.scheme}")
    fig.tight_layout()

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"grid_convergence_nu_{args.grid}_{args.scheme}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"\nfigure saved to: {out_path}")

    np.save(RESULTS_DIR / f"grid_convergence_nu_{args.grid}_{args.scheme}_results.npy", results, allow_pickle=True)
    print(f"results saved to: {RESULTS_DIR / f'grid_convergence_nu_{args.grid}_{args.scheme}_results.npy'}")


if __name__ == "__main__":
    main()
