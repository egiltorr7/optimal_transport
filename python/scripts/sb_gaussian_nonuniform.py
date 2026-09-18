"""Schrodinger bridge on a non-uniform time grid.

Sibling of sb_gaussian.py, swapped over to the *_nu modules (time_grid,
grid_nu, operators_nu, projection_nu, pipeline_nu) end to end. Nothing this
script imports touches the uniform-grid path -- see those modules'
docstrings for how each generalizes (and where it was validated to reduce
exactly to the uniform-grid result: scripts/validate_nu.py).

Compared against the closed-form SB solution sampled at the *actual*
(non-uniform) grid points -- problems_nu.analytical_sb_gaussian_nu -- rather
than at uniform k*dt fractions.

Grid choices (--grid): all four of study_utils.TIME_GRIDS, so this script is
the general driver rather than the non-uniform-only one its name suggests.
See study_utils.build_time_grid for what each is; briefly:
  uniform      the plain grid -- included so uniform and non-uniform runs go
               through one code path and are directly comparable.
  clustered    cosine clustering at both ends, --strength in [0,1].
               --strength 0 reproduces `uniform` exactly (bit-for-bit).
  graded_ends  geometric grading over --n-boundary cells at each end.
  refine_ends  cell 0 and cell nt-1 of the base nt-cell grid each split in
               half (nt+2 cells total) -- see
               docs/nonuniform_grid_refine_ends.tex.

FP-projection scheme (--scheme): "CN" (banded/Crank-Nicolson, projection_nu.py)
or "ETD" (exponential time differencing, projection_expsemi_nu.py). Reuses the
SCHEMES registry from sb_gaussian_refine_ends_vs_uniform.py rather than a
second copy -- both schemes' precompute/projection functions share the same
(problem, vareps) / (state, problem, vareps, precomp) signatures, so
discretize_then_optimize_nu takes either one interchangeably.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from optimal_transport.grid_nu import setup_problem_nu
from optimal_transport.operators_nu import build_operators_nu
from optimal_transport.ladmm_linesearch import MalitskyLinesearch
from optimal_transport.ladmm_reordered import GlyAdaptiveStepSize
from optimal_transport.pipeline_nu import (
    AliaConfig,
    LadmmConfig,
    discretize_then_optimize_nu,
    discretize_then_optimize_nu_alia,
)
from optimal_transport.problems import prob_gaussian
from optimal_transport.sinkhorn import sinkhorn_fit
from optimal_transport.time_grid import uniform_time_grid
from sb_gaussian_refine_ends_vs_uniform import SCHEMES, reference_rho_at
from study_utils import (
    TIME_GRIDS,
    BooleanOptionalAction,
    build_time_grid,
    SINKHORN_MAX_ITER,
    SINKHORN_NX_FINE,
    SINKHORN_TOL,
    SINKHORN_VAREPS_THRESHOLD,
    SINKHORN_GRIDS,
    print_tail_rates,
    sinkhorn_reference_fit,
)

# ============================================================================
# Edit these directly to change the default run -- every one is still
# overridable from the command line (e.g. `--nt 64`), but for repeated local
# tinkering it's usually less tedious to just edit the numbers here and run
# `python3 sb_gaussian_nonuniform.py` with no flags.
# ============================================================================
GRID = "uniform"          # any of study_utils.TIME_GRIDS:
                            # "uniform" | "clustered" | "graded_ends" | "refine_ends"
SCHEME = "ETD"                 # "CN" | "ETD" -- see SCHEMES in sb_gaussian_refine_ends_vs_uniform.py
NT = 32                    # base time resolution (refine_ends: total cells = nt+2)
NX = 32
VAREPS = 1e-4                # above study_utils.SINKHORN_VAREPS_THRESHOLD (5e-3) -> compared
                              # against a Sinkhorn-Hopf-Cole fit instead of the analytical SB
                              # reference (see reference_rho_at, imported below)
STRENGTH = 1.0                # clustered grid only: 0=uniform, 1=full cosine clustering
N_BOUNDARY = 5                # graded_ends grid only: cells graded at each end
GAMMA = 1.0
ratio = 0.9
TAU = GAMMA* ratio                    # None -> gamma * 101/100
MAX_ITER = 40_000
EPS_ABS = 1e-10
EPS_REL = 1e-15

# Reference solution (only used when VAREPS > SINKHORN_VAREPS_THRESHOLD, below
# which the analytical SB is used and neither setting applies).  See
# docs/sinkhorn_reference.tex for what each choice actually measures.
SINKHORN_GRID = "same"   # "fine": fit at SINKHORN_NX_FINE, reduce to NX -- the
                         #         reference is ~the CONTINUUM solution, so this
                         #         measures TOTAL discretization error.
                         # "same": fit on this run's own NX -- no reduction, the
                         #         marginals are the solver's own exactly, and it
                         #         isolates the TIME error (blind to spatial
                         #         error by construction).  Especially apt here:
                         #         grading shrinks dt near the boundary, so the
                         #         time error is the quantity of interest.
RHO_REDUCE = "spectral"  # SINKHORN_GRID="fine" only: "spectral" | "pointwise" |
                         # "average".  "average" leaves a spurious O(dx^2/dt)
                         # floor at t=0 and t=1; use only for old numbers.
SOLVER = "ladmm"              # which algorithm to run. One of:
#   "ladmm"     ladmm.py's original (r,x,y,delta) cycle, CONSTANT gamma/tau.
#   "reordered" ladmm_reordered.py's (y,r,delta,x) cycle, constant gamma/tau.
#               Same fixed point; this cycle IS Goldstein-Li-Yuan's PDHG.
#   "gly"       "reordered" + Goldstein-Li-Yuan residual balancing. Scales
#               gamma and tau by the SAME factor, so gamma/tau is invariant.
#   "malitsky"  ladmm.py's OWN cycle + Malitsky-Pock linesearch on gamma.
#               Holds gamma*tau = 1/beta fixed, so gamma/tau DOES move --
#               orthogonal to "gly". TAU below is ignored.
#   "alia"      ALiA (Jang, Sun, Yin & Ryu). A different algorithm, not a
#               step-size rule: both blocks get linearized proximal steps, so
#               the y-subproblem is no longer solved exactly. TAU is ignored.
# See docs/adaptive_pdhg_goldstein.tex and docs/malitsky_linesearch_ladmm.tex.
#
# GAMMA/TAU above are the starting steps. For "malitsky" and "alia", GAMMA is
# only the INITIAL value (both adapt it) and TAU is unused.
#
# -- "gly" only -------------------------------------------------------------
OMEGA0 = 0.95                 # their alpha_0 (NIPS'15 Algorithm 1)
ETA = 0.95                    # adaptivity decay
THRESHOLD = 2.0               # their hard-coded imbalance factor
# -- "malitsky" only --------------------------------------------------------
LS_BETA = 0.5                 # ties tau = 1/(beta*gamma), so gamma/tau = beta*gamma^2
LS_MU = 0.7                   # backtracking factor, (0,1)
LS_KAPPA = 0.99               # linesearch tolerance, (0,1), near 1
LS_GROWTH = 0.2               # 0 = never grow gamma, 1 = always try the largest
                              # step (costs ~2 FP projections per iteration)
# -- "alia" only ------------------------------------------------------------
ALIA_SIGMA = 1000            # primal-dual step ratio. The real knob: mean L2
                              # spanned 3 orders of magnitude across sigma.
ALIA_EPS_FRAC = 0.0           # alia_eps = this * min(1/2, 1/(4*sigma)).
                              # ALiA's OWN epsilon -- NOT the SB vareps.
# ============================================================================


N_EDGE_PROFILE = 10   # rho profiles: also draw the worst-error time among the
                      # first and the last this-many staggered rows


def _drop_leading_zero(iters, vals):
    """Drop the leading sample of a semilogy series when it is machine zero.

    The first residual / iterate change is often exactly 0 (or a denormal) from
    the initialization. An exact 0 is silently dropped by a log axis anyway, but
    a denormal is not: it is plotted, and it drags the y-range down by ~300
    decades, flattening everything of interest into the top pixel row. Only the
    FIRST sample is ever dropped, and only when it is zero, negative, or below
    machine epsilon relative to the series maximum.
    """
    v = np.asarray(vals, float)
    if v.size and (v[0] <= 0.0 or v[0] < np.finfo(float).eps * np.max(v)):
        return iters[1:], v[1:]
    return iters, v


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid", choices=list(TIME_GRIDS), default=GRID)
    p.add_argument("--scheme", choices=["CN", "ETD"], default=SCHEME)
    p.add_argument("--nt", type=int, default=NT, help="base time resolution (refine_ends: total cells = nt+2)")
    p.add_argument("--nx", type=int, default=NX)
    p.add_argument("--vareps", type=float, default=VAREPS,
                    help="above study_utils.SINKHORN_VAREPS_THRESHOLD (5e-3), compared against a "
                         "Sinkhorn-Hopf-Cole fit instead of the analytical SB reference")
    p.add_argument("--strength", type=float, default=STRENGTH, help="clustered grid only: 0=uniform, 1=full cosine clustering")
    p.add_argument("--n-boundary", type=int, default=N_BOUNDARY, help="graded_ends grid only: cells graded at each end")
    p.add_argument("--gamma", type=float, default=GAMMA)
    p.add_argument("--tau", type=float, default=TAU, help="default: gamma * 101/100")
    p.add_argument("--max-iter", type=int, default=MAX_ITER)
    p.add_argument("--eps-abs", type=float, default=EPS_ABS)
    p.add_argument("--eps-rel", type=float, default=EPS_REL)
    p.add_argument("--sinkhorn-grid", choices=list(SINKHORN_GRIDS), default=SINKHORN_GRID,
                    help="spatial grid for the Sinkhorn reference fit (see CONFIG)")
    p.add_argument("--rho-reduce", choices=["spectral", "pointwise", "average"],
                    default=RHO_REDUCE,
                    help="--sinkhorn-grid=fine only: how rho reaches the coarse grid")
    p.add_argument("--solver", choices=["ladmm", "reordered", "gly", "malitsky", "alia"],
                   default=SOLVER, help="which algorithm to run; see SOLVER in the source")
    p.add_argument("--omega0", type=float, default=OMEGA0, help="gly only")
    p.add_argument("--eta", type=float, default=ETA, help="gly only")
    p.add_argument("--threshold", type=float, default=THRESHOLD, help="gly only")
    p.add_argument("--ls-beta", type=float, default=LS_BETA, help="malitsky only")
    p.add_argument("--ls-mu", type=float, default=LS_MU, help="malitsky only")
    p.add_argument("--ls-kappa", type=float, default=LS_KAPPA, help="malitsky only")
    p.add_argument("--ls-growth", type=float, default=LS_GROWTH, help="malitsky only")
    p.add_argument("--alia-sigma", type=float, default=ALIA_SIGMA, help="alia only")
    p.add_argument("--alia-eps-frac", type=float, default=ALIA_EPS_FRAC, help="alia only")
    return p.parse_args()


def main():
    args = parse_args()
    nt, nx, vareps = args.nt, args.nx, args.vareps
    gamma = args.gamma
    tau = args.tau if args.tau is not None else gamma * (101.0 / 100.0)
    # gamma/tau/max_iter calibrated against the uniform-grid sb_gaussian.py
    # at comparable vareps: this is a slow-converging regime (small vareps ->
    # near-degenerate KE prox), several e4 iterations needed for the
    # residuals to bottom out. LADMM converges at the same *rate* on both
    # grids (see validate_nu.py's uniform-grid regression), so the same
    # gamma/tau ballpark carries over.

    prob_def = prob_gaussian()
    time_grid, grid_label, grid_tag = build_time_grid(
        args.grid, nt, strength=args.strength, n_boundary=args.n_boundary)
    problem = setup_problem_nu(prob_def, time_grid, nx=nx)
    problem.ops = build_operators_nu(problem)

    sink_fit = None
    ref_name = "analytical"
    if vareps > SINKHORN_VAREPS_THRESHOLD:
        ref_name = f"Sinkhorn/{args.sinkhorn_grid}"
        print(f"vareps={vareps:g} > {SINKHORN_VAREPS_THRESHOLD:g}: using a Sinkhorn-Hopf-Cole "
              f"reference instead of the analytical solution.")
        # sinkhorn_fit reads only the SPATIAL grid, so the fit problem's time grid
        # is a placeholder; uniform_time_grid(2) is the cheapest valid one.
        sink_fit = sinkhorn_reference_fit(
            problem, vareps, grid=args.sinkhorn_grid,
            make_fine_problem=lambda n: setup_problem_nu(
                prob_gaussian(), uniform_time_grid(2), nx=n),
        )

    precomp_fn, proj_fn = SCHEMES[args.scheme]
    bp = precomp_fn(problem, vareps)
    sv = args.solver
    if sv == "alia":
        alia_eps = args.alia_eps_frac * min(0.5, 1.0 / (4.0 * args.alia_sigma))
        solve_cfg = AliaConfig(gamma0=gamma, sigma=args.alia_sigma, alia_eps=alia_eps,
                               max_iter=args.max_iter, eps_abs=args.eps_abs,
                               eps_rel=args.eps_rel)
        detail = (f"gamma0={gamma:g}  sigma={args.alia_sigma:g}  "
                  f"alia_eps={alia_eps:.3g}   (tau unused)")
    else:
        ls_cfg = (MalitskyLinesearch(beta=args.ls_beta, mu=args.ls_mu, kappa=args.ls_kappa,
                                     growth=args.ls_growth) if sv == "malitsky" else None)
        gly_cfg = (GlyAdaptiveStepSize(omega0=args.omega0, eta=args.eta,
                                       threshold=args.threshold) if sv == "gly" else None)
        solve_cfg = LadmmConfig(gamma=gamma, tau=tau, max_iter=args.max_iter,
                                eps_abs=args.eps_abs, eps_rel=args.eps_rel,
                                linesearch=ls_cfg, step_size=gly_cfg,
                                reordered=sv in ("reordered", "gly"))
        if sv == "malitsky":
            detail = (f"gamma0={gamma:g}  beta={args.ls_beta:g}  mu={args.ls_mu:g}  "
                      f"kappa={args.ls_kappa:g}  growth={args.ls_growth:g}   (tau unused)")
        elif sv == "gly":
            detail = (f"gamma={gamma:g}  tau={tau:g}  omega0={args.omega0:g}  "
                      f"eta={args.eta:g}  threshold={args.threshold:g}")
        else:
            detail = f"gamma={gamma:g}  tau={tau:g}   (constant)"

    print(
        f"Running SB Gaussian (grid={grid_label}, scheme={args.scheme}): "
        f"nt={nt} (total cells={problem.nt}) nx={nx} eps={vareps:g}\n"
        f"  solver={sv}   {detail}"
    )
    print(f"  smallest dt={problem.dt_vec.min():.2e}  largest dt={problem.dt_vec.max():.2e}  "
          f"(uniform reference dt={problem.dt_ref:.2e})")

    if sv == "alia":
        result = discretize_then_optimize_nu_alia(problem, proj_fn, bp, vareps, solve_cfg)
    else:
        result = discretize_then_optimize_nu(problem, proj_fn, bp, vareps, solve_cfg)
    info = result.info
    print(
        f"  iters={info.iters}  converged={info.converged}  wall={info.walltime:.2f}s\n"
        f"  final residuals:  dx={info.dx[-1]:.2e}  dy={info.dy[-1]:.2e}  "
        f"dz={info.dz[-1]:.2e}  D={info.D[-1]:.2e}  P={info.P[-1]:.2e}"
    )
    if not info.converged:
        # A residual that "looks" plateaued on a semilogy plot can still be
        # genuinely, if slowly, converging -- plain (non-accelerated)
        # ADMM/LADMM is only guaranteed O(1/k) for a general convex problem,
        # so slope~-1 here means "still converging, just slowly", not stuck.
        print_tail_rates(info)

    # reference_rho_at (imported from sb_gaussian_refine_ends_vs_uniform.py)
    # evaluates at any array of query times directly -- analytical below
    # SINKHORN_VAREPS_THRESHOLD, else the Sinkhorn fit above -- so the
    # staggered (t_edges[1:-1]) and cell-centre (t_centers) references are
    # just two different query-time arrays, no throwaway view needed.
    rho_ana_stag = reference_rho_at(problem.t_edges[1:-1], nx, vareps, sink_fit,
                                    rho_reduce=args.rho_reduce)
    rho_ana_cc = reference_rho_at(problem.t_centers, nx, vareps, sink_fit,
                                  rho_reduce=args.rho_reduce)

    err_cc = np.sqrt(problem.dx * np.sum((result.rho_cc - rho_ana_cc) ** 2, axis=1))
    err_stag = np.sqrt(problem.dx * np.sum((result.rho_stag - rho_ana_stag) ** 2, axis=1))
    t_cc_flat = problem.t_centers
    t_stag_flat = problem.t_edges[1:-1]
    print(
        f"  L2 error in rho vs {ref_name} (cell-centre):  max={err_cc.max():.3e}  mean={err_cc.mean():.3e}  "
        f"(at t={t_cc_flat[np.argmax(err_cc)]:.3f})\n"
        f"  L2 error in rho vs {ref_name} (staggered):    max={err_stag.max():.3e}  mean={err_stag.mean():.3e}  "
        f"(at t={t_stag_flat[np.argmax(err_stag)]:.3f})"
    )

    # --- plot ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ntm = problem.nt - 1
    # 0.0/1.0 included on top of the interior fractions -- clustering shrinks
    # dt most right at the two ends, which is exactly where these grids carry
    # their largest density error (see err_cc/err_stag above), so the
    # boundary-most staggered profiles are worth plotting explicitly rather
    # than only the interior snapshots. searchsorted+clip already maps
    # frac=0.0 -> k=0 and frac=1.0 -> k=ntm-1 (t_stag_flat holds only the
    # interior edges), so no separate indexing path is needed for them.
    t_fracs = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    colors = plt.cm.viridis(np.linspace(0, 1, len(t_fracs)))
    stride = max(1, nx // 60)
    for frac, color in zip(t_fracs, colors):
        k = int(np.clip(np.searchsorted(t_stag_flat, frac), 1, ntm) - 1)
        axes[0, 0].plot(problem.xx, rho_ana_stag[k, :], "-", color=color, lw=1.5)
        axes[0, 0].plot(problem.xx[::stride], result.rho_stag[k, ::stride], "o", color=color, ms=4)

    # The two worst-error profiles in the boundary layers. The fixed fractions
    # above land wherever the grid happens to put a node, which on a graded grid
    # is not where the error actually peaks -- and both ends carry a boundary
    # layer whose height is the quantity of interest (see the endpoint discussion
    # in docs/sinkhorn_reference.tex and etd_theta_projection.tex Sec 9). So
    # single out the largest-error time within the first and last N_EDGE_PROFILE
    # staggered rows and draw those explicitly.
    n_edge = int(min(N_EDGE_PROFILE, err_stag.size))
    k_first = int(np.argmax(err_stag[:n_edge]))
    k_last = int(err_stag.size - n_edge + np.argmax(err_stag[-n_edge:]))
    for k, c, mark, lbl in ((k_first, "crimson", "s", "first"),
                            (k_last, "black", "^", "last")):
        axes[0, 0].plot(problem.xx, rho_ana_stag[k, :], "--", color=c, lw=1.4)
        axes[0, 0].plot(problem.xx[::stride], result.rho_stag[k, ::stride], mark,
                        color=c, ms=5, mfc="none", mew=1.3,
                        label=f"worst of {lbl} {n_edge}: t={t_stag_flat[k]:.4f}, "
                              f"err={err_stag[k]:.2e}")
    print(f"  worst-error times among the first/last {n_edge} staggered rows:  "
          f"t={t_stag_flat[k_first]:.4f} (err={err_stag[k_first]:.3e})   "
          f"t={t_stag_flat[k_last]:.4f} (err={err_stag[k_last]:.3e})")

    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel(r"$\rho$")
    axes[0, 0].set_title(f"Density: {ref_name} (line) vs LADMM (dots)")
    axes[0, 0].legend(fontsize=7.5, loc="upper left")
    axes[0, 0].grid(True)

    axes[0, 1].semilogy(t_cc_flat, err_cc, "-", color="tab:blue", lw=1.5, label="cell-centre (y)")
    axes[0, 1].semilogy(t_stag_flat, err_stag, "-", color="tab:orange", lw=1.5, label="staggered (x)")
    axes[0, 1].set_xlabel("t")
    axes[0, 1].set_ylabel(r"$\|\rho_{LADMM} - \rho_{ref}\|_{L^2(x)}$")
    axes[0, 1].set_title(f"L2 error vs {ref_name} SB")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True)

    # --- LADMM convergence diagnostics (see LadmmInfo in ladmm.py) ---
    iters_ax = np.arange(1, info.iters + 1)
    axes[1, 0].semilogy(*_drop_leading_zero(iters_ax, info.D),
                        "-", color="tab:red", lw=1.2, label=r"dual $\|D^k\|$")
    axes[1, 0].semilogy(*_drop_leading_zero(iters_ax, info.P),
                        "-", color="tab:purple", lw=1.2, label=r"primal $\|P^k\|$")
    axes[1, 0].set_xlabel("iteration")
    axes[1, 0].set_ylabel("residual norm")
    axes[1, 0].set_title("Primal/dual residuals")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True)

    axes[1, 1].semilogy(*_drop_leading_zero(iters_ax, info.dx),
                        "-", color="tab:blue", lw=1.2, label=r"$\|x^{k+1}-x^k\|$")
    axes[1, 1].semilogy(*_drop_leading_zero(iters_ax, info.dz),
                        "-", color="tab:green", lw=1.2, label=r"$\|\delta^{k+1}-\delta^k\|$")
    axes[1, 1].set_xlabel("iteration")
    axes[1, 1].set_ylabel("iterate change")
    axes[1, 1].set_title("Iterate-to-iterate change")
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True)

    fig.suptitle(
        f"SB Gaussian (grid={grid_label}, scheme={args.scheme})  nt={nt} nx={nx} "
        f"eps={vareps:g}\n{sv}:  {detail}"
    )
    fig.tight_layout()

    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    run_tag = f"nt{nt}_nx{nx}_eps{vareps:g}_{grid_tag}_{args.scheme}"
    out_path = out_dir / f"sb_gaussian_nonuniform_{run_tag}.png"
    fig.savefig(out_path, dpi=130)
    print(f"  figure saved to: {out_path}")

    # Raw solve output, saved so a future comparison against a different
    # reference (e.g. re-running Sinkhorn at a tighter tolerance, or vice
    # versa switching analytical<->Sinkhorn near the threshold) doesn't
    # require re-solving the LADMM problem from scratch -- only the plot
    # above needed matplotlib; everything here is arrays LADMM already
    # computed. t_edges/t_centers/dt_vec are enough to rebuild the query
    # points reference_rho_at needs.
    npz_path = out_dir / f"sb_gaussian_nonuniform_{run_tag}.npz"
    np.savez(
        npz_path,
        rho_cc=result.rho_cc, mx_cc=result.mx_cc,
        rho_stag=result.rho_stag, mx_stag=result.mx_stag,
        t_edges=problem.t_edges, t_centers=problem.t_centers, dt_vec=problem.dt_vec,
        nx=nx, vareps=vareps, ref_name=ref_name, scheme=args.scheme,
    )
    print(f"  raw result arrays saved to: {npz_path}")


if __name__ == "__main__":
    main()
