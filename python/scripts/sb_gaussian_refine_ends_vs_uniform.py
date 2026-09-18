"""Compare graded_ends_time_grid against a plain uniform time grid, each run
through *both* FP projection schemes (CN/banded and ETD), all sharing the
identical non-uniform-time solver scaffolding (grid_nu, operators_nu,
pipeline_nu -- see time_grid.py / docs/dyadic_graded_ladmm.tex).
graded_ends_time_grid is the natural next step past refine_ends_time_grid:
instead of splitting just the one boundary cell, it dyadically grades the
first/last N_BOUNDARY cells (each one half the width of its next-further
neighbor) -- refine_ends is exactly graded_ends with n_boundary=1 (see
time_grid.py's docstring / validate_nu.py). ETD (projection_expsemi_nu.py)
is included alongside CN (projection_nu.py) because grading shrinks dt near
the boundary, which *increases* the local eps/dt ratio exactly where CN is
weakest per the main paper's own Fig. 6 -- see docs/dyadic_graded_ladmm.tex
Section 7 for the derivation and a first (inconclusive) head-to-head test.

Deliberately does NOT compare against sb_gaussian.py's uniform-grid pipeline
(grid.py/operators.py/projection_expsemi.py): that uses a different FP
projection algorithm (exact heat-semigroup) than projection_nu.py's banded
solve, so any difference would conflate "different grid" with "different
projection algorithm". Running uniform_time_grid(nt) through the exact same
*_nu code path as refine_ends_time_grid(nt) isolates the grid as the only
variable -- and validate_nu.py already confirms that path reduces exactly to
sb_gaussian.py's own reference on a uniform grid, so nothing is lost by using
it here as the baseline.

Reference solution: the closed-form analytical SB (analytical_sb_gaussian_nu,
free-space Brownian motion) below SINKHORN_VAREPS_THRESHOLD (5e-3), else a
Sinkhorn-Hopf-Cole fit (sinkhorn.py) -- the analytical formula assumes the
density's spread stays small relative to the domain, which breaks down once
vareps is large enough for the density to feel the (reflecting) boundary, so
the two grids would otherwise just be racing to match an already-wrong
target. Following study_utils.reference_solution's convention: the Sinkhorn
fit is done ONCE on a fine common spatial grid (SINKHORN_NX_FINE=4096, same
constant study_utils.py uses) and rho is then brought down to the run's own nx
by reference_rho_at -- by default with the spectral interpolant
(study_utils.resample_rho_spectral), which sums the fine grid's own DCT-II
series at the coarse cell centres and so matches the solver's own rho0/rho1 to
~1e-15. Do NOT block-average: coarsen_rho disagrees with those marginals by
the O(dx^2) coarsening error, and since the marginals are baked into A exactly
at t=0 and t=1 that leaves a spurious floor at both endpoints (and, seen
through the FP residual, an O(dx^2/dt) term that GROWS under time refinement).
See docs/sinkhorn_reference.tex. Fitting on the fine grid rather than directly
at nx avoids letting the LADMM solve partly validate against a reference built
at its own coarse resolution; study_utils.sinkhorn_reference_fit offers the
"same"-grid alternative for when isolating the TIME error is what is wanted
instead. SINKHORN_NX_FINE=4096=16*2^8 divides evenly into nx=16*2^k for
every k=1..8, so the same fine grid can serve as a shared reference across
that whole family if this script is ever extended to sweep nx too (today it
only varies the time grid, so both grids under comparison already share one
nx, but the fine-grid reference costs little and removes the self-reference
concern regardless). sinkhorn_eval evaluates at an arbitrary array of t, so
each grid's own (non-uniform) t_edges/t_centers are used directly on the
fine grid, no time resampling -- only space is coarsened.

Usage:
    python3 scripts/sb_gaussian_refine_ends_vs_uniform.py --nt 128 --nx 128 --vareps 1e-3
    python3 scripts/sb_gaussian_refine_ends_vs_uniform.py --nt 64 --nx 64 --vareps 1e-4 --match-total-cells
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
from optimal_transport.operators_nu import build_operators_nu
from optimal_transport.pipeline_nu import LadmmConfig, discretize_then_optimize_nu
from optimal_transport.problems import prob_gaussian
from optimal_transport.problems_nu import analytical_sb_gaussian_nu
from optimal_transport.projection_expsemi_nu import precomp_expsemi_proj_nu, proj_fokker_planck_expsemi_nu
from optimal_transport.projection_nu import precomp_banded_proj_nu, proj_fokker_planck_banded_nu
from optimal_transport.sinkhorn import sinkhorn_eval, sinkhorn_fit
from optimal_transport.time_grid import (
    clustered_time_grid,
    graded_ends_time_grid,
    refine_ends_time_grid,
    uniform_time_grid,
)
from study_utils import (SINKHORN_MAX_ITER, SINKHORN_NX_FINE, SINKHORN_TOL,
                         SINKHORN_VAREPS_THRESHOLD, coarsen_rho, resample_rho_spectral,
                         sample_rho_pointwise)

# ============================================================================
# Edit these directly to change the default run -- every one is still
# overridable from the command line (e.g. `--nt 64`), but for repeated local
# tinkering it's usually less tedious to just edit the numbers here and run
# `python3 sb_gaussian_refine_ends_vs_uniform.py` with no flags.
# ============================================================================
NT = 256             # base time resolution. refine_ends gets nt+2 cells;
                       # uniform gets nt cells (or nt+2 too, with MATCH_TOTAL_CELLS)
NX = 256
VAREPS = 100.0          # keep < SINKHORN_VAREPS_THRESHOLD (5e-3) for the analytical
                       # SB reference to stay valid
GAMMA = 0.1
TAU = None             # None -> gamma * 101/100
MAX_ITER = 20_000
EPS_ABS = 1e-10
EPS_REL = 1e-15
MATCH_TOTAL_CELLS = False  # True: run the uniform baseline at nt+2 cells instead of
                            # nt, matching refine_ends's cell count (graded_ends's own,
                            # much larger, cell count is not matched by this flag)
N_BOUNDARY = 5              # graded_ends_time_grid's n_boundary: dyadically grades the
                             # first/last N_BOUNDARY base cells (2*N_BOUNDARY must be <= nt)
CHEB_STRENGTH = 1.0          # clustered_time_grid's strength: 0=uniform, 1=full cosine
                             # (Chebyshev-Gauss-Lobatto-like) clustering at both ends
# ============================================================================


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nt", type=int, default=NT,
                    help="base time resolution. refine_ends gets nt+2 cells; uniform gets nt cells "
                         "(or nt+2 too, with --match-total-cells)")
    p.add_argument("--nx", type=int, default=NX)
    p.add_argument("--vareps", type=float, default=VAREPS,
                    help="keep < %.0e (study_utils.SINKHORN_VAREPS_THRESHOLD) for the analytical SB "
                         "reference to be valid" % SINKHORN_VAREPS_THRESHOLD)
    p.add_argument("--gamma", type=float, default=GAMMA)
    p.add_argument("--tau", type=float, default=TAU, help="default: gamma * 101/100")
    p.add_argument("--max-iter", type=int, default=MAX_ITER)
    p.add_argument("--eps-abs", type=float, default=EPS_ABS)
    p.add_argument("--eps-rel", type=float, default=EPS_REL)
    p.add_argument("--match-total-cells", action="store_true", default=MATCH_TOTAL_CELLS,
                    help="run the uniform baseline at nt+2 cells instead of nt, matching refine_ends's "
                         "cell count (graded_ends's own, much larger, cell count is not matched)")
    p.add_argument("--n-boundary", type=int, default=N_BOUNDARY,
                    help="graded_ends_time_grid's n_boundary: dyadically grades the first/last "
                         "n_boundary base cells (2*n_boundary must be <= nt)")
    p.add_argument("--cheb-strength", type=float, default=CHEB_STRENGTH,
                    help="clustered_time_grid's strength in [0,1]: 0=uniform, 1=full cosine "
                         "(Chebyshev-Gauss-Lobatto-like) clustering")
    return p.parse_args()


def reference_rho_at(t_array, nx, vareps, sink_fit=None, rho_reduce="spectral"):
    """Reference (analytical or Sinkhorn) density profile at arbitrary times
    t_array, on an nx-cell spatial grid. Shape (len(t_array), nx).

    Used both by run_one (queried at each grid's own native t_centers/
    t_edges, for the error metrics) and by main's endpoint-comparison plot
    (queried at a single *shared* target time, so refine_ends's profile can
    be compared to uniform's at the exact same instant).
    """
    t_array = np.atleast_1d(t_array)
    if sink_fit is None:
        pd = prob_gaussian()
        nx_, dx_, mu0_, mu1_, sigma_ = nx, 1.0 / nx, pd.mu0, pd.mu1, pd.sigma

        class _View:
            t_edges = np.concatenate([[0.0], t_array, [1.0]])
            t_centers = t_array  # only used for mx, which we discard below
            nx, dx, mu0, mu1, sigma = nx_, dx_, mu0_, mu1_, sigma_

        rho, _ = analytical_sb_gaussian_nu(_View, vareps)
        return rho

    # sinkhorn_eval evaluates at any t directly (exact spectral heat kernel,
    # no fixed grid) -- so each grid's own irregular t_edges/t_centers can be
    # passed straight in, no time resampling.
    #
    # In SPACE, sink_fit may live either on a fine common grid (then rho must be
    # reduced to nx) or on the solver's own grid (then every reducer below is
    # the identity, since the factor is 1). Never block-average: the solver's
    # own rho0/rho1 are pointwise samples, so a block-averaged reference
    # disagrees with them by the O(dx^2) coarsening error and puts a spurious
    # floor under the error at t=0 and t=1. See docs/sinkhorn_reference.tex.
    rho_fine, _ = sinkhorn_eval(sink_fit, t_array)
    if rho_reduce == "spectral":
        return resample_rho_spectral(rho_fine, nx)
    if rho_reduce == "pointwise":
        return sample_rho_pointwise(rho_fine, nx)
    if rho_reduce == "average":
        return coarsen_rho(rho_fine, rho_fine.shape[-1] // nx)
    raise ValueError(f"unknown rho_reduce={rho_reduce!r}")


def interp_profile_at_t(rho_cc, t_centers, t_query):
    """Linear interpolation in time of a (nt, nx) cell-centre density field
    to an arbitrary query time t_query (a scalar), from its two nearest
    t_centers. Used to compare refine_ends's density profile against
    uniform's at uniform's own cell-centre time, rather than refine_ends's
    own (finer, closer-to-the-boundary) native cell centres.

    Reduces to a plain average when t_query is exactly the midpoint of the
    two bracketing t_centers -- e.g. refine_ends's first two cell centres
    bracket the original (pre-split) cell's centre exactly at their
    midpoint by construction, so this recovers the same "midpoint
    interpolation is exact" property used throughout the solver (see
    docs/nonuniform_grid_refine_ends.tex, Lemma 1).
    """
    idx = int(np.clip(np.searchsorted(t_centers, t_query), 1, len(t_centers) - 1))
    t0, t1 = t_centers[idx - 1], t_centers[idx]
    w = (t_query - t0) / (t1 - t0)
    return (1 - w) * rho_cc[idx - 1, :] + w * rho_cc[idx, :]


# scheme label -> (precompute_fn, projection_fn), both taking the same
# (problem, vareps) / (state, problem, vareps, precomp) signatures.
SCHEMES = {
    "CN": (precomp_banded_proj_nu, proj_fokker_planck_banded_nu),
    "ETD": (precomp_expsemi_proj_nu, proj_fokker_planck_expsemi_nu),  # re-add to compare again
}


def run_one(time_grid, nx, vareps, ladmm_cfg, scheme, sink_fit=None):
    precomp_fn, proj_fn = SCHEMES[scheme]
    prob_def = prob_gaussian()
    problem = setup_problem_nu(prob_def, time_grid, nx=nx)
    problem.ops = build_operators_nu(problem)
    bp = precomp_fn(problem, vareps)

    t0 = time.perf_counter()
    result = discretize_then_optimize_nu(problem, proj_fn, bp, vareps, ladmm_cfg)
    wall = time.perf_counter() - t0

    rho_ana_stag = reference_rho_at(problem.t_edges[1:-1], nx, vareps, sink_fit)
    rho_ana_cc = reference_rho_at(problem.t_centers, nx, vareps, sink_fit)

    err_cc = np.sqrt(problem.dx * np.sum((result.rho_cc - rho_ana_cc) ** 2, axis=1))
    err_stag = np.sqrt(problem.dx * np.sum((result.rho_stag - rho_ana_stag) ** 2, axis=1))

    return dict(
        problem=problem, result=result, wall=wall,
        t_cc=problem.t_centers, t_stag=problem.t_edges[1:-1],
        err_cc=err_cc, err_stag=err_stag, rho_ana_cc=rho_ana_cc,
    )


def top_k_pointwise_errors(problem, rho_cc, rho_ana_cc, k=5):
    """The k individual (time, space) cells with the largest pointwise density
    error |rho_cc - rho_ana_cc| -- unlike err_cc (L2-aggregated over space,
    one number per time row), this pinpoints exactly which (t, x) cells are
    worst, largest first. Returns a list of (t, x, err, rho_ladmm, rho_ref).
    """
    diff = np.abs(rho_cc - rho_ana_cc)
    flat = diff.ravel()
    top = np.argpartition(flat, -k)[-k:]
    top = top[np.argsort(-flat[top])]
    rows, cols = np.unravel_index(top, diff.shape)
    return [
        (problem.t_centers[row], problem.xx[col], diff[row, col], rho_cc[row, col], rho_ana_cc[row, col])
        for row, col in zip(rows, cols)
    ]


def main():
    args = parse_args()

    sink_fit = None
    ref_name = "analytical"
    if args.vareps > SINKHORN_VAREPS_THRESHOLD:
        ref_name = "Sinkhorn"
        assert SINKHORN_NX_FINE % args.nx == 0, (
            f"SINKHORN_NX_FINE={SINKHORN_NX_FINE} must be a multiple of nx={args.nx} to coarsen down to it"
        )
        print(f"vareps={args.vareps:g} > {SINKHORN_VAREPS_THRESHOLD:g}: using a Sinkhorn-Hopf-Cole "
              f"reference (fit once at nx={SINKHORN_NX_FINE}, coarsened to nx={args.nx}) instead of "
              f"the analytical solution.")
        # Sinkhorn's fit only depends on the spatial grid (nx, dx, rho0, rho1,
        # lambda_x), not on nt/dt -- see sinkhorn.py -- so any of our ProblemNU
        # objects would do; a throwaway 2-cell time grid keeps this from
        # depending on either grid under comparison below.
        fit_problem = setup_problem_nu(prob_gaussian(), uniform_time_grid(2), nx=SINKHORN_NX_FINE)
        sink_fit = sinkhorn_fit(fit_problem, args.vareps, SINKHORN_MAX_ITER, SINKHORN_TOL)
        print(f"  Sinkhorn fit: iters={sink_fit.iters}  converged={sink_fit.converged}  "
              f"error={sink_fit.error:.2e}  wall={sink_fit.walltime:.2f}s")

    gamma = args.gamma
    tau = args.tau if args.tau is not None else gamma * (101.0 / 100.0)
    ladmm_cfg = LadmmConfig(gamma=gamma, tau=tau, max_iter=args.max_iter,
                             eps_abs=args.eps_abs, eps_rel=args.eps_rel)

    nt_uniform = args.nt + 2 if args.match_total_cells else args.nt
    grids = {
        "uniform": uniform_time_grid(nt_uniform),
        # "graded_ends": graded_ends_time_grid(args.nt, n_boundary=args.n_boundary),
        "clustered": clustered_time_grid(args.nt, strength=args.cheb_strength),
    }
    # combined key "grid/scheme" -> run; grid_of[key]/scheme_of[key] recover
    # the two factors for coloring/labeling below.
    keys = [f"{g}/{s}" for g in grids for s in SCHEMES]
    grid_of = {f"{g}/{s}": g for g in grids for s in SCHEMES}
    scheme_of = {f"{g}/{s}": s for g in grids for s in SCHEMES}

    print(f"\nComparing {' vs '.join(grids)}  x  {' vs '.join(SCHEMES)}: base nt={args.nt} nx={args.nx} "
          f"vareps={args.vareps:g} gamma={gamma:g} tau={tau:g} max_iter={args.max_iter} "
          f"n_boundary={args.n_boundary}  ref={ref_name}"
          + (" (uniform baseline resolution-matched to nt+2)" if args.match_total_cells else ""))

    runs = {}
    for key in keys:
        tg = grids[grid_of[key]]
        print(f"\n[{key}]  total cells={tg.nt}")
        runs[key] = run_one(tg, args.nx, args.vareps, ladmm_cfg, scheme_of[key], sink_fit=sink_fit)
        r, info = runs[key], runs[key]["result"].info
        print(f"  iters={info.iters}  converged={info.converged}  wall={r['wall']:.2f}s\n"
              f"  final residuals:  dx={info.dx[-1]:.2e}  dy={info.dy[-1]:.2e}  "
              f"dz={info.dz[-1]:.2e}  D={info.D[-1]:.2e}  P={info.P[-1]:.2e}\n"
              f"  L2 error rho (cell-centre):  max={r['err_cc'].max():.3e}  mean={r['err_cc'].mean():.3e}\n"
              f"  L2 error rho (staggered):    max={r['err_stag'].max():.3e}  mean={r['err_stag'].mean():.3e}")
        print("  top-5 pointwise |rho_cc - rho_ana| errors (t, x, err, rho_ladmm, rho_ref):")
        for t, x, err, rl, rr in top_k_pointwise_errors(r["problem"], r["result"].rho_cc, r["rho_ana_cc"]):
            print(f"    t={t:.5f}  x={x:.5f}  |err|={err:.4e}  rho_ladmm={rl:.4e}  rho_ref={rr:.4e}")

    print("\nSummary (mean L2 error, cell-centre rho):")
    for key in keys:
        r = runs[key]
        print(f"  {key:20s}  cells={grids[grid_of[key]].nt:4d}  mean_err={r['err_cc'].mean():.3e}  "
              f"max_err={r['err_cc'].max():.3e}  iters={r['result'].info.iters:6d}  wall={r['wall']:.2f}s")

    # --- endpoint density profiles: uniform's own first/last cell-centre
    # time, vs the reference at that same time, vs every other grid's
    # profile interpolated (in time) to that same time (interp_profile_at_t)
    # -- "uniform" runs need no interpolation, that's the native query point.
    xx = runs[keys[0]]["problem"].xx  # same spatial grid (nx) for every run
    any_uniform_key = next(k for k in keys if grid_of[k] == "uniform")
    uni_problem = runs[any_uniform_key]["problem"]
    t_first, t_last = uni_problem.t_centers[0], uni_problem.t_centers[-1]

    ref_first = reference_rho_at(t_first, args.nx, args.vareps, sink_fit)[0]
    ref_last = reference_rho_at(t_last, args.nx, args.vareps, sink_fit)[0]

    profiles = {}
    for key in keys:
        rho_cc = runs[key]["result"].rho_cc
        if grid_of[key] == "uniform":
            profiles[key] = (rho_cc[0, :], rho_cc[-1, :])
        else:
            t_centers = runs[key]["problem"].t_centers
            profiles[key] = (
                interp_profile_at_t(rho_cc, t_centers, t_first),
                interp_profile_at_t(rho_cc, t_centers, t_last),
            )

    # --- NATIVE endpoint density profiles: each grid's own actual first/
    # last cell-centre time (unlike the block above, NOT aligned to
    # uniform's) vs the reference evaluated at that same (grid-specific)
    # time -- no interpolation anywhere, both sides are queried at the
    # grid's real point. Shows how much closer to the true t=0/1 boundary a
    # refined grid actually reaches, and how accurate it is once it's there.
    # Native times only depend on the grid, not the scheme, so the reference
    # is computed once per grid and shared by both schemes' runs on it.
    native_ref = {}
    for label, tg in grids.items():
        t0, t1 = tg.t_centers[0], tg.t_centers[-1]
        native_ref[label] = (
            (t0, reference_rho_at(t0, args.nx, args.vareps, sink_fit)[0]),
            (t1, reference_rho_at(t1, args.nx, args.vareps, sink_fit)[0]),
        )
    # --- plot ---
    fig = plt.figure(figsize=(11, 15.5))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 0.7])
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(3)])
    ax_grid = fig.add_subplot(gs[3, :])
    grid_colors = dict(zip(grids, plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    scheme_style = {"CN": dict(ls="-", marker="o"), "ETD": dict(ls="--", marker="^")}

    def style(key):
        return dict(color=grid_colors[grid_of[key]], **scheme_style[scheme_of[key]])

    for key in keys:
        r = runs[key]
        axes[0, 0].semilogy(r["t_cc"], r["err_cc"], style(key)["ls"], color=style(key)["color"], lw=1.5,
                             label=f"{key} (cells={grids[grid_of[key]].nt})")
    axes[0, 0].set_xlabel("t")
    axes[0, 0].set_ylabel(r"$\|\rho_{LADMM} - \rho_{ref}\|_{L^2(x)}$")
    axes[0, 0].set_title(f"L2 error vs {ref_name} SB (cell-centre)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True)

    for key in keys:
        r = runs[key]
        axes[0, 1].semilogy(r["t_stag"], r["err_stag"], style(key)["ls"], color=style(key)["color"], lw=1.5,
                             label=f"{key} (cells={grids[grid_of[key]].nt})")
    axes[0, 1].set_xlabel("t")
    axes[0, 1].set_ylabel(r"$\|\rho_{LADMM} - \rho_{ref}\|_{L^2(x)}$")
    axes[0, 1].set_title(f"L2 error vs {ref_name} SB (staggered)")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True)

    stride = max(1, args.nx // 60)
    for ax, t_val, ref_prof, idx, side in [
        (axes[1, 0], t_first, ref_first, 0, "first"),
        (axes[1, 1], t_last, ref_last, 1, "last"),
    ]:
        ax.plot(xx, ref_prof, "-", color="black", lw=1.5, label=f"{ref_name} (t={t_val:.4f})")
        for key in keys:
            prof = profiles[key][idx]
            src = "native" if grid_of[key] == "uniform" else "interp."
            ax.plot(xx[::stride], prof[::stride], style(key)["marker"], color=style(key)["color"], ms=4,
                     mfc="none", label=f"{key} ({src})")
        ax.set_xlabel("x")
        ax.set_ylabel(r"$\rho$")
        ax.set_title(f"{side.capitalize()} cell-centre time vs {ref_name}")
        ax.legend(fontsize=7)
        ax.grid(True)

    for ax, side_idx, side in [(axes[2, 0], 0, "first"), (axes[2, 1], 1, "last")]:
        for label in grids:
            t_val, ref_prof = native_ref[label][side_idx]
            ax.plot(xx, ref_prof, "--", color=grid_colors[label], lw=1.2, alpha=0.7,
                     label=f"{ref_name} @ {label} t={t_val:.4f}")
        for key in keys:
            prof = runs[key]["result"].rho_cc[0, :] if side_idx == 0 else runs[key]["result"].rho_cc[-1, :]
            ax.plot(xx[::stride], prof[::stride], style(key)["marker"], color=style(key)["color"], ms=4,
                     mfc="none", label=f"{key} (native)")
        ax.set_xlabel("x")
        ax.set_ylabel(r"$\rho$")
        ax.set_title(f"{side.capitalize()} NATIVE cell-centre time, per grid vs {ref_name}")
        ax.legend(fontsize=7)
        ax.grid(True)

    # --- grid points in time: one row per grid, a tick at every edge, so
    # clustering/grading patterns are directly visible and comparable.
    for i, (label, tg) in enumerate(grids.items()):
        ax_grid.plot(tg.t_edges, np.full_like(tg.t_edges, i), "|", color=grid_colors[label],
                     ms=14, mew=1.3)
    ax_grid.set_yticks(range(len(grids)))
    ax_grid.set_yticklabels([f"{label} ({tg.nt} cells)" for label, tg in grids.items()])
    ax_grid.set_ylim(-0.6, len(grids) - 0.4)
    ax_grid.set_xlim(-0.02, 1.02)
    ax_grid.set_xlabel("t")
    ax_grid.set_title("Grid edge positions in time")
    ax_grid.grid(True, axis="x")

    fig.suptitle(
        f"{' vs '.join(grids)}  x  {' vs '.join(SCHEMES)}  base_nt={args.nt} nx={args.nx} "
        f"eps={args.vareps:g} gamma={gamma:g} tau={tau:g} n_boundary={args.n_boundary} "
        f"cheb_strength={args.cheb_strength:g}  (ref: {ref_name})",
        fontsize=10,
    )
    fig.tight_layout()

    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    tag = "_matched" if args.match_total_cells else ""
    out_path = out_dir / f"sb_gaussian_refine_ends_vs_uniform_nt{args.nt}_nx{args.nx}_eps{args.vareps:g}{tag}.png"
    fig.savefig(out_path, dpi=130)
    print(f"\nfigure saved to: {out_path}")


if __name__ == "__main__":
    main()
