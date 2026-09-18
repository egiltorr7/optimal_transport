"""Schrodinger bridge between two Gaussians: N(1/3, 0.05^2) -> N(2/3, 0.05^2).

Simplest end-to-end check of the ported pipeline: staggered discretization +
expsemi (exact heat-semigroup) FP projection + cubic KE prox + linearized
ADMM. The banded (Pade/Crank-Nicolson) projection produces visible
oscillatory ringing in the diffusion step at large vareps*dt/dx^2 -- see
projection_expsemi.py and cfg_ladmm_gaussian_expsemi.m in MATLAB, which
switches to the exact exponential semigroup for exactly this reason.
Compared against either the closed-form analytical SB solution (vareps <=
study_utils.SINKHORN_VAREPS_THRESHOLD) or the Sinkhorn-Hopf-Cole solution on
a fine grid (above it) -- see study_utils.reference_solution.

Mirrors matlab/discretize_first/1d/experiments/test_expsemi.m.

The FP projection scheme and every run parameter are set in the CONFIG block
below -- edit them there and just run the file:

    python scripts/sb_gaussian.py

The same values are also exposed as optional command-line overrides, for
sweeping without editing the file (any flag left off keeps the CONFIG value):

    python scripts/sb_gaussian.py --scheme CN --vareps 1.0
    python scripts/sb_gaussian.py --sinkhorn-grid same --vareps 1.0

The three schemes share the same (problem, vareps) / (state, problem, vareps,
precomp) signatures, so they drop into discretize_then_optimize
interchangeably. See docs/etd_theta_projection.tex for what ETD-theta
changes; its gap over ETD grows with vareps and is nil at vareps=0 by
construction.
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

from optimal_transport.grid import setup_problem
from optimal_transport.operators import build_operators
from optimal_transport.pipeline import LadmmConfig, discretize_then_optimize
from optimal_transport.problems import prob_gaussian
from optimal_transport.projection import precomp_banded_proj, proj_fokker_planck_banded
from optimal_transport.projection_expsemi import precomp_expsemi_proj, proj_fokker_planck_expsemi
from optimal_transport.projection_expsemi_theta import (
    precomp_expsemi_theta_proj,
    proj_fokker_planck_expsemi_theta,
)
from optimal_transport.sinkhorn import sinkhorn_fit
from study_utils import (
    SINKHORN_GRIDS,
    SINKHORN_NX_FINE,
    SINKHORN_VAREPS_THRESHOLD,
    reference_solution,
    sinkhorn_reference_fit,
)

SCHEMES = {
    "CN": (precomp_banded_proj, proj_fokker_planck_banded),
    "ETD": (precomp_expsemi_proj, proj_fokker_planck_expsemi),
    "ETD-theta": (precomp_expsemi_theta_proj, proj_fokker_planck_expsemi_theta),
}

# ===========================================================================
# CONFIG -- edit these, then run `python scripts/sb_gaussian.py`.
# (Each is also overridable on the command line; see the module docstring.)
# ===========================================================================
SCHEME = "ETD"          # "CN" | "ETD" | "ETD-theta"   (key of SCHEMES above)
VAREPS = 0.1            # diffusion coefficient
NT = 128                # time cells
NX = 128                # space cells
GAMMA = 1.0             # LADMM penalty; tau is set to gamma*101/100 below
MAX_ITER = 20_000

# Reference solution (only used when VAREPS > SINKHORN_VAREPS_THRESHOLD, below
# which the closed-form analytical SB is used and neither setting applies).
# See docs/sinkhorn_reference.tex for what the choice actually measures.
SINKHORN_GRID = "fine"  # "fine": fit at SINKHORN_NX_FINE and reduce to NX. The
                        #         reference is ~the CONTINUUM solution, so this
                        #         measures the solver's TOTAL discretization
                        #         error -- use for convergence studies.
                        # "same": fit on this run's own NX. No reduction, and the
                        #         marginals are the solver's own exactly. Shares
                        #         the solver's spatial discretization, so it
                        #         isolates the TIME error -- but is blind to
                        #         spatial error by construction.
RHO_REDUCE = "spectral"  # SINKHORN_GRID="fine" only: how rho reaches the coarse
                         # grid. "spectral" | "pointwise" | "average".
                         # "average" is the old behaviour and leaves a spurious
                         # O(dx^2/dt) floor at t=0 and t=1 -- don't use it except
                         # to reproduce pre-fix numbers.
# ===========================================================================


def parse_args():
    """CLI overrides for the CONFIG block; defaults ARE the CONFIG values, so
    running with no flags is identical to running the hard-coded settings."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scheme", choices=sorted(SCHEMES), default=SCHEME)
    ap.add_argument("--vareps", type=float, default=VAREPS)
    ap.add_argument("--nt", type=int, default=NT)
    ap.add_argument("--nx", type=int, default=NX)
    ap.add_argument("--gamma", type=float, default=GAMMA)
    ap.add_argument("--max-iter", type=int, default=MAX_ITER)
    ap.add_argument("--sinkhorn-grid", choices=list(SINKHORN_GRIDS), default=SINKHORN_GRID,
                    help="spatial grid for the Sinkhorn reference fit (see CONFIG)")
    ap.add_argument("--rho-reduce", choices=["spectral", "pointwise", "average"],
                    default=RHO_REDUCE,
                    help="--sinkhorn-grid=fine only: how rho reaches the coarse grid")
    return ap.parse_args()


def main(scheme=SCHEME, vareps=VAREPS, nt=NT, nx=NX, gamma=GAMMA, max_iter=MAX_ITER,
         sinkhorn_grid=SINKHORN_GRID, rho_reduce=RHO_REDUCE):
    """Runs one solve. Called with no arguments it uses the CONFIG block, so it
    is also importable: `from sb_gaussian import main; main(scheme="CN")`."""
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; choose from {sorted(SCHEMES)}")
    precomp_fn, proj_fn = SCHEMES[scheme]

    prob_def = prob_gaussian()
    problem = setup_problem(prob_def, nt=nt, nx=nx)
    problem.ops = build_operators(problem)
    ops = problem.ops

    ep = precomp_fn(problem, vareps)
    ladmm_cfg = LadmmConfig(gamma=gamma, tau=gamma*(101.0/100.0), max_iter=max_iter,
                            eps_abs=1e-10, eps_rel=1e-15)

    print(
        f"Running SB Gaussian: scheme={scheme} nt={nt} nx={nx} "
        f"gamma={ladmm_cfg.gamma:g} tau={ladmm_cfg.tau:g} eps={vareps:g} ..."
    )
    result = discretize_then_optimize(problem, proj_fn, ep, vareps, ladmm_cfg)
    info = result.info
    print(
        f"  iters={info.iters}  converged={info.converged}  wall={info.walltime:.2f}s\n"
        f"  final residuals:  dx={info.dx[-1]:.2e}  dy={info.dy[-1]:.2e}  "
        f"dz={info.dz[-1]:.2e}  D={info.D[-1]:.2e}  P={info.P[-1]:.2e}"
    )

    # --- reference solution: analytical (small vareps) or Sinkhorn (large) ---
    sink_fit = None
    ref_name = "analytical"
    if vareps > SINKHORN_VAREPS_THRESHOLD:
        ref_name = f"Sinkhorn/{sinkhorn_grid}"
        print(f"vareps={vareps:g} > {SINKHORN_VAREPS_THRESHOLD:g}: using a Sinkhorn "
              f"reference instead of the analytical solution.")
        sink_fit = sinkhorn_reference_fit(
            problem, vareps, grid=sinkhorn_grid,
            make_fine_problem=lambda n: setup_problem(prob_gaussian(), nt=n, nx=n),
        )

    rho_ana_stag, _, rho_ana_cc, _ = reference_solution(
        problem, vareps, ops, sink_fit, rho_reduce=rho_reduce
    )  # (ntm, nx), (nt, nx)

    # rho is a genuine density now, so the L2(x) error at each time is a
    # direct Riemann-sum weighting by dx -- no correction needed (unlike the
    # old mass-per-cell convention, which needed 1/dx). Both grid variables:
    # cell-centre (y, the KE-optimal variable) and staggered (x, the
    # FP-feasible variable).
    err_cc = np.sqrt(problem.dx * np.sum((result.rho_cc - rho_ana_cc) ** 2, axis=1))
    err_stag = np.sqrt(problem.dx * np.sum((result.rho_stag - rho_ana_stag) ** 2, axis=1))
    t_cc = (np.arange(1, nt + 1) - 0.5) * problem.dt
    t_stag = np.arange(1, nt) * problem.dt
    print(
        f"  L2 error in rho (cell-centre):  max={err_cc.max():.3e}  mean={err_cc.mean():.3e}  "
        f"(at t={t_cc[np.argmax(err_cc)]:.3f})\n"
        f"  L2 error in rho (staggered):    max={err_stag.max():.3e}  mean={err_stag.mean():.3e}  "
        f"(at t={t_stag[np.argmax(err_stag)]:.3f})"
    )

    # --- plot ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Match matlab/discretize_first/1d/experiments/test_expsemi.m exactly:
    # fixed fractions (not cosine-clustered near t=0,1), plotted on the
    # staggered grid (rho_stag), with its 1-indexed k = max(1,min(ntm,round(frac*nt))).
    ntm = nt - 1
    t_fracs = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    colors = plt.cm.viridis(np.linspace(0, 1, len(t_fracs)))
    stride = max(1, nx // 60)
    for frac, color in zip(t_fracs, colors):
        k = int(np.clip(round(frac * nt), 1, ntm)) - 1  # 0-indexed row into rho_stag
        axes[0].plot(problem.xx, rho_ana_stag[k, :], "-", color=color, lw=1.5)
        axes[0].plot(problem.xx[::stride], result.rho_stag[k, ::stride], "o", color=color, ms=4)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel(r"$\rho$")
    axes[0].set_title(f"Density: {ref_name} (line) vs LADMM (dots)")
    axes[0].grid(True)

    axes[1].semilogy(t_cc, err_cc, "-", color="tab:blue", lw=1.5, label="cell-centre (y)")
    axes[1].semilogy(t_stag, err_stag, "-", color="tab:orange", lw=1.5, label="staggered (x)")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel(r"$\|\rho_{LADMM} - \rho_{ref}\|_{L^2(x)}$")
    axes[1].set_title(f"L2 error vs {ref_name} SB")
    axes[1].legend(fontsize=9)
    axes[1].grid(True)

    fig.suptitle(
        f"SB Gaussian [{scheme}]  nt={nt} nx={nx} gamma={ladmm_cfg.gamma:g} "
        f"tau={ladmm_cfg.tau:g} eps={vareps:g}  (ref: {ref_name})"
    )
    fig.tight_layout()

    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    ref_tag = "" if vareps <= SINKHORN_VAREPS_THRESHOLD else f"_{sinkhorn_grid}"
    out_path = out_dir / f"sb_gaussian_{scheme}{ref_tag}_nt{nt}_nx{nx}_eps{vareps:g}.png"
    fig.savefig(out_path, dpi=130)
    print(f"  figure saved to: {out_path}")


if __name__ == "__main__":
    a = parse_args()
    main(scheme=a.scheme, vareps=a.vareps, nt=a.nt, nx=a.nx,
         gamma=a.gamma, max_iter=a.max_iter,
         sinkhorn_grid=a.sinkhorn_grid, rho_reduce=a.rho_reduce)
