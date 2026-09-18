"""ETD1 vs ETD-theta, end to end through the full LADMM solve.

Same grid-convergence study as grid_convergence.py, run twice per vareps with
only the FP projection swapped, so the difference isolates the scheme.  Both
runs share the identical LadmmConfig and the identical reference solution
(closed-form SB below study_utils.SINKHORN_VAREPS_THRESHOLD, Sinkhorn above).

See projection_expsemi_theta.py for what ETD-theta changes and why.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from optimal_transport.pipeline import LadmmConfig
from optimal_transport.projection_expsemi import (
    precomp_expsemi_proj,
    proj_fokker_planck_expsemi,
)
from optimal_transport.projection_expsemi_theta import (
    precomp_expsemi_theta_proj,
    proj_fokker_planck_expsemi_theta,
)
from study_utils import (
    SINKHORN_MAX_ITER,
    SINKHORN_NX_FINE,
    SINKHORN_TOL,
    SINKHORN_VAREPS_THRESHOLD,
    l2_error,
    reference_solution,
    run_study,
    sinkhorn_fit,
)
from optimal_transport.grid import setup_problem
from optimal_transport.operators import build_operators
from optimal_transport.pipeline import discretize_then_optimize
from optimal_transport.problems import prob_gaussian

SCHEMES = {
    "ETD1": (precomp_expsemi_proj, proj_fokker_planck_expsemi),
    "ETD-theta": (precomp_expsemi_theta_proj, proj_fokker_planck_expsemi_theta),
}


def orders(resolutions, errs):
    return [np.nan] + [np.log(errs[i - 1] / errs[i]) / np.log(resolutions[i] / resolutions[i - 1])
                       for i in range(1, len(errs))]


def solve_one(nt, nx, vareps, cfg, precomp, projection, sink_fit):
    problem = setup_problem(prob_gaussian(), nt=nt, nx=nx)
    problem.ops = build_operators(problem)
    ep = precomp(problem, vareps)
    res = discretize_then_optimize(problem, projection, ep, vareps, cfg)
    r_stag, m_stag, _, _ = reference_solution(problem, vareps, problem.ops, sink_fit)
    return l2_error(res.rho_stag, res.mx_stag, r_stag, m_stag, problem.dt, problem.dx), res.info


def _sink(vareps, verbose=True):
    """One Sinkhorn reference fit, shared by both schemes (run_study refits per
    call, which would waste a fine-grid solve per scheme)."""
    if vareps <= SINKHORN_VAREPS_THRESHOLD:
        return None
    fine = setup_problem(prob_gaussian(), nt=SINKHORN_NX_FINE, nx=SINKHORN_NX_FINE)
    f = sinkhorn_fit(fine, vareps, SINKHORN_MAX_ITER, SINKHORN_TOL)
    if verbose:
        print(f"  (Sinkhorn reference: iters={f.iters} converged={f.converged} "
              f"err={f.error:.2e} wall={f.walltime:.1f}s)")
    return f


def _table(label, xs, xlabel, errs):
    print(f"\n  --- {label} ---")
    print(f"  {xlabel:>5} | " + " | ".join(f"{s:>12} {'order':>6}" for s in SCHEMES)
          + " |  ETD1/theta")
    o = {s: orders(xs, errs[s]) for s in SCHEMES}
    for i, n in enumerate(xs):
        row = f"  {n:5d} | "
        row += " | ".join(f"{errs[s][i]:12.5e} {o[s][i]:6.2f}" for s in SCHEMES)
        row += f" |  {errs['ETD1'][i] / errs['ETD-theta'][i]:9.2f}x"
        print(row)


def main():
    # gamma=1 (not 100): at gamma=100 the LADMM iterate is still solver-limited
    # at these sizes and the scheme difference is invisible underneath it.  At
    # gamma=1 the error is unchanged between 20k and 100k iterations, i.e. it is
    # discretization-limited, which is what this study needs to measure.
    cfg = LadmmConfig(gamma=1.0, tau=1.01, max_iter=20_000, eps_abs=1e-10, eps_rel=1e-15)

    print("=" * 74)
    print("JOINT REFINEMENT  nt = nx = n   (the conventional study)")
    print("=" * 74)
    for vareps in [1e-3, 1e-1, 1.0]:
        print(f"\nvareps = {vareps:g}")
        sf = _sink(vareps)
        ns = [16, 32, 64]
        errs = {name: [solve_one(n, n, vareps, cfg, pc, pj, sf)[0] for n in ns]
                for name, (pc, pj) in SCHEMES.items()}
        _table(f"vareps = {vareps:g}", ns, "n", errs)

    print("\n" + "=" * 74)
    print("dt-ONLY REFINEMENT  nx = 64 fixed   (isolates the temporal error,")
    print("                                     which is all ETD-theta changes)")
    print("=" * 74)
    for vareps in [1.0, 10.0]:
        print(f"\nvareps = {vareps:g}")
        sf = _sink(vareps)
        nts = [16, 32, 64, 128]
        errs = {name: [solve_one(nt, 64, vareps, cfg, pc, pj, sf)[0] for nt in nts]
                for name, (pc, pj) in SCHEMES.items()}
        _table(f"vareps = {vareps:g}, nx=64", nts, "nt", errs)


if __name__ == "__main__":
    main()
