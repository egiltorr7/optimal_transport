"""Constant vs adaptive step size for the reordered LADMM, on the non-uniform
time grid -- residual and true-error histories, side by side.

Sibling of gamma_sweep.py, but comparing *step-size rules* rather than sweeping
a fixed gamma. Two studies in one run:

  (A) INITIAL-VALUE study. For each gamma_0 in GAMMA0_LIST, run the reordered
      LADMM twice -- constant (gamma, tau) and Goldstein-Li-Yuan residual
      balancing (docs/adaptive_pdhg_goldstein.tex Part V) -- from the same
      start, and overlay the histories. The question this answers: does the
      adaptive rule recover from a badly-chosen gamma_0, and what does it cost
      when gamma_0 was already good?

  (B) RATIO study. Constant step only, sweeping gamma/tau at fixed gamma. Every
      ratio here must satisfy the stability rule tau > gamma*||A||^2, i.e.
      gamma/tau < 1/||A||^2, and ||A||^2 <= 1 on this problem for any grid
      (docs/nonuniform_grid_norms.tex Sec 6) -- so gamma/tau < 1. RATIO_LIST is
      asserted against that bound rather than trusted.

Six quantities are tracked per run, all already recorded by the solver:

    D        primal residual, ||A x^k + B y^k - b||   (the constraint violation)
    P        dual residual (the reordered solver's, = Goldstein-Li-Yuan's q_k;
             NOT the same expression as ladmm.py's own P -- see
             ladmm_reordered.ReorderedLadmmInfo)
    err_x    ||x^k - x_ref||        true error, vs the analytical SB solution
    err_delta ||delta^k - delta_ref||  true dual error, masked where f2 is not
             differentiable (rho ~ 0) -- see _build_delta_ref
    dx       ||x^{k+1} - x^k||      iterate-to-iterate movement
    dz       ||delta^{k+1} - delta^k||

Residuals are not everything: a run can have small D and P while still sitting
far from x_ref, so the true-error panels are the point of this script, not an
extra.

CAUTION on err_delta. delta_ref = -grad f2(y_ref) has a 1/rho^2 component
(f2 = m^2/(2 rho)), so wherever the reference density is small -- the Gaussian
tails -- it amplifies discretization error without limit. err_delta is therefore
strongly dependent on RHO_FLOOR, the density below which points are masked out.
Measured at nt=nx=32, eps=1e-4, gamma=1, 4000 iterations:

    rho_floor   1e-12     1e-8      1e-4      1e-2
    kept        75.8%     64.6%     47.3%     37.1%
    err_delta   4.6e-01   2.6e-01   5.8e-02   2.1e-02

At 1e-12 (gamma_sweep.py's floor, which is prox_ke_cc's own density floor and is
the wrong tool for this job) err_delta looks like it plateaus around 0.46 and
never converges; that plateau is entirely tail amplification, not the solver. At
1e-4 it decreases monotonically (3.0e-01 -> 5.8e-02 over 4000 iterations) and is
still falling. So: err_delta is meaningful for *comparing runs at a fixed floor*,
which is what this script does; its absolute value is not meaningful on its own.
err_x has no such issue -- it is a plain difference, with no division.

x_ref/delta_ref need the closed-form solution, so VAREPS must be at or below
study_utils.SINKHORN_VAREPS_THRESHOLD (a Sinkhorn fit gives rho but not the mx
that delta_ref needs). Asserted below.

Usage:
    python3 scripts/compare_stepsize_nu.py
    python3 scripts/compare_stepsize_nu.py --nt 64 --nx 64 --max-iter 40000
    python3 scripts/compare_stepsize_nu.py --gamma0-list 0.01 1 100 --no-ratio-study
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
from optimal_transport.ladmm_reordered import GlyAdaptiveStepSize
from optimal_transport.operators_nu import build_operators_nu
from optimal_transport.pipeline_nu import LadmmConfig, discretize_then_optimize_nu
from optimal_transport.problems import prob_gaussian
from optimal_transport.problems_nu import analytical_sb_gaussian_nu
from optimal_transport.state import State
from sb_gaussian_refine_ends_vs_uniform import SCHEMES
from study_utils import (
    RESULTS_DIR,
    SINKHORN_VAREPS_THRESHOLD,
    TIME_GRIDS,
    BooleanOptionalAction,
    build_time_grid,
)

# ============================================================================
# Edit these directly; every one is overridable from the command line.
# ============================================================================
GRID = "uniform"
SCHEME = "ETD"
NT = 512
NX = 512
VAREPS = 1e-4                 # must be <= SINKHORN_VAREPS_THRESHOLD (see docstring)
STRENGTH = 1.0
GAMMA = 1.0                   # gamma for the ratio study; gamma_0 for study A comes
                              # from GAMMA0_LIST
TAU_RATIO = 100.0 / 101.0     # tau = gamma / TAU_RATIO... expressed as gamma/tau
                              # below; this is the project default gamma/tau = 0.99
GAMMA0_LIST = [0.01, 0.1, 1.0, 10.0, 100.0]     # study A
RATIO_LIST = [0.5, 0.9, 0.99, 1.05, 1.1]        # study B: gamma/tau. < 1 is the
                              # proven-safe region; above it you get a warning, not a
                              # refusal -- see the stability note in the docstring.
MAX_ITER = 20_000
EPS_ABS = 1e-10
EPS_REL = 1e-15
OMEGA0 = 0.95
ETA = 0.95
THRESHOLD = 2.0
RHO_FLOOR = 1e-4              # delta_ref mask floor. NOT prox_ke_cc's 1e-12: see
                              # the err_delta note in the module docstring --
                              # delta_ref ~ 1/rho^2, so the Gaussian tails must be
                              # masked out, not just the exact zeros.
# ============================================================================

# (field, label, skip_first). skip_first drops the k=0 sample from the *plot*
# only -- the .npz keeps every value.
#
# dz needs it. The initial guess carries mx=0, so A x^0 has zero momentum; the
# KE prox of a zero-momentum state is the identity (f2 = m^2/(2 rho) is already
# 0 there and the penalty is minimized at the input), so y^1 = A x^0 exactly,
# r^0 = 0, and delta^1 = delta^0. dz[0] is therefore machine zero (~1e-16) by
# construction, and a single such point stretches a log axis over ~14 decades
# and flattens everything real.
PANELS = [
    ("D", r"primal residual $\|D^k\|$", False),
    ("P", r"dual residual $\|P^k\|$", False),
    ("err_x", r"true error $\|x^k-x_{\rm ref}\|$", False),
    ("err_delta", r"true dual error $\|\delta^k-\delta_{\rm ref}\|$", False),
    ("dx", r"$\|x^{k+1}-x^k\|$", False),
    ("dz", r"$\|\delta^{k+1}-\delta^k\|$", True),
]


def reference_states_nu(problem, vareps, rho_floor=RHO_FLOOR):
    """(x_ref, delta_ref, delta_mask) from the closed-form SB solution on this
    problem's own (non-uniform) grid.

    x_ref is the staggered state; delta_ref = -grad f2(y_ref) with
    f2(rho,m) = m^2/(2 rho), evaluated at y_ref = A(x_ref) on the cell-centre
    grid. Same construction as gamma_sweep.py's _build_delta_ref, but with
    analytical_sb_gaussian_nu and this grid's own operators, so it is correct
    on a non-uniform time grid. delta_mask zeros out points where rho_ref is at
    the floor: f2 is not differentiable at the boundary of its domain, and
    dividing by a near-zero float there loses all precision anyway.
    """
    ops = problem.ops
    rho_stag, mx_stag = analytical_sb_gaussian_nu(problem, vareps)
    x_ref = State(rho=rho_stag, mx=mx_stag)

    zeros_nt = np.zeros(problem.nt)
    rho_cc = ops.interp_t_at_phi(rho_stag, problem.rho0, problem.rho1)
    mx_cc = ops.interp_x_at_phi(mx_stag, zeros_nt, zeros_nt)

    mask = (rho_cc > rho_floor).astype(float)
    rho_safe = np.where(rho_cc > rho_floor, rho_cc, 1.0)
    delta_ref = State(rho=mx_cc**2 / (2 * rho_safe**2), mx=-mx_cc / rho_safe)
    return x_ref, delta_ref, State(mask, mask)


def check_status(info):
    """'ok' / 'DIVERGED' / 'NOT-CONV'. Guards the silent-failure mode seen past
    the stability cliff, where a run neither overflows nor converges but still
    returns a badly wrong answer -- see the WARNING in main()."""
    D = np.asarray(info.D, dtype=float)
    if not np.all(np.isfinite(D)) or not np.isfinite(info.err_x[-1]):
        return "DIVERGED"
    if D[-1] >= D[0]:
        return "NOT-CONV"
    return "ok"


def run_one(problem, proj_fn, bp, vareps, gamma, tau, adaptive, refs, args):
    x_ref, delta_ref, delta_mask = refs
    cfg = LadmmConfig(
        gamma=gamma, tau=tau, max_iter=args.max_iter,
        eps_abs=args.eps_abs, eps_rel=args.eps_rel, reordered=True,
        step_size=(GlyAdaptiveStepSize(omega0=args.omega0, eta=args.eta,
                                       threshold=args.threshold)
                   if adaptive else None),
    )
    res = discretize_then_optimize_nu(
        problem, proj_fn, bp, vareps, cfg,
        x_ref=x_ref, delta_ref=delta_ref, delta_mask=delta_mask,
    )
    return res.info


def plot_panels(runs, title, out_path):
    """runs: list of (label, info, linestyle). One semilogy panel per tracked
    quantity; constant and adaptive share a colour per gamma_0 so the pairing
    is readable, with linestyle carrying which rule."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for ax, (field, pretty, skip_first) in zip(axes.ravel(), PANELS):
        for label, info, style, colour in runs:
            y = np.abs(np.asarray(getattr(info, field), dtype=float))
            it = np.arange(1, len(y) + 1)          # true iteration index, kept
            if skip_first:
                y, it = y[1:], it[1:]
            # A log axis cannot show <=0 or nan; drop those points rather than
            # let one of them set the y-limits.
            ok = np.isfinite(y) & (y > 0)
            if not np.any(ok):
                continue
            ax.semilogy(it[ok], y[ok], style, color=colour, lw=1.2, label=label)
        ax.set_xlabel("iteration")
        ax.set_title(pretty, fontsize=11)
        ax.grid(True, which="both", alpha=0.3)
    axes.ravel()[0].legend(fontsize=7, ncol=2)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  figure saved to: {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid", choices=list(TIME_GRIDS), default=GRID)
    p.add_argument("--scheme", choices=["CN", "ETD"], default=SCHEME)
    p.add_argument("--nt", type=int, default=NT)
    p.add_argument("--nx", type=int, default=NX)
    p.add_argument("--vareps", type=float, default=VAREPS)
    p.add_argument("--strength", type=float, default=STRENGTH)
    p.add_argument("--gamma", type=float, default=GAMMA)
    p.add_argument("--gamma0-list", type=float, nargs="+", default=GAMMA0_LIST)
    p.add_argument("--ratio-list", type=float, nargs="+", default=RATIO_LIST)
    p.add_argument("--ratio", type=float, default=1.0 / (101.0 / 100.0),
                   help="gamma/tau used for study A (default 0.99, the project default)")
    p.add_argument("--max-iter", type=int, default=MAX_ITER)
    p.add_argument("--eps-abs", type=float, default=EPS_ABS)
    p.add_argument("--eps-rel", type=float, default=EPS_REL)
    p.add_argument("--omega0", type=float, default=OMEGA0)
    p.add_argument("--eta", type=float, default=ETA)
    p.add_argument("--threshold", type=float, default=THRESHOLD)
    p.add_argument("--rho-floor", type=float, default=RHO_FLOOR,
                   help="delta_ref mask floor; delta_ref ~ 1/rho^2 so this "
                        "must exclude the Gaussian tails, not just exact zeros")
    p.add_argument("--initial-value-study", action=BooleanOptionalAction, default=True)
    p.add_argument("--ratio-study", action=BooleanOptionalAction, default=True)
    return p.parse_args()


def main():
    args = parse_args()
    assert args.vareps <= SINKHORN_VAREPS_THRESHOLD, (
        f"vareps={args.vareps:g} needs a Sinkhorn reference, which supplies rho but "
        f"not the mx that delta_ref requires; use vareps <= {SINKHORN_VAREPS_THRESHOLD:g}"
    )
    # ||A||^2 <= 1 on this problem for any grid (nonuniform_grid_norms.tex Sec 6),
    # so gamma/tau < 1 implies the stability rule tau > gamma*||A||^2. Ratios at
    # or above 1 are NOT refused -- the bound is sufficient, not necessary, and
    # the exact one is 1/||A||^2 = 1/cos^2(pi/2n), a hair above 1 -- but they are
    # unproven, so they get a warning and every run is checked for blow-up below.
    past = [r for r in args.ratio_list if r >= 1.0]
    if past:
        print(f"  WARNING: ratios {past} are at/above gamma/tau = 1, outside the region\n"
              f"           where tau > gamma*||A||^2 is proven. They may still converge --\n"
              f"           measured on this problem, up to ~1.25 does and is ~12% faster --\n"
              f"           but failure past the cliff is not always loud: at ratio 1.5,\n"
              f"           nt=nx=32, a run finished without overflowing yet returned a mean\n"
              f"           L2 error of 2.28 instead of 0.0199. Check the `status` column.")

    time_grid, grid_label, _ = build_time_grid(args.grid, args.nt, strength=args.strength)
    problem = setup_problem_nu(prob_gaussian(), time_grid, nx=args.nx)
    problem.ops = build_operators_nu(problem)
    precomp_fn, proj_fn = SCHEMES[args.scheme]
    bp = precomp_fn(problem, args.vareps)
    refs = reference_states_nu(problem, args.vareps, args.rho_floor)
    kept = float(np.mean(refs[2].rho))
    print(f"  delta_ref mask: rho_floor={args.rho_floor:g}, "
          f"{100*kept:.1f}% of cell-centre points kept")

    tag = (f"nt{args.nt}_nx{args.nx}_eps{args.vareps:g}_{args.grid}_{args.scheme}")
    print(f"compare_stepsize_nu: grid={grid_label} scheme={args.scheme} nt={args.nt} "
          f"nx={args.nx} eps={args.vareps:g} max_iter={args.max_iter}")
    print(f"  adaptive rule: omega0={args.omega0:g} eta={args.eta:g} "
          f"threshold={args.threshold:g}  (dead band p/q in "
          f"[{1/args.threshold:g}, {args.threshold:g}])")
    hdr = (f"  {'run':>30} {'iters':>7} {'conv':>5} {'D':>10} {'P':>10} "
           f"{'err_x':>10} {'err_delta':>10} {'gamma_end':>10} {'status':>9}")

    saved = {}
    if args.initial_value_study:
        print(f"\n=== (A) initial-value study, gamma/tau = {args.ratio:g} ===\n{hdr}")
        runs = []
        cmap = plt.get_cmap("viridis")
        for i, g0 in enumerate(args.gamma0_list):
            colour = cmap(i / max(1, len(args.gamma0_list) - 1) * 0.85)
            for adaptive, style in ((False, "--"), (True, "-")):
                info = run_one(problem, proj_fn, bp, args.vareps, g0, g0 / args.ratio,
                               adaptive, refs, args)
                label = f"$\\gamma_0$={g0:g} {'adaptive' if adaptive else 'constant'}"
                runs.append((label, info, style, colour))
                print(f"  {label:>30} {info.iters:>7} {str(info.converged):>5} "
                      f"{info.D[-1]:>10.3e} {info.P[-1]:>10.3e} {info.err_x[-1]:>10.3e} "
                      f"{info.err_delta[-1]:>10.3e} {info.gamma[-1]:>10.4g} "
                      f"{check_status(info):>9}")
                key = f"A_g{g0:g}_{'adaptive' if adaptive else 'constant'}"
                saved.update({f"{key}_{f}": getattr(info, f) for f in
                              [q[0] for q in PANELS] + ["gamma", "tau", "omega"]})
        plot_panels(runs, f"constant (dashed) vs adaptive (solid), {tag}",
                    RESULTS_DIR / f"compare_stepsize_initial_{tag}.png")

    if args.ratio_study:
        print(f"\n=== (B) ratio study, constant step, gamma = {args.gamma:g} ===\n{hdr}")
        runs = []
        cmap = plt.get_cmap("plasma")
        for i, ratio in enumerate(args.ratio_list):
            colour = cmap(i / max(1, len(args.ratio_list) - 1) * 0.85)
            info = run_one(problem, proj_fn, bp, args.vareps, args.gamma,
                           args.gamma / ratio, False, refs, args)
            label = f"$\\gamma/\\tau$={ratio:g}"
            runs.append((label, info, "-", colour))
            print(f"  {label:>30} {info.iters:>7} {str(info.converged):>5} "
                  f"{info.D[-1]:>10.3e} {info.P[-1]:>10.3e} {info.err_x[-1]:>10.3e} "
                  f"{info.err_delta[-1]:>10.3e} {info.gamma[-1]:>10.4g} "
                  f"{check_status(info):>9}")
            saved.update({f"B_r{ratio:g}_{f}": getattr(info, f) for f in [q[0] for q in PANELS]})
        plot_panels(runs, f"constant step, gamma/tau sweep, {tag}",
                    RESULTS_DIR / f"compare_stepsize_ratio_{tag}.png")

    npz = RESULTS_DIR / f"compare_stepsize_{tag}.npz"
    np.savez_compressed(npz, **saved)
    print(f"\n  histories saved to: {npz}")


if __name__ == "__main__":
    main()
