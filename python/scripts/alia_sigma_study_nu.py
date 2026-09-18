"""ALiA sigma study on the Gaussian Schrodinger bridge: sweep sigma across
grids x vareps x FP-projection schemes, against a LADMM baseline.

ALiA (Jang, Sun, Yin & Ryu, arXiv:2602.15000; see optimal_transport/alia.py)
takes three inputs, and they are not equally interesting:

  gamma0  the initial stepsize. Self-correcting -- this is the paper's whole
          point. Verified here rather than assumed: --gamma0-check reruns the
          best sigma from several gamma0 and reports whether the gamma
          trajectories agree.
  sigma   the primal-dual stepsize ratio. NOT self-correcting, and the real
          knob: on one configuration the mean L2 error spanned three orders of
          magnitude across sigma while everything else was held fixed. Hence
          this script.
  eps     a small constant needed for point convergence, constrained by
          0 < eps < min(1/2, 1/(4 sigma)). It enters the stepsize bound only
          through an 8*sigma*eps term, so it is a theoretical device rather
          than a performance knob -- set from --eps-frac as a fixed fraction
          of its own upper limit, which keeps it valid as sigma varies.

The binding constraint in Subroutine 1 for this problem is always the a/b
bound, gamma <~ 1/(4 sqrt(sigma (a^2+b^2))): with f2 = g2 = 0 the curvature
terms vanish identically and Gamma_x/Gamma_y are never the minimum (the
`minsel` column reports the tally, so this is checked every run rather than
assumed). That is why sigma matters so much -- it sets the whole stepsize
scale through that bound.

Error metric matches the other drivers: per-time-row L2 in rho at the
cell-centres against reference_rho_at (analytical below
study_utils.SINKHORN_VAREPS_THRESHOLD, Sinkhorn-Hopf-Cole above), reduced by
mean and max over rows. A LADMM run at the project's default gamma/tau is
included in every block as the baseline -- note it is a *different algorithm*,
not another step-size rule, so read it as a reference point, not a control.

Usage:
    python3 scripts/alia_sigma_study_nu.py
    python3 scripts/alia_sigma_study_nu.py --grids uniform clustered --schemes ETD
    python3 scripts/alia_sigma_study_nu.py --eps-list 1e-4 1 --sigma-list 1e-3 1e-2 1e-1
    python3 scripts/alia_sigma_study_nu.py --gamma0-check
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
    RESULTS_DIR,
    SINKHORN_MAX_ITER,
    SINKHORN_NX_FINE,
    SINKHORN_TOL,
    SINKHORN_VAREPS_THRESHOLD,
    TIME_GRIDS,
    build_time_grid,
)

# ============================================================================
# Edit these directly; all overridable from the command line.
# ============================================================================
GRIDS = ["uniform", "clustered"]        # any of study_utils.TIME_GRIDS
SCHEME_LIST = ["CN", "ETD"]
VAREPS_LIST = [1e-4, 1e-2, 1.0]   # the SB diffusion parameter
SIGMA_LIST = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1.0, 10.0]
NT = 32
NX = 32
STRENGTH = 1.0
N_BOUNDARY = 5
GAMMA0 = 1.0                  # self-correcting; --gamma0-check verifies that
ALIA_EPS_FRAC = 0.1           # alia_eps = ALIA_EPS_FRAC * min(1/2, 1/(4 sigma)).
                              # ALiA's OWN epsilon -- distinct from VAREPS_LIST
                              # above and from EPS_ABS/EPS_REL below; see the
                              # THREE DIFFERENT EPSILONS note in alia.py.
MAX_ITER = 8_000
EPS_ABS = 1e-10
EPS_REL = 1e-15
LADMM_GAMMA = 1.0             # baseline
LADMM_RATIO = 100.0 / 101.0   # baseline gamma/tau (the project default 0.99)
GAMMA0_CHECK_LIST = [0.01, 1.0, 100.0]
# ============================================================================


def alia_eps_for(sigma, frac):
    return frac * min(0.5, 1.0 / (4.0 * sigma))


def run_block(problem, proj_fn, bp, vareps, sink_fit, args):
    """One (grid, scheme, vareps) block: LADMM baseline + a sigma sweep."""
    nx = problem.nx
    ref = reference_rho_at(problem.t_centers, nx, vareps, sink_fit)
    def errs(res):
        e = np.sqrt(problem.dx * np.sum((res.rho_cc - ref) ** 2, axis=1))
        return float(e.mean()), float(e.max())

    rows = []
    cfg = LadmmConfig(gamma=args.ladmm_gamma, tau=args.ladmm_gamma / args.ladmm_ratio,
                      max_iter=args.max_iter, eps_abs=args.eps_abs,
                      eps_rel=args.eps_rel, reordered=True)
    r = discretize_then_optimize_nu(problem, proj_fn, bp, vareps, cfg)
    m, mx = errs(r)
    rows.append(dict(method="LADMM", sigma=np.nan, iters=r.info.iters,
                     D=float(r.info.D[-1]), P=float(r.info.P[-1]), mean=m, max=mx,
                     gamma_end=float(r.info.gamma[-1]), a_end=np.nan,
                     minsel="", wall=r.info.walltime, status="ok"))

    for sigma in args.sigma_list:
        e = alia_eps_for(sigma, args.alia_eps_frac)
        try:
            acfg = AliaConfig(gamma0=args.gamma0, sigma=sigma, alia_eps=e,
                              max_iter=args.max_iter, eps_abs=args.eps_abs,
                              eps_rel=args.eps_rel)
            ra = discretize_then_optimize_nu_alia(problem, proj_fn, bp, vareps, acfg)
            i = ra.info
            m, mx = errs(ra)
            sel = np.bincount(i.which.astype(int), minlength=4)
            bad = (not np.all(np.isfinite(i.D))) or not np.isfinite(m)
            rows.append(dict(method="ALiA", sigma=sigma, iters=i.iters,
                             D=float(i.D[-1]), P=float(i.P[-1]), mean=m, max=mx,
                             gamma_end=float(i.gamma[-1]), a_end=float(i.a[-1]),
                             minsel="/".join(str(v) for v in sel),
                             wall=i.walltime,
                             status="DIVERGED" if bad else "ok"))
        except Exception as exc:  # a stepsize that goes non-finite raises
            rows.append(dict(method="ALiA", sigma=sigma, iters=0, D=np.nan, P=np.nan,
                             mean=np.nan, max=np.nan, gamma_end=np.nan, a_end=np.nan,
                             minsel="", wall=0.0,
                             status=f"{type(exc).__name__}"))
    return rows


def print_rows(rows):
    print(f"    {'method':>6} {'sigma':>8} {'iters':>6} {'D':>10} {'P':>10} "
          f"{'meanL2':>10} {'maxL2':>10} {'gamma_end':>10} {'a_end':>7} "
          f"{'minsel':>13} {'status':>9}")
    for r in rows:
        sg = "  --" if np.isnan(r["sigma"]) else f"{r['sigma']:.3g}"
        ae = "   --" if np.isnan(r["a_end"]) else f"{r['a_end']:.4f}"
        print(f"    {r['method']:>6} {sg:>8} {r['iters']:>6} {r['D']:>10.3e} "
              f"{r['P']:>10.3e} {r['mean']:>10.4e} {r['max']:>10.4e} "
              f"{r['gamma_end']:>10.4g} {ae:>7} {r['minsel']:>13} {r['status']:>9}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grids", nargs="+", choices=list(TIME_GRIDS), default=GRIDS)
    p.add_argument("--schemes", nargs="+", choices=list(SCHEMES), default=SCHEME_LIST)
    p.add_argument("--vareps-list", type=float, nargs="+", default=VAREPS_LIST,
                   help="SB diffusion parameter (NOT ALiA's epsilon)")
    p.add_argument("--sigma-list", type=float, nargs="+", default=SIGMA_LIST)
    p.add_argument("--nt", type=int, default=NT)
    p.add_argument("--nx", type=int, default=NX)
    p.add_argument("--strength", type=float, default=STRENGTH)
    p.add_argument("--n-boundary", type=int, default=N_BOUNDARY)
    p.add_argument("--gamma0", type=float, default=GAMMA0)
    p.add_argument("--alia-eps-frac", type=float, default=ALIA_EPS_FRAC,
                   help="alia_eps as a fraction of its own upper limit "
                        "min(1/2, 1/(4 sigma)); NOT the SB vareps")
    p.add_argument("--max-iter", type=int, default=MAX_ITER)
    p.add_argument("--eps-abs", type=float, default=EPS_ABS)
    p.add_argument("--eps-rel", type=float, default=EPS_REL)
    p.add_argument("--ladmm-gamma", type=float, default=LADMM_GAMMA)
    p.add_argument("--ladmm-ratio", type=float, default=LADMM_RATIO)
    p.add_argument("--gamma0-check", action=argparse.BooleanOptionalAction, default=False,
                   help="rerun the best sigma from several gamma0 to confirm self-correction")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"alia_sigma_study_nu: nt={args.nt} nx={args.nx} max_iter={args.max_iter}\n"
          f"  grids={args.grids}  schemes={args.schemes}  vareps_list={args.vareps_list}\n"
          f"  sigma_list={args.sigma_list}  gamma0={args.gamma0:g} "
          f"(alia_eps = {args.alia_eps_frac:g} * min(1/2, 1/(4 sigma)))\n"
          f"  baseline: reordered LADMM, gamma={args.ladmm_gamma:g}, "
          f"gamma/tau={args.ladmm_ratio:g}  [a different algorithm, not a control]\n"
          f"  minsel = how often each of (1.5*gamma_k, a/b bound, Gamma_x, Gamma_y)\n"
          f"           attained the min in Subroutine 1\n")

    # One Sinkhorn fit per large vareps, reused across every grid and scheme.
    fits = {}
    for ve in args.vareps_list:
        if ve > SINKHORN_VAREPS_THRESHOLD:
            assert SINKHORN_NX_FINE % args.nx == 0, (
                f"SINKHORN_NX_FINE={SINKHORN_NX_FINE} must be a multiple of nx={args.nx}")
            fp = setup_problem_nu(prob_gaussian(), uniform_time_grid(2), nx=SINKHORN_NX_FINE)
            fits[ve] = sinkhorn_fit(fp, ve, SINKHORN_MAX_ITER, SINKHORN_TOL)
            print(f"  Sinkhorn reference for eps={ve:g}: iters={fits[ve].iters} "
                  f"converged={fits[ve].converged} error={fits[ve].error:.2e}")

    all_rows, best = [], {}
    for grid in args.grids:
        tg, label, _ = build_time_grid(grid, args.nt, strength=args.strength,
                                       n_boundary=args.n_boundary)
        problem = setup_problem_nu(prob_gaussian(), tg, nx=args.nx)
        problem.ops = build_operators_nu(problem)
        for scheme in args.schemes:
            precomp_fn, proj_fn = SCHEMES[scheme]
            for ve in args.vareps_list:
                bp = precomp_fn(problem, ve)
                print(f"\n=== grid={label}  scheme={scheme}  vareps={ve:g}  "
                      f"(cells={problem.nt}) ===")
                rows = run_block(problem, proj_fn, bp, ve, fits.get(ve), args)
                print_rows(rows)
                ok = [r for r in rows if r["method"] == "ALiA" and r["status"] == "ok"
                      and np.isfinite(r["mean"])]
                if ok:
                    b = min(ok, key=lambda r: r["mean"])
                    lad = rows[0]["mean"]
                    best[(grid, scheme, ve)] = b["sigma"]
                    print(f"    -> best sigma={b['sigma']:g} (meanL2={b['mean']:.4e}) vs "
                          f"LADMM {lad:.4e}  ->  {b['mean']/lad:.2f}x "
                          f"({'ALiA better' if b['mean'] < lad else 'LADMM better'}, "
                          f"at this iteration budget)")
                for r in rows:
                    all_rows.append(dict(grid=grid, scheme=scheme, eps=ve, **r))

    if args.gamma0_check and best:
        print("\n=== gamma0 self-correction check (best sigma per block) ===")
        for (grid, scheme, ve), sg in best.items():
            tg, label, _ = build_time_grid(grid, args.nt, strength=args.strength,
                                           n_boundary=args.n_boundary)
            problem = setup_problem_nu(prob_gaussian(), tg, nx=args.nx)
            problem.ops = build_operators_nu(problem)
            precomp_fn, proj_fn = SCHEMES[scheme]
            bp = precomp_fn(problem, ve)
            ends = []
            for g0 in GAMMA0_CHECK_LIST:
                acfg = AliaConfig(gamma0=g0, sigma=sg, alia_eps=alia_eps_for(sg, args.alia_eps_frac),
                                  max_iter=min(args.max_iter, 2000),
                                  eps_abs=args.eps_abs, eps_rel=args.eps_rel)
                ra = discretize_then_optimize_nu_alia(problem, proj_fn, bp, ve, acfg)
                ends.append(float(ra.info.gamma[-1]))
            spread = (max(ends) - min(ends)) / max(abs(min(ends)), 1e-300)
            print(f"  {label:>22} {scheme} eps={ve:<7g} sigma={sg:<7g} "
                  f"gamma_end from gamma0={GAMMA0_CHECK_LIST}: "
                  f"{[f'{v:.4g}' for v in ends]}  spread {spread:.1e} "
                  f"(~{max(0, int(-np.log10(max(spread, 1e-300)))):d} significant figures)"
                  f"{'  <- self-corrected' if spread < 1e-2 else '  <- DID NOT agree'}")
        # Threshold 1e-2, not machine precision. gamma0 far from the settled
        # value costs a transient (the 3/2 growth cap needs ~log_1.5 of the gap
        # to climb), so the trajectories agree to a few significant figures and
        # tighten with iteration count -- measured 1.2e-3 at 1000 iterations
        # and 1.9e-4 at 1500, from gamma0 spanning four orders of magnitude.

    npz = RESULTS_DIR / f"alia_sigma_study_nt{args.nt}_nx{args.nx}.npz"
    keys = all_rows[0].keys()
    np.savez_compressed(npz, **{k: np.array([r[k] for r in all_rows]) for k in keys})
    print(f"\n  table saved to: {npz}")

    # meanL2 vs sigma, one panel per (grid, scheme), one curve per eps.
    combos = [(g, s) for g in args.grids for s in args.schemes]
    fig, axes = plt.subplots(1, max(1, len(combos)), figsize=(6 * max(1, len(combos)), 4.6),
                             squeeze=False)
    for ax, (g, sc) in zip(axes[0], combos):
        for ve in args.vareps_list:
            sel = [r for r in all_rows if r["grid"] == g and r["scheme"] == sc
                   and r["eps"] == ve and r["method"] == "ALiA" and np.isfinite(r["mean"])]
            if sel:
                ax.loglog([r["sigma"] for r in sel], [r["mean"] for r in sel],
                          "o-", lw=1.3, label=f"vareps={ve:g}")
            lad = [r for r in all_rows if r["grid"] == g and r["scheme"] == sc
                   and r["eps"] == ve and r["method"] == "LADMM"]
            if lad:
                ax.axhline(lad[0]["mean"], ls="--", lw=0.9, alpha=0.6)
        ax.set_xlabel(r"$\sigma$"); ax.set_ylabel("mean L2 error in rho")
        ax.set_title(f"{g}, {sc}  (dashed = LADMM baseline)", fontsize=10)
        ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout()
    out = RESULTS_DIR / f"alia_sigma_study_nt{args.nt}_nx{args.nx}.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  figure saved to: {out}")


if __name__ == "__main__":
    main()
