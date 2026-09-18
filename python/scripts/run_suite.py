"""Batch runner: every (case, vareps, resolution) solve, saved for later plotting.

The solves are expensive and the plotting is not, so this script does the solves
ONCE and writes everything a plot could need to disk. Nothing here plots. One
.npz per run, so a crashed or interrupted sweep resumes by simply re-running
(finished runs are skipped unless --force).

  cases        gaussian  N(1/3, .05^2) -> N(2/3, .05^2)        (analytic SB exists)
               bimodal   1/2 N(.25,.05^2)+1/2 N(.75,.05^2) -> N(.5,.05^2)
  vareps       0, 1e-8, 1e-4, 1e-3, 1e-2, 1e-1, 1
  time grid    uniform for vareps < 1e-2, clustered (strength 1) for vareps >= 1e-2
  resolutions  nt = nx = 32, 64, 128, 256
  scheme       ETD  (projection_expsemi / projection_expsemi_nu)
  solver       ladmm, gamma=1, tau=gamma*101/100, 40k iterations

Reference solutions saved alongside each run, so error metrics are computed at
PLOT time and can be changed without re-solving:

  vareps <= SINKHORN_VAREPS_THRESHOLD (5e-3)  ->  "analytic"
      Closed-form SB. Exists for the gaussian case ONLY. The bimodal case has NO
      reference at these vareps -- see LIMITATIONS below.
  vareps >  SINKHORN_VAREPS_THRESHOLD         ->  "sinkhorn_same" AND "sinkhorn_fine"
      Both are saved every time. "same" fits on the run's own nx (spatially
      consistent with the solver, isolates the TIME error); "fine" fits at
      SINKHORN_NX_FINE and is brought down with the spectral interpolant
      (approximates the continuum, measures TOTAL discretization error). They
      differ by the reference's own O(dx^2); see docs/sinkhorn_reference.tex.

Note the vareps split lines up with the reference switch by construction: the
four smallest vareps are all below the Sinkhorn threshold and use the uniform
grid; the three largest are above it and use the clustered grid.

WHAT IS SAVED (per run, see save_run for exact key names)
  fields      rho_stag, mx_stag, rho_cc, mx_cc   -- full density/momentum fields,
              enough for density profiles at any time
  grids       t_edges, t_centers, dt_vec, xx, dx, nt, nx
  references  ref_<name>_rho_stag / ref_<name>_mx_stag for each available ref
  histories   D, P (primal/dual residuals), dx_hist, dy_hist, dz_hist,
              scale_D, scale_P                     -- per-iteration
              err_full_<ref>, err_rho_<ref>        -- per-iteration error against
              EVERY saved reference, both as a full-state norm and in rho alone.
              All length = iters.
  metadata    case, vareps, grid kind, scheme, gamma, tau, iters, converged,
              walltime, ref_names

LIMITATIONS to be aware of before reading any plot from this data
  1. The bimodal case has no reference for vareps <= 5e-3. Those runs still save
     fields and residual histories; the intended substitute is a grid-refinement
     comparison against the nt=nx=256 run, which costs nothing extra at plot time
     since all fields are saved.
  2. 40k iterations does not mean converged. Check the `converged` flag and the
     residual histories per run rather than assuming.

Usage:
    python scripts/run_suite.py --dry-run     # list the runs + cost estimate
    python scripts/run_suite.py               # run everything not yet on disk
    python scripts/run_suite.py --case bimodal --eps 1.0
    python scripts/run_suite.py --force       # re-run even if the .npz exists
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from optimal_transport.grid import setup_problem
from optimal_transport.grid_nu import setup_problem_nu
from optimal_transport.operators import build_operators
from optimal_transport.operators_nu import build_operators_nu
from optimal_transport.pipeline import LadmmConfig, discretize_then_optimize
from optimal_transport.pipeline_nu import (
    LadmmConfig as LadmmConfigNU,
    discretize_then_optimize_nu,
)
from optimal_transport.problems import (
    analytical_sb_gaussian,
    prob_bimodal,
    prob_gaussian,
)
from optimal_transport.problems_nu import analytical_sb_gaussian_nu
from optimal_transport.projection_expsemi import (
    precomp_expsemi_proj,
    proj_fokker_planck_expsemi,
)
from optimal_transport.projection_expsemi_nu import (
    precomp_expsemi_proj_nu,
    proj_fokker_planck_expsemi_nu,
)
from optimal_transport.sinkhorn import sinkhorn_eval, sinkhorn_fit
from optimal_transport.state import State
from study_utils import (
    SINKHORN_MAX_ITER,
    SINKHORN_NX_FINE,
    SINKHORN_TOL,
    SINKHORN_VAREPS_THRESHOLD,
    _coarsen_mx,
    build_time_grid,
    resample_rho_spectral,
)

# ===========================================================================
# CONFIG -- edit here, then `python scripts/run_suite.py`
# ===========================================================================
CASES = {
    "gaussian": dict(prob=prob_gaussian, analytic=True),
    "bimodal": dict(prob=prob_bimodal, analytic=False),
}
EPS_LIST = [0.0, 1e-8, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
RESOLUTIONS = [32, 64, 128, 256]

GAMMA = 1.0
TAU_RATIO = 101.0 / 100.0
MAX_ITER = 40_000
EPS_ABS = 1e-10
EPS_REL = 1e-15

GRID_SWITCH_EPS = 1e-2      # vareps >= this -> clustered; below -> uniform
CLUSTER_STRENGTH = 1.0      # clustered_time_grid strength (1 = full clustering)

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "suite"
# ===========================================================================


def grid_kind(vareps):
    return "clustered" if vareps >= GRID_SWITCH_EPS else "uniform"


def run_tag(case, vareps, n):
    return f"{case}_eps{vareps:g}_n{n}_{grid_kind(vareps)}_ETD"


# ---------------------------------------------------------------- references
def _sinkhorn_ref(fit, t_edge, t_cc, nx):
    """(rho_stag, mx_stag) from a SinkhornFit, reduced to nx.

    The two live on DIFFERENT time grids and need separate evaluations: rho_stag
    sits at the interior EDGE times (nt-1 rows), mx_stag at the CELL-CENTRE times
    (nt rows). Evaluating both at one time array silently produces an mx with the
    wrong number of rows.

    In space, rho uses the spectral interpolant and mx exact subsampling
    (staggered points are at j*dx, and node grids ARE nested). The factor is
    nx_fine//nx, NOT (nx_fine-1)//(nx-1): _coarsen_mx derives its own output
    length from the factor, so it wants the cell-count ratio (4096//32 = 128, not
    4095//31 = 132). Both reduce to the identity when the fit is already at nx.
    """
    factor = fit.phi_0.shape[0] // nx
    rho_f, _ = sinkhorn_eval(fit, t_edge)
    _, mx_f = sinkhorn_eval(fit, t_cc)
    return resample_rho_spectral(rho_f, nx), _coarsen_mx(mx_f, factor)


def build_references(problem, vareps, case, kind, verbose=True):
    """{name: (rho_stag, mx_stag)} for every reference available at this vareps."""
    refs = {}
    nx, nt = problem.nx, problem.nt
    if kind == "clustered":
        t_edge, t_cc = problem.t_edges[1:-1], problem.t_centers
    else:
        t_edge = np.arange(1, nt) * problem.dt
        t_cc = (np.arange(1, nt + 1) - 0.5) * problem.dt

    if vareps <= SINKHORN_VAREPS_THRESHOLD:
        if CASES[case]["analytic"]:
            refs["analytic"] = (analytical_sb_gaussian_nu(problem, vareps) if kind == "clustered"
                                else analytical_sb_gaussian(problem, vareps))
        elif verbose:
            print(f"    (no reference: {case} has no closed form and "
                  f"vareps={vareps:g} <= {SINKHORN_VAREPS_THRESHOLD:g})")
        return refs

    prob_def = CASES[case]["prob"]()
    f_same = sinkhorn_fit(problem, vareps, SINKHORN_MAX_ITER, SINKHORN_TOL)
    refs["sinkhorn_same"] = _sinkhorn_ref(f_same, t_edge, t_cc, nx)

    fine = (setup_problem_nu(prob_def, build_time_grid("uniform", 2)[0], nx=SINKHORN_NX_FINE)
            if kind == "clustered" else
            setup_problem(prob_def, nt=SINKHORN_NX_FINE, nx=SINKHORN_NX_FINE))
    f_fine = sinkhorn_fit(fine, vareps, SINKHORN_MAX_ITER, SINKHORN_TOL)
    refs["sinkhorn_fine"] = _sinkhorn_ref(f_fine, t_edge, t_cc, nx)

    if verbose:
        print(f"    sinkhorn same: iters={f_same.iters} t0err={f_same.error:.1e} "
              f"t1err={f_same.error_right:.1e} | "
              f"fine(nx={SINKHORN_NX_FINE}): iters={f_fine.iters} "
              f"t0err={f_fine.error:.1e} t1err={f_fine.error_right:.1e}")
    return refs


# ---------------------------------------------------------------- one run
def run_one(case, vareps, n, out_path):
    kind = grid_kind(vareps)
    prob_def = CASES[case]["prob"]()
    tau = GAMMA * TAU_RATIO

    if kind == "clustered":
        tg, _, _ = build_time_grid("clustered", n, strength=CLUSTER_STRENGTH)
        problem = setup_problem_nu(prob_def, tg, nx=n)
        problem.ops = build_operators_nu(problem)
        precomp, proj = precomp_expsemi_proj_nu, proj_fokker_planck_expsemi_nu
        cfg = LadmmConfigNU(gamma=GAMMA, tau=tau, max_iter=MAX_ITER,
                            eps_abs=EPS_ABS, eps_rel=EPS_REL)
        solve = discretize_then_optimize_nu
        t_edges, t_centers, dt_vec = problem.t_edges, problem.t_centers, problem.dt_vec
    else:
        problem = setup_problem(prob_def, nt=n, nx=n)
        problem.ops = build_operators(problem)
        precomp, proj = precomp_expsemi_proj, proj_fokker_planck_expsemi
        cfg = LadmmConfig(gamma=GAMMA, tau=tau, max_iter=MAX_ITER,
                          eps_abs=EPS_ABS, eps_rel=EPS_REL)
        solve = discretize_then_optimize
        t_edges = np.arange(problem.nt + 1) * problem.dt
        t_centers = (np.arange(1, problem.nt + 1) - 0.5) * problem.dt
        dt_vec = np.full(problem.nt, problem.dt)

    refs = build_references(problem, vareps, case, kind)

    # EVERY reference is scored in the SAME solve: ladmm tracks one history per
    # entry of x_refs, so sinkhorn_same and sinkhorn_fine both get a full
    # per-iteration error from one run rather than one run each.
    x_refs = {nm: State(r, m) for nm, (r, m) in refs.items()}

    t0 = time.perf_counter()
    res = solve(problem, proj, precomp(problem, vareps), vareps, cfg, x_refs=x_refs)
    wall = time.perf_counter() - t0
    info = res.info
    tail = "  ".join(f"{nm}: full={info.err_x_refs[nm][-1]:.3e} "
                     f"rho={info.err_rho_refs[nm][-1]:.3e}" for nm in sorted(refs))
    print(f"    iters={info.iters} converged={info.converged} wall={wall:.1f}s  "
          f"D={info.D[-1]:.2e} P={info.P[-1]:.2e}", flush=True)
    print(f"    final error -- {tail if tail else '(no reference)'}", flush=True)

    payload = dict(
        rho_stag=res.rho_stag, mx_stag=res.mx_stag,
        rho_cc=res.rho_cc, mx_cc=res.mx_cc,
        t_edges=t_edges, t_centers=t_centers, dt_vec=dt_vec,
        xx=problem.xx, dx=problem.dx, nt=problem.nt, nx=problem.nx,
        D=info.D, P=info.P, dx_hist=info.dx, dy_hist=info.dy, dz_hist=info.dz,
        scale_D=info.scale_D, scale_P=info.scale_P,
        iters=info.iters, converged=info.converged, walltime=wall,
        **{f"err_full_{nm}": info.err_x_refs[nm] for nm in refs},
        **{f"err_rho_{nm}": info.err_rho_refs[nm] for nm in refs},
        case=case, vareps=vareps, grid=kind, scheme="ETD", solver="ladmm",
        gamma=GAMMA, tau=tau, max_iter=MAX_ITER,
        cluster_strength=(CLUSTER_STRENGTH if kind == "clustered" else np.nan),
        ref_names=np.array(sorted(refs), dtype=object),
    )
    for nm, (r, m) in refs.items():
        payload[f"ref_{nm}_rho_stag"] = r
        payload[f"ref_{nm}_mx_stag"] = m

    # Write to a temp file and rename. np.savez_compressed on a 40k-iteration
    # payload is not instantaneous, and a crash partway through would otherwise
    # leave a truncated .npz that LOOKS finished -- and would then be silently
    # SKIPPED on resume. os.replace is atomic on POSIX, so the final path only
    # ever exists complete.
    # (savez_compressed appends ".npz" to a PATH whose name lacks it, which would
    # defeat the rename -- so hand it an open file object instead.)
    tmp = out_path.with_name(out_path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **payload)
    os.replace(tmp, out_path)
    size_mb = out_path.stat().st_size / 1e6
    print(f"    saved  {out_path.name}  ({size_mb:.1f} MB)", flush=True)

    append_manifest(dict(
        case=case, vareps=vareps, n=n, grid=kind, nt=problem.nt, nx=problem.nx,
        iters=info.iters, converged=int(info.converged), walltime_s=round(wall, 1),
        D_final=float(info.D[-1]), P_final=float(info.P[-1]),
        refs=";".join(sorted(refs)) or "none",
        **{f"err_rho_{nm}": float(info.err_rho_refs[nm][-1]) for nm in refs},
        **{f"err_full_{nm}": float(info.err_x_refs[nm][-1]) for nm in refs},
        file=out_path.name,
    ))
    return wall


MANIFEST = "manifest.csv"
MANIFEST_COLS = [
    "case", "vareps", "n", "grid", "nt", "nx", "iters", "converged",
    "walltime_s", "D_final", "P_final", "refs",
    "err_rho_analytic", "err_rho_sinkhorn_same", "err_rho_sinkhorn_fine",
    "err_full_analytic", "err_full_sinkhorn_same", "err_full_sinkhorn_fine",
    "file",
]


def append_manifest(row):
    """Append one summary line per finished run, immediately after it is saved.

    Lets the sweep be inspected -- or reported on -- without loading a single
    .npz, and survives a crash for the same reason the per-run files do. The
    column set is FIXED (MANIFEST_COLS) rather than derived per row, so runs with
    different references still line up in one table; inapplicable cells are left
    empty.
    """
    path = OUT_DIR / MANIFEST
    write_header = not path.exists() or path.stat().st_size == 0

    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.8e}"
        return str(v)

    with open(path, "a") as fh:
        if write_header:
            fh.write(",".join(MANIFEST_COLS) + "\n")
        fh.write(",".join(fmt(row.get(c)) for c in MANIFEST_COLS) + "\n")


# ---------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="list runs and estimated cost")
    ap.add_argument("--force", action="store_true", help="re-run even if the .npz exists")
    ap.add_argument("--case", choices=sorted(CASES), action="append")
    ap.add_argument("--eps", type=float, action="append")
    ap.add_argument("--n", type=int, action="append")
    a = ap.parse_args()

    cases = a.case or sorted(CASES)
    epss = a.eps or EPS_LIST
    ns = a.n or RESOLUTIONS
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    jobs = [(c, e, n) for c in cases for e in epss for n in ns]
    todo = [(c, e, n) for c, e, n in jobs
            if a.force or not (OUT_DIR / f"{run_tag(c, e, n)}.npz").exists()]

    # 13.8 s for 40k iterations at 32x32, measured on this machine; cost is
    # linear in iterations and in nt*nx, so scale from there. Rough, not a
    # promise -- and it ignores the Sinkhorn fits, which are seconds at most.
    est = sum(13.8 * (MAX_ITER / 40_000) * (n * n) / (32 * 32) for _, _, n in todo)
    print(f"{len(jobs)} runs requested, {len(todo)} not yet on disk.")
    print(f"rough estimate: {est/3600:.1f} h total "
          f"({est/max(len(todo),1):.0f} s/run average)")
    print(f"output: {OUT_DIR}")
    if a.dry_run:
        for c, e, n in todo:
            print(f"  {run_tag(c, e, n)}"
                  + ("" if e > SINKHORN_VAREPS_THRESHOLD or CASES[c]["analytic"]
                     else "     [NO REFERENCE]"))
        return

    t_sweep = time.perf_counter()
    done_s = 0.0
    for i, (c, e, n) in enumerate(todo, 1):
        tag = run_tag(c, e, n)
        ref_note = ("" if e > SINKHORN_VAREPS_THRESHOLD or CASES[c]["analytic"]
                    else "   [NO REFERENCE]")
        print(f"\n{'-' * 72}", flush=True)
        print(f"[{i}/{len(todo)}]  case={c}   vareps={e:g}   nt=nx={n}   "
              f"grid={grid_kind(e)}{ref_note}", flush=True)
        print(f"{'-' * 72}", flush=True)
        done_s += run_one(c, e, n, OUT_DIR / f"{tag}.npz")
        rate = done_s / i
        print(f"    elapsed {(time.perf_counter()-t_sweep)/60:.1f} min   "
              f"remaining ~{rate*(len(todo)-i)/60:.0f} min", flush=True)
    print(f"\ndone -- {len(todo)} runs written to {OUT_DIR}")
    print(f"summary table: {OUT_DIR / MANIFEST}")


if __name__ == "__main__":
    main()
