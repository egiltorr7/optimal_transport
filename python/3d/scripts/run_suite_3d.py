"""Batch runner for the 3D Gaussian grid-convergence study -- the 3D sibling
of 2d/scripts/run_suite_2d.py, reduced to what the uniform-grid 3D solver
supports today (no clustered time grid, no Sinkhorn reference).

  case         gaussian  N((.4,.4,.4), .08^2 I) -> N((.6,.6,.6), .08^2 I)
               (prob_gaussian_3d defaults: boundary-free, wall density ~1e-4 of peak)
  vareps       default 1e-3 (closed-form reference valid for vareps <= 5e-3)
  time grid    uniform
  resolutions  nt = nx = ny = nz = 16, 32, 48, 64, 96 (see the memory table
               in sb_gaussian_3d.py: 96^4 is ~37-54 GB in fp64; 112^4 needs
               --precision mixedc on a 120 GB GH200)
  scheme       ETD, DCT-in-t mode solver, matmul DCT
  solver       ladmm, jit, gamma=0.05 by default (the density peak scales as
               1/(2 pi sigma^2)^(3/2) ~ 124 here vs 64 in 2D vs 8 in 1D, and
               gamma should scale as 1/peak -- memory: ladmm-gamma-density-scale),
               tau=gamma*101/100, 40k iterations

Reference: "analytic", the closed-form free-space SB (problems_3d).

WHAT IS SAVED (per run) -- same layout as run_suite_2d.py so the post script
is a straight port:
  rows_analytic_rho (ntm,), rows_analytic_mx/my/mz (nt,): per-time-row
  spatial L2^2 of the final iterate; err_full_analytic / err_rho_analytic
  (iters,) per-iteration errors; rho_diag (ntm, nx) on the space diagonal
  x=y=z; snap_idx / snap_rho (k, nx, ny) z-INTEGRATED planes at k times (a
  full (ntm,nx,ny,nz) field is 680 MB at 96^4); mass_t, min_rho_t; grids;
  D/P histories; metadata incl. ms_per_iter and peak_gpu_mem_mb.

Usage:
    python scripts/run_suite_3d.py --dry-run
    python scripts/run_suite_3d.py --device gpu
    python scripts/run_suite_3d.py --device gpu --n 16 32 --gamma 0.1 --out-dir results/gamma_probe
    python scripts/run_suite_3d.py --device gpu --n 112 --precision mixedc
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))          # 3d/ -> setup

# ===========================================================================
# CONFIG -- edit here, or override from the command line
# ===========================================================================
EPS_LIST = [1e-3]
RESOLUTIONS = [16, 32, 48, 64, 96]
MU0, MU1, SIGMA = (0.4, 0.4, 0.4), (0.6, 0.6, 0.6), 0.08

GAMMA = 0.05
TAU_RATIO = 101.0 / 100.0
MAX_ITER = 40_000
EPS_ABS = 1e-10
EPS_REL = 1e-15
CHECK_EVERY = 100
DIAG_LEVEL = "full"
N_SNAPSHOTS = 9
DCT_METHOD = "matmul"
ANALYTIC_MAX_EPS = 5e-3      # closed form is a valid reference only below this

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "suite"
LONG_POINTS = {}
# ===========================================================================

PRECISIONS = ("fp64", "fp32", "mixed", "mixedm", "mixedc", "fp64rx", "fp64px")


def _pre_parse():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-preallocate", action="store_true")
    p.add_argument("--mem-fraction", type=float, default=None)
    p.add_argument("--precision", choices=PRECISIONS, default="fp64")
    return p.parse_known_args()[0]


_PRE = _pre_parse()
PRECISION = _PRE.precision
from setup import jax_config  # noqa: E402

jax_config.configure(device=_PRE.device, x64=True,
                     preallocate=False if _PRE.no_preallocate else None,
                     mem_fraction=_PRE.mem_fraction)

import jax  # noqa: E402

if PRECISION != "fp64":
    jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from setup import projection_3d as P3  # noqa: E402
from setup.grid_3d import setup_problem_3d  # noqa: E402
from setup.operators_3d import build_operators_3d  # noqa: E402
from setup.pipeline_3d import LadmmConfig, discretize_then_optimize_3d  # noqa: E402
from setup.precision import configure_precision  # noqa: E402
from setup.problems_3d import analytical_sb_gaussian_3d, prob_gaussian_3d  # noqa: E402
from setup.projection_expsemi_3d import precomp_expsemi_proj_3d, proj_fokker_planck_expsemi_3d  # noqa: E402
from setup.state_3d import State3D  # noqa: E402

BACKEND = jax.default_backend()


def run_tag(vareps, n):
    return f"gaussian_eps{vareps:g}_n{n}_uniform_ETD"


def pick_times(t_grid, vareps, n_times=N_SNAPSHOTS):
    t_diff = np.inf if vareps <= 0 else SIGMA**2 / vareps
    if t_diff >= 0.5:
        want = np.linspace(t_grid[0], t_grid[-1], n_times)
    else:
        u = np.linspace(0.0, 1.0, n_times)
        want = t_grid[0] + (t_grid[-1] - t_grid[0]) * 0.5 * (1 - np.cos(np.pi * u))
    return np.unique([int(np.argmin(np.abs(t_grid - w))) for w in want])


def _peak_mem_mb():
    try:
        return jax.devices()[0].memory_stats().get("peak_bytes_in_use", 0) / 2**20
    except Exception:  # noqa: BLE001
        return float("nan")


def max_iter_for(vareps, n, long_points=True):
    return LONG_POINTS.get((vareps, n), MAX_ITER) if long_points else MAX_ITER


# ---------------------------------------------------------------- one run
def run_one(vareps, n, out_path, long_points=True):
    if vareps > ANALYTIC_MAX_EPS:
        raise ValueError(f"vareps={vareps:g}: no reference above {ANALYTIC_MAX_EPS:g} in 3D yet "
                         "(the closed form is free-space; a product-Sinkhorn reference is not ported)")
    prob_def = prob_gaussian_3d(mu0=MU0, mu1=MU1, sigma=SIGMA)
    tau = GAMMA * TAU_RATIO
    n_iter = max_iter_for(vareps, n, long_points)
    P3.set_dct_method(DCT_METHOD)

    jax.config.update("jax_enable_x64", True)
    problem = setup_problem_3d(prob_def, nt=n, nx=n, ny=n, nz=n)
    problem.ops = build_operators_3d(problem)
    ps, proj = precomp_expsemi_proj_3d(problem, vareps, method="dct"), proj_fokker_planck_expsemi_3d
    nt, nx, ny, nz = problem.nt, problem.nx, problem.ny, problem.nz
    dx, dy, dz, dt = problem.dx, problem.dy, problem.dz, problem.dt
    t_edges = np.arange(n + 1) * dt
    t_centers = (np.arange(1, n + 1) - 0.5) * dt
    dt_vec = np.full(n, dt)

    t0 = time.perf_counter()
    ref = tuple(np.asarray(a, dtype=np.float64) for a in analytical_sb_gaussian_3d(problem, vareps))
    refs = {"analytic": ref}
    print(f"    reference built ({time.perf_counter() - t0:.1f}s)", flush=True)

    problem_solve, ps, proj, prec_kw = configure_precision(PRECISION, problem, ps, proj, build_operators_3d)
    if PRECISION == "fp32":
        jax.config.update("jax_enable_x64", False)
    dtype = problem_solve.rho0.dtype
    x_refs = {nm: State3D(*(jnp.asarray(a, dtype) for a in r)) for nm, r in refs.items()}
    cfg = LadmmConfig(gamma=GAMMA, tau=tau, max_iter=n_iter, eps_abs=EPS_ABS, eps_rel=EPS_REL,
                      jit=True, check_every=CHECK_EVERY, diag_level=DIAG_LEVEL, **prec_kw)

    t0 = time.perf_counter()
    res = discretize_then_optimize_3d(problem_solve, proj, ps, vareps, cfg, x_refs=x_refs, track_rho=True)
    jax.block_until_ready(res.rho_stag)
    wall = time.perf_counter() - t0
    jax.config.update("jax_enable_x64", True)
    info = res.info
    solve_dtype = str(res.rho_stag.dtype) + (f"/{res.rho_cc.dtype}" if res.rho_cc.dtype != res.rho_stag.dtype else "")
    peak_mb = _peak_mem_mb()
    ms_iter = 1e3 * (wall - info.walltime_first_chunk) / max(info.iters - CHECK_EVERY, 1)
    rho = np.asarray(res.rho_stag, dtype=np.float64)
    mx, my, mz = (np.asarray(a, dtype=np.float64) for a in (res.mx_stag, res.my_stag, res.mz_stag))
    tail = "  ".join(f"{nm}: full={info.err_x_refs[nm][-1]:.3e} rho={info.err_rho_refs[nm][-1]:.3e}"
                     for nm in sorted(refs))
    print(f"    iters={info.iters} converged={info.converged} wall={wall:.1f}s "
          f"({ms_iter:.2f} ms/iter, compile {info.walltime_first_chunk:.1f}s, dtype {solve_dtype}, "
          f"peak GPU mem {peak_mb:.0f} MB)  D={info.D[-1]:.2e} P={info.P[-1]:.2e}", flush=True)
    print(f"    final error -- {tail}", flush=True)

    cell = dx * dy * dz
    idx = pick_times(t_edges[1:-1], vareps)
    diag = np.arange(nx)
    payload = dict(
        rho_diag=rho[:, diag, diag, diag], snap_idx=idx, snap_rho=rho[idx].sum(axis=3) * dz,
        mass_t=rho.sum(axis=(1, 2, 3)) * cell, min_rho_t=rho.min(axis=(1, 2, 3)),
        t_edges=t_edges, t_centers=t_centers, dt_vec=dt_vec,
        xx=np.asarray(problem.xx), dx=dx, dy=dy, dz=dz, nt=nt, nx=nx, ny=ny, nz=nz,
        D=info.D, P=info.P, scale_D=info.scale_D, scale_P=info.scale_P,
        dx_hist=info.dx, dy_hist=info.dy, dz_hist=info.dz,
        iters=info.iters, converged=info.converged, walltime=wall,
        walltime_first_chunk=info.walltime_first_chunk, ms_per_iter=ms_iter,
        peak_gpu_mem_mb=peak_mb, precision=PRECISION, solve_dtype=solve_dtype,
        **{f"err_full_{nm}": info.err_x_refs[nm] for nm in refs},
        **{f"err_rho_{nm}": info.err_rho_refs[nm] for nm in refs},
        case="gaussian", vareps=vareps, grid="uniform", scheme="ETD", solver="ladmm-jit",
        gamma=GAMMA, tau=tau, max_iter=n_iter, check_every=CHECK_EVERY, cluster_strength=np.nan,
        sigma=SIGMA, mu0=np.array(MU0), mu1=np.array(MU1), backend=BACKEND, dct_method=DCT_METHOD,
        ref_names=np.array(sorted(refs), dtype=object),
    )
    for nm, (r, m_x, m_y, m_z) in refs.items():
        payload[f"rows_{nm}_rho"] = cell * np.sum((rho - r) ** 2, axis=(1, 2, 3))
        payload[f"rows_{nm}_mx"] = cell * np.sum((mx - m_x) ** 2, axis=(1, 2, 3))
        payload[f"rows_{nm}_my"] = cell * np.sum((my - m_y) ** 2, axis=(1, 2, 3))
        payload[f"rows_{nm}_mz"] = cell * np.sum((mz - m_z) ** 2, axis=(1, 2, 3))
        payload[f"ref_{nm}_rho_diag"] = r[:, diag, diag, diag]
        payload[f"ref_{nm}_snap_rho"] = r[idx].sum(axis=3) * dz

    tmp = out_path.with_name(out_path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **payload)
    os.replace(tmp, out_path)
    print(f"    saved  {out_path.name}  ({out_path.stat().st_size / 1e6:.1f} MB)", flush=True)

    append_manifest(dict(
        vareps=vareps, n=n, grid="uniform", iters=info.iters, converged=int(info.converged),
        walltime_s=round(wall, 1), ms_per_iter=round(ms_iter, 3),
        peak_gpu_mem_mb=(round(peak_mb) if np.isfinite(peak_mb) else -1),
        precision=PRECISION, gamma=GAMMA, D_final=float(info.D[-1]), P_final=float(info.P[-1]),
        refs=";".join(sorted(refs)),
        **{f"err_rho_{nm}": float(info.err_rho_refs[nm][-1]) for nm in refs},
        **{f"err_full_{nm}": float(info.err_x_refs[nm][-1]) for nm in refs},
        file=out_path.name,
    ))
    # free the device before the next (larger) grid
    del res, x_refs, ps, problem, problem_solve
    return wall


MANIFEST = "manifest.csv"
MANIFEST_COLS = ["vareps", "n", "grid", "iters", "converged", "walltime_s", "ms_per_iter", "peak_gpu_mem_mb",
                 "precision", "gamma", "D_final", "P_final", "refs", "err_rho_analytic", "err_full_analytic", "file"]


def append_manifest(row):
    path = OUT_DIR / MANIFEST
    write_header = not path.exists() or path.stat().st_size == 0

    def fmt(v):
        if v is None:
            return ""
        return f"{v:.8e}" if isinstance(v, float) else str(v)

    with open(path, "a") as fh:
        if write_header:
            fh.write(",".join(MANIFEST_COLS) + "\n")
        fh.write(",".join(fmt(row.get(c)) for c in MANIFEST_COLS) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-preallocate", action="store_true")
    ap.add_argument("--mem-fraction", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--eps", type=float, action="append")
    ap.add_argument("--n", type=int, nargs="+", default=None)
    ap.add_argument("--precision", choices=PRECISIONS, default="fp64")
    ap.add_argument("--max-iter", type=int, default=None)
    ap.add_argument("--gamma", type=float, default=None, help="override GAMMA (tau follows)")
    ap.add_argument("--out-dir", default=None, help="override OUT_DIR, e.g. for a gamma probe")
    a = ap.parse_args()
    global MAX_ITER, GAMMA, OUT_DIR
    if a.max_iter is not None:
        MAX_ITER = a.max_iter
    if a.gamma is not None:
        GAMMA = a.gamma
    if PRECISION != "fp64" and a.out_dir is None:
        OUT_DIR = OUT_DIR.with_name(OUT_DIR.name + "_" + PRECISION)
    if a.out_dir is not None:
        OUT_DIR = Path(a.out_dir)

    epss = a.eps or EPS_LIST
    ns = a.n or RESOLUTIONS
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(e, n) for e in epss for n in ns]
    todo = [(e, n) for e, n in jobs if a.force or not (OUT_DIR / f"{run_tag(e, n)}.npz").exists()]

    # rough GH200 guess: 2D 256^3 (16.8M cells) was 6-7 ms/iter; scale linearly in cells
    per_it = {n: max(0.1, 7.0 * n**4 / 256**3) for n in ns}
    est = sum(per_it[n] * 1e-3 * max_iter_for(e, n) + 5 for e, n in todo)
    print(f"backend={BACKEND}  precision={PRECISION}  gamma={GAMMA:g}  {len(jobs)} runs requested, "
          f"{len(todo)} not yet on disk.  GH200 estimate ~{est / 60:.0f} min.  output: {OUT_DIR}")
    if a.dry_run:
        for e, n in todo:
            print(f"  {run_tag(e, n)}  iters={max_iter_for(e, n)}  ~{per_it[n]:.1f} ms/iter")
        return

    t_sweep = time.perf_counter()
    for i, (e, n) in enumerate(todo, 1):
        print(f"\n{'-' * 72}\n[{i}/{len(todo)}]  vareps={e:g}   nt=nx=ny=nz={n}\n{'-' * 72}", flush=True)
        run_one(e, n, OUT_DIR / f"{run_tag(e, n)}.npz")
        print(f"    elapsed {(time.perf_counter() - t_sweep) / 60:.1f} min", flush=True)
    print(f"\ndone -- {len(todo)} runs written to {OUT_DIR}\nsummary table: {OUT_DIR / MANIFEST}")


if __name__ == "__main__":
    main()
