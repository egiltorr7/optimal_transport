"""Batch runner for the 2D Gaussian grid-convergence study -- the 2D sibling
of scripts/run_suite.py. Solves once per (vareps, resolution), saves what the
plots need, plots nothing.

  case         gaussian  N((1/3,1/3), .05^2 I) -> N((2/3,2/3), .05^2 I)
  vareps       0, 1e-8, 1e-4, 1e-3, 1e-2, 1e-1, 1     (same list as 1D)
  time grid    uniform for vareps < 1e-2, clustered (strength 1) for >= 1e-2
  resolutions  nt = nx = ny = 32, 64, 128, 256
  scheme       ETD (projection_expsemi_2d / projection_expsemi_nu_2d)
  solver       ladmm, jit, gamma=0.1 (see memory: gamma scales with the density
               peak, 64 here vs 8 in 1D), tau=gamma*101/100, 40k iterations

References
  vareps <= 5e-3   "analytic": closed-form free-space SB (problems_2d /
                   problems_nu_2d), valid while the density does not feel the
                   walls -- at sigma=0.05, mu 1/3->2/3 the wall is 6.7 sigma out.
  vareps  > 5e-3   "sinkhorn_same" and "sinkhorn_fine", both EXACT for this
                   problem on the box: the marginals are products g(x)g(y) and
                   the Neumann heat kernel is K_x (x) K_y, so the 2D
                   discrete-marginal bridge is the product of two copies of the
                   1D bridge (Sinkhorn's fixed point is unique up to scaling,
                   and the product of the 1D potentials is a fixed point).
                   rho(t,x,y) = r(t,x) r(t,y), mx(t,x,y) = m(t,x) r(t,y). The
                   1D fits come from optimal_transport.sinkhorn exactly as
                   run_suite.py uses them ("same": fitted on this nx, isolates
                   the time error; "fine": nx=4096 + spectral resample, total
                   discretization error).

WHAT IS SAVED (per run)
  errors      rows_<ref>_rho (ntm,), rows_<ref>_mx / _my (nt,): per-time-row
              spatial L2^2 (dx*dy*sum over the plane) of the FINAL iterate
              against each reference -- post_grid_convergence_2d.py reduces
              these to slab / interior / max_t with the row weights, so no
              reference field needs to be stored.
              err_full_<ref>, err_rho_<ref> (iters,): per-iteration errors.
  fields      rho_diag (ntm, nx): rho on the diagonal x=y at EVERY time, and
              ref_<ref>_rho_diag; snap_idx (k,), snap_rho (k, nx, ny),
              ref_<ref>_snap_rho: full planes at k representative times
              (pick_times, as post_density_profiles.py chooses them). The
              full (ntm, nx, ny) field is NOT saved: at n=256 it is 128 MB per
              array and the diagonal + snapshots carry what the plots draw.
              mass_t (ntm,), min_rho_t (ntm,).
  grids       t_edges, t_centers, dt_vec, xx, dx, nt, nx, ny
  histories   D, P, scale_D, scale_P, dx_hist, dy_hist, dz_hist (iters,)
  metadata    vareps, grid, scheme, gamma, tau, iters, converged, walltime,
              walltime_first_chunk, ref_names, backend

Precision: --precision fp32 solves in float32 (setup, precompute and references
stay float64 and are cast; matmul precision pinned to 'highest'), writes to
results/suite_fp32/, and records ms_per_iter / peak_gpu_mem_mb / solve_dtype
per run for the cost comparison (post_cost_2d.py). --precision mixed keeps
the LADMM state, consensus maps, KE prox and norms in float32 but runs the FP
projection in float64 (setup/precision.py explains which piece needs it and
why); writes to results/suite_mixed/. --precision mixedm is mixed plus float64
master copies of x and delta (ladmm_jit.py's master_dtype; the mixed suite
stalled at 4e-4 because float32 cannot hold LADMM's sub-ulp late steps);
writes to results/suite_mixedm/. --precision mixedc is mixedm with the
KE prox arithmetic in float64 (output stored float32) and the consistent dual
update (dual = -grad KE at the stored float32 y, in float64; ladmm_jit.py) --
the variant meant to remove the float32 floor; writes to results/suite_mixedc/. Diagnostics: fp64rx is fp64 everywhere except
x is rounded to float32 before the consensus map A; fp64px is fp64 everywhere
except the KE prox runs in float32 (so y, By, D are float32). They isolate
which float32 rounding the mixed modes' error floor comes from.

Usage:
    python scripts/run_suite_2d.py --dry-run
    python scripts/run_suite_2d.py --device gpu
    python scripts/run_suite_2d.py --device gpu --precision fp32
    python scripts/run_suite_2d.py --device gpu --precision mixed
    python scripts/run_suite_2d.py --device gpu --eps 1e-4 --n 32 64
    python scripts/run_suite_2d.py --force
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))          # 2d/      -> setup
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))   # python/  -> optimal_transport
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))  # 1D study_utils

# ===========================================================================
# CONFIG -- edit here, or override from the command line
# ===========================================================================
EPS_LIST = [0.0, 1e-8, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
RESOLUTIONS = [32, 64, 128, 256]
MU0, MU1, SIGMA = 1 / 3, 2 / 3, 0.05

GAMMA = 0.1
TAU_RATIO = 101.0 / 100.0
MAX_ITER = 40_000
EPS_ABS = 1e-10
EPS_REL = 1e-15
CHECK_EVERY = 100            # iterations per compiled call (ladmm_jit.py)
DIAG_LEVEL = "full"

GRID_SWITCH_EPS = 1e-2       # vareps >= this -> clustered; below -> uniform
CLUSTER_STRENGTH = 1.0
N_SNAPSHOTS = 9
DCT_METHOD = "matmul"        # projection_2d.set_dct_method (fastest on GPU for n <= 512)

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "suite"   # fp64; fp32/mixed -> suite_fp32/suite_mixed
# Points that were still descending at 40k in the fp64 study and were rerun at
# 150k (see memory: 2d-grid-convergence-suite). Applied by default so a new
# sweep reproduces the FINAL fp64 data set with one command; --no-long-points
# gives the flat 40k everywhere.
LONG_POINTS = {(0.0, 256): 150_000, (1e-8, 256): 150_000, (1e-4, 256): 150_000,
               (1e-1, 128): 150_000, (1e-1, 256): 150_000, (1.0, 128): 150_000, (1.0, 256): 150_000}
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

# x64 is switched ON here regardless of --precision: the problem setup, the
# per-mode precompute and the references are always built in float64 (the
# mode matrices subtract 1/dt^2-sized terms from each other and would lose
# ~4 digits in float32). For fp32 the solve-time arrays are cast to float32
# and x64 is switched off before the operators/solver are built; for mixed
# x64 stays on (the projection needs float64 to exist) and the float32 half
# is float32 purely by the dtype of the cast problem -- see run_one.
jax_config.configure(device=_PRE.device, x64=True,
                     preallocate=False if _PRE.no_preallocate else None,
                     mem_fraction=_PRE.mem_fraction)

import jax  # noqa: E402

if PRECISION in ("fp32", "mixed", "mixedm", "mixedc"):
    # Never let a float32 matmul (the DCT) fall back to TF32 tensor cores
    # (10-bit mantissa). No-op for mixed, whose DCTs are float64.
    jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from setup import projection_2d as P2  # noqa: E402
from setup.grid_2d import setup_problem_2d  # noqa: E402
from setup.grid_nu_2d import setup_problem_nu_2d  # noqa: E402
from setup.operators_2d import build_operators_2d  # noqa: E402
from setup.operators_nu_2d import build_operators_nu_2d  # noqa: E402
from setup.pipeline_2d import LadmmConfig, discretize_then_optimize_2d  # noqa: E402
from setup.pipeline_nu_2d import LadmmConfig as LadmmConfigNU  # noqa: E402
from setup.pipeline_nu_2d import discretize_then_optimize_nu_2d  # noqa: E402
from setup.precision import cast_problem, mixed_projection  # noqa: E402
from setup.problems_2d import analytical_sb_gaussian_2d, prob_gaussian_2d  # noqa: E402
from setup.problems_nu_2d import analytical_sb_gaussian_nu_2d  # noqa: E402
from setup.projection_expsemi_2d import precomp_expsemi_proj_2d, proj_fokker_planck_expsemi_2d  # noqa: E402
from setup.projection_expsemi_nu_2d import (precomp_expsemi_proj_nu_2d,  # noqa: E402
                                            proj_fokker_planck_expsemi_nu_2d)
from setup.state_2d import State2D  # noqa: E402
from setup.time_grid import clustered_time_grid  # noqa: E402

from optimal_transport.grid import setup_problem as setup_problem_1d  # noqa: E402
from optimal_transport.problems import prob_gaussian as prob_gaussian_1d  # noqa: E402
from optimal_transport.sinkhorn import sinkhorn_eval, sinkhorn_fit  # noqa: E402
from study_utils import (SINKHORN_MAX_ITER, SINKHORN_NX_FINE, SINKHORN_TOL,  # noqa: E402
                         SINKHORN_VAREPS_THRESHOLD, _coarsen_mx, resample_rho_spectral)

BACKEND = jax.default_backend()


def grid_kind(vareps):
    return "clustered" if vareps >= GRID_SWITCH_EPS else "uniform"


def run_tag(vareps, n):
    return f"gaussian_eps{vareps:g}_n{n}_{grid_kind(vareps)}_ETD"


def pick_times(t_grid, vareps, n_times=N_SNAPSHOTS):
    """Same rule as post_density_profiles.py: uniform in t while the diffusion
    time sigma^2/vareps exceeds 1/2, cosine-clustered toward both ends once
    diffusion flattens the structure early."""
    t_diff = np.inf if vareps <= 0 else SIGMA**2 / vareps
    if t_diff >= 0.5:
        want = np.linspace(t_grid[0], t_grid[-1], n_times)
    else:
        u = np.linspace(0.0, 1.0, n_times)
        want = t_grid[0] + (t_grid[-1] - t_grid[0]) * 0.5 * (1 - np.cos(np.pi * u))
    return np.unique([int(np.argmin(np.abs(t_grid - w))) for w in want])


# ---------------------------------------------------------------- references
def _product_ref(fit, t_edge, t_cc, nx):
    """(rho, mx, my) on the 2D staggered grid from a 1D SinkhornFit, via the
    product structure (module docstring). rho at the interior edge times,
    mx/my at the cell-centre times; spatially reduced to nx exactly as
    run_suite.py's _sinkhorn_ref does."""
    factor = fit.phi_0.shape[0] // nx
    r_e, _ = sinkhorn_eval(fit, t_edge)          # (ntm, nx_fine)
    r_c, m_c = sinkhorn_eval(fit, t_cc)          # (nt, nx_fine), (nt, nx_fine-1)
    r_e = resample_rho_spectral(r_e, nx)
    r_c = resample_rho_spectral(r_c, nx)
    m_c = _coarsen_mx(m_c, factor)
    rho = r_e[:, :, None] * r_e[:, None, :]      # (ntm, nx, nx)
    mx = m_c[:, :, None] * r_c[:, None, :]       # (nt, nx-1, nx)
    my = r_c[:, :, None] * m_c[:, None, :]       # (nt, nx, nx-1)
    return rho, mx, my


def build_references(problem, vareps, kind):
    refs = {}
    if vareps <= SINKHORN_VAREPS_THRESHOLD:
        ana = (analytical_sb_gaussian_nu_2d(problem, vareps) if kind == "clustered"
               else analytical_sb_gaussian_2d(problem, vareps))
        refs["analytic"] = tuple(np.asarray(a) for a in ana)
        return refs, ""

    nx, nt = problem.nx, problem.nt
    if kind == "clustered":
        t_edge, t_cc = np.asarray(problem.t_edges)[1:-1], np.asarray(problem.t_centers)
    else:
        t_edge = np.arange(1, nt) * problem.dt
        t_cc = (np.arange(1, nt + 1) - 0.5) * problem.dt
    pd1 = prob_gaussian_1d(mu0=MU0, mu1=MU1, sigma=SIGMA)
    p_same = setup_problem_1d(pd1, nt=2, nx=nx)
    f_same = sinkhorn_fit(p_same, vareps, SINKHORN_MAX_ITER, SINKHORN_TOL)
    refs["sinkhorn_same"] = _product_ref(f_same, t_edge, t_cc, nx)
    p_fine = setup_problem_1d(pd1, nt=2, nx=SINKHORN_NX_FINE)
    f_fine = sinkhorn_fit(p_fine, vareps, SINKHORN_MAX_ITER, SINKHORN_TOL)
    refs["sinkhorn_fine"] = _product_ref(f_fine, t_edge, t_cc, nx)
    # the 2D marginal really is the product of the 1D ones (checked, not assumed)
    r0 = np.asarray(p_same.rho0)
    dev = float(np.abs(np.asarray(problem.rho0) - r0[:, None] * r0[None, :]).max())
    note = (f"sinkhorn same: iters={f_same.iters} err={f_same.error:.1e} | "
            f"fine(nx={SINKHORN_NX_FINE}): iters={f_fine.iters} err={f_fine.error:.1e} | "
            f"2D rho0 == 1D x 1D to {dev:.1e}")
    return refs, note


def _peak_mem_mb():
    try:
        return jax.devices()[0].memory_stats().get("peak_bytes_in_use", 0) / 2**20
    except Exception:  # noqa: BLE001  (CPU backend has no memory_stats)
        return float("nan")


def max_iter_for(vareps, n, long_points=True):
    return LONG_POINTS.get((vareps, n), MAX_ITER) if long_points else MAX_ITER


# ---------------------------------------------------------------- one run
def run_one(vareps, n, out_path, long_points=True):
    kind = grid_kind(vareps)
    prob_def = prob_gaussian_2d(mu0=(MU0, MU0), mu1=(MU1, MU1), sigma=SIGMA)
    tau = GAMMA * TAU_RATIO
    n_iter = max_iter_for(vareps, n, long_points)
    P2.set_dct_method(DCT_METHOD)

    # Everything up to and including the precompute in float64 (see the note
    # at the top), then cast for the solve when running fp32.
    jax.config.update("jax_enable_x64", True)
    if kind == "clustered":
        problem = setup_problem_nu_2d(prob_def, clustered_time_grid(n, CLUSTER_STRENGTH), nx=n, ny=n)
        problem.ops = build_operators_nu_2d(problem)
        ps, proj = precomp_expsemi_proj_nu_2d(problem, vareps), proj_fokker_planck_expsemi_nu_2d
        t_edges, t_centers, dt_vec = (np.asarray(problem.t_edges), np.asarray(problem.t_centers),
                                      np.asarray(problem.dt_vec))
    else:
        problem = setup_problem_2d(prob_def, nt=n, nx=n, ny=n)
        problem.ops = build_operators_2d(problem)
        ps, proj = precomp_expsemi_proj_2d(problem, vareps, method="dct"), proj_fokker_planck_expsemi_2d
        t_edges = np.arange(n + 1) * problem.dt
        t_centers = (np.arange(1, n + 1) - 0.5) * problem.dt
        dt_vec = np.full(n, problem.dt)
    nt, nx, ny, dx, dy = problem.nt, problem.nx, problem.ny, problem.dx, problem.dy

    t0 = time.perf_counter()
    refs, note = build_references(problem, vareps, kind)   # float64 numpy
    if note:
        print(f"    {note}  ({time.perf_counter() - t0:.1f}s)", flush=True)

    build_ops = build_operators_nu_2d if kind == "clustered" else build_operators_2d
    master = None
    if PRECISION == "fp32":
        dtype = jnp.float32
        problem = cast_problem(problem, dtype)
        ps = jax.tree_util.tree_map(lambda a: a.astype(dtype), ps)
        jax.config.update("jax_enable_x64", False)
        # operators allocate their BC zero-planes at build time -> rebuild so
        # they are float32 too (a float64 plane would promote the whole solve)
        problem.ops = build_ops(problem)
    elif PRECISION in ("mixed", "mixedm", "mixedc"):
        # float32 problem (+ float32 zero planes) drives the LADMM half; the
        # float64 problem, ops and precompute stay inside the projection.
        # mixedm: x and delta are float64 master copies, so the projection
        # returns float64 (no output cast) and the solver casts x down for A.
        dtype = jnp.float32
        problem_hi = problem
        problem = cast_problem(problem, dtype)
        problem.ops = build_ops(problem)
        master = jnp.float64 if PRECISION in ("mixedm", "mixedc") else None
        proj = mixed_projection(proj, problem_hi, lo_dtype=(None if master else dtype), hi_dtype=jnp.float64)
    else:
        dtype = jnp.float64
        # both diagnostics keep x and delta float64 (fp64px needs it: y/D come
        # back float32 and delta would otherwise change dtype in the scan)
        master = jnp.float64 if PRECISION in ("fp64rx", "fp64px") else None
    x_refs = {nm: State2D(jnp.asarray(r, dtype), jnp.asarray(mx, dtype), jnp.asarray(my, dtype))
              for nm, (r, mx, my) in refs.items()}
    Cfg = LadmmConfigNU if kind == "clustered" else LadmmConfig
    cfg = Cfg(gamma=GAMMA, tau=tau, max_iter=n_iter, eps_abs=EPS_ABS, eps_rel=EPS_REL,
              jit=True, check_every=CHECK_EVERY, diag_level=DIAG_LEVEL, master_dtype=master,
              round_x_dtype={"fp64rx": jnp.float32, "fp64px": jnp.float64}.get(PRECISION),
              prox_dtype={"fp64px": jnp.float32, "mixedc": jnp.float64}.get(PRECISION),
              y_dtype={"fp64px": jnp.float32, "mixedc": jnp.float32}.get(PRECISION),
              consistent_dual=(PRECISION == "mixedc"))
    solve = discretize_then_optimize_nu_2d if kind == "clustered" else discretize_then_optimize_2d

    t0 = time.perf_counter()
    res = solve(problem, proj, ps, vareps, cfg, x_refs=x_refs, track_rho=True)
    jax.block_until_ready(res.rho_stag)
    wall = time.perf_counter() - t0
    info = res.info
    solve_dtype = str(res.rho_stag.dtype) + (f"/{res.rho_cc.dtype}" if res.rho_cc.dtype != res.rho_stag.dtype else "")
    want_cc = jnp.float32 if PRECISION == "fp64px" else dtype
    if res.rho_cc.dtype != jnp.dtype(want_cc) or res.rho_stag.dtype != jnp.dtype(master or dtype):
        raise RuntimeError(f"LADMM state is {res.rho_cc.dtype}, expected {dtype}: a float64 array leaked "
                           "into the float32 half (see setup/precision.py)")
    peak_mb = _peak_mem_mb()
    ms_iter = 1e3 * (wall - info.walltime_first_chunk) / max(info.iters - CHECK_EVERY, 1)
    rho = np.asarray(res.rho_stag, dtype=np.float64)
    mx, my = np.asarray(res.mx_stag, dtype=np.float64), np.asarray(res.my_stag, dtype=np.float64)
    tail = "  ".join(f"{nm}: full={info.err_x_refs[nm][-1]:.3e} rho={info.err_rho_refs[nm][-1]:.3e}"
                     for nm in sorted(refs))
    print(f"    iters={info.iters} converged={info.converged} wall={wall:.1f}s "
          f"({ms_iter:.2f} ms/iter, compile {info.walltime_first_chunk:.1f}s, dtype {solve_dtype}, "
          f"peak GPU mem {peak_mb:.0f} MB)  D={info.D[-1]:.2e} P={info.P[-1]:.2e}", flush=True)
    print(f"    final error -- {tail}", flush=True)

    # --- per-time-row spatial errors of the final iterate, per reference ---
    cell = dx * dy
    idx = pick_times(t_edges[1:-1], vareps)
    diag = np.arange(nx)
    payload = dict(
        rho_diag=rho[:, diag, diag], snap_idx=idx, snap_rho=rho[idx],
        mass_t=rho.sum(axis=(1, 2)) * cell, min_rho_t=rho.min(axis=(1, 2)),
        t_edges=t_edges, t_centers=t_centers, dt_vec=dt_vec,
        xx=np.asarray(problem.xx), dx=dx, dy=dy, nt=nt, nx=nx, ny=ny,
        D=info.D, P=info.P, scale_D=info.scale_D, scale_P=info.scale_P,
        dx_hist=info.dx, dy_hist=info.dy, dz_hist=info.dz,
        iters=info.iters, converged=info.converged, walltime=wall,
        walltime_first_chunk=info.walltime_first_chunk, ms_per_iter=ms_iter,
        peak_gpu_mem_mb=peak_mb, precision=PRECISION, solve_dtype=solve_dtype,
        **{f"err_full_{nm}": info.err_x_refs[nm] for nm in refs},
        **{f"err_rho_{nm}": info.err_rho_refs[nm] for nm in refs},
        case="gaussian", vareps=vareps, grid=kind, scheme="ETD", solver="ladmm-jit",
        gamma=GAMMA, tau=tau, max_iter=n_iter, check_every=CHECK_EVERY,
        cluster_strength=(CLUSTER_STRENGTH if kind == "clustered" else np.nan),
        sigma=SIGMA, mu0=MU0, mu1=MU1, backend=BACKEND, dct_method=DCT_METHOD,
        ref_names=np.array(sorted(refs), dtype=object),
    )
    for nm, (r, m_x, m_y) in refs.items():
        payload[f"rows_{nm}_rho"] = cell * np.sum((rho - r) ** 2, axis=(1, 2))
        payload[f"rows_{nm}_mx"] = cell * np.sum((mx - m_x) ** 2, axis=(1, 2))
        payload[f"rows_{nm}_my"] = cell * np.sum((my - m_y) ** 2, axis=(1, 2))
        payload[f"ref_{nm}_rho_diag"] = r[:, diag, diag]
        payload[f"ref_{nm}_snap_rho"] = r[idx]

    tmp = out_path.with_name(out_path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **payload)
    os.replace(tmp, out_path)
    print(f"    saved  {out_path.name}  ({out_path.stat().st_size / 1e6:.1f} MB)", flush=True)

    append_manifest(dict(
        vareps=vareps, n=n, grid=kind, iters=info.iters, converged=int(info.converged),
        walltime_s=round(wall, 1), ms_per_iter=round(ms_iter, 3), peak_gpu_mem_mb=(round(peak_mb) if np.isfinite(peak_mb) else -1),
        precision=PRECISION, D_final=float(info.D[-1]), P_final=float(info.P[-1]),
        refs=";".join(sorted(refs)),
        **{f"err_rho_{nm}": float(info.err_rho_refs[nm][-1]) for nm in refs},
        **{f"err_full_{nm}": float(info.err_x_refs[nm][-1]) for nm in refs},
        file=out_path.name,
    ))
    return wall


MANIFEST = "manifest.csv"
MANIFEST_COLS = ["vareps", "n", "grid", "iters", "converged", "walltime_s", "ms_per_iter", "peak_gpu_mem_mb",
                 "precision", "D_final", "P_final", "refs",
                 "err_rho_analytic", "err_rho_sinkhorn_same", "err_rho_sinkhorn_fine",
                 "err_full_analytic", "err_full_sinkhorn_same", "err_full_sinkhorn_fine", "file"]


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
    ap.add_argument("--n", type=int, action="append")
    ap.add_argument("--precision", choices=PRECISIONS, default="fp64")
    ap.add_argument("--max-iter", type=int, default=None, help="override MAX_ITER (and disables LONG_POINTS)")
    ap.add_argument("--no-long-points", dest="long_points", action="store_false", default=True)
    ap.add_argument("--gamma", type=float, default=None, help="override GAMMA (tau follows)")
    ap.add_argument("--out-dir", default=None, help="override OUT_DIR, e.g. for a gamma probe")
    a = ap.parse_args()
    global MAX_ITER, GAMMA, OUT_DIR
    long_points = a.long_points
    if a.max_iter is not None:
        MAX_ITER = a.max_iter
        long_points = False
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

    # GH200 steady-state ms/iter measured 2026-09-24 (ETD, x64): uniform/dct
    # 0.12, 0.23, 1.1, ~7 at n=32..256; clustered/scan-Thomas ~0.6, 1.3, 3.4, 13.
    per_it = {"uniform": {32: 0.12, 64: 0.25, 128: 1.1, 256: 7.0},
              "clustered": {32: 0.6, 64: 1.3, 128: 3.4, 256: 13.0}}
    # measured 2026-09-26: fp32 1.05-1.6x; mixed guessed at 1.3x until measured
    speedup = {"fp64": 1.0, "fp32": 1.4, "mixed": 1.35, "mixedm": 1.3, "mixedc": 1.25}.get(PRECISION, 1.0)
    est = sum(per_it[grid_kind(e)].get(n, 60.0) * 1e-3 * max_iter_for(e, n, long_points) / speedup + 2
              for e, n in todo)
    print(f"backend={BACKEND}  precision={PRECISION}  {len(jobs)} runs requested, {len(todo)} not yet on disk.  "
          f"GH200 estimate ~{est / 60:.0f} min.  output: {OUT_DIR}")
    if a.dry_run:
        for e, n in todo:
            print(f"  {run_tag(e, n)}  iters={max_iter_for(e, n, long_points)}")
        return

    t_sweep = time.perf_counter()
    for i, (e, n) in enumerate(todo, 1):
        print(f"\n{'-' * 72}\n[{i}/{len(todo)}]  vareps={e:g}   nt=nx=ny={n}   grid={grid_kind(e)}\n{'-' * 72}",
              flush=True)
        run_one(e, n, OUT_DIR / f"{run_tag(e, n)}.npz", long_points)
        print(f"    elapsed {(time.perf_counter() - t_sweep) / 60:.1f} min", flush=True)
    print(f"\ndone -- {len(todo)} runs written to {OUT_DIR}\nsummary table: {OUT_DIR / MANIFEST}")


if __name__ == "__main__":
    main()
