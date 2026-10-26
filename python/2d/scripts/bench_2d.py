"""Per-iteration timing of the 2D LADMM solve across the dispatch/solver
variants, on whatever backend jax_config picks.

    python scripts/bench_2d.py --device gpu --sizes 32 64 128 --iters 50

Variants (each an independent knob):
  loop:    eager (ladmm.py)  |  jit (ladmm_jit.py, check_every=1)  |  jit10 (check_every=10)
  modes:   thomas (lax.scan Thomas sweep)  |  dct (DCT-in-t + rank-2 Woodbury)
  dct:     fft2n | fftn | matmul   (projection_2d.set_dct_method)
  scheme:  cn | etd

Prints ms/iter (steady state: the first call compiles, and is timed
separately as "compile"), and peak device memory when the backend reports it.
Correctness is checked as a side effect: every variant's D-residual history
over the timed iterations is compared with the eager/thomas/fft2n baseline.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="auto")
ap.add_argument("--sizes", type=int, nargs="+", default=[32, 64])
ap.add_argument("--nt", type=int, default=None, help="time steps (default: same as spatial size)")
ap.add_argument("--iters", type=int, default=50)
ap.add_argument("--eps", type=float, default=1e-3)
ap.add_argument("--schemes", nargs="+", default=["cn", "etd"])
ap.add_argument("--loops", nargs="+", default=["eager", "jit", "jit10"], help="eager | jit | jit<k> (check_every=k)")
ap.add_argument("--modes", nargs="+", default=["thomas", "dct"])
ap.add_argument("--dcts", nargs="+", default=["fft2n", "fftn", "matmul"])
ap.add_argument("--eager-max", type=int, default=64, help="skip eager above this size")
ap.add_argument("--diag", nargs="+", default=["full"], help="jit diag_level(s): full, stopping")
ap.add_argument("--grid", default="uniform", help="uniform | clustered (the *_nu_2d path, CN/thomas only)")
ap.add_argument("--no-preallocate", action="store_true")
args = ap.parse_args()

from setup import jax_config

jax_config.configure(device=args.device, x64=True, preallocate=False if args.no_preallocate else None)

import jax
import jax.numpy as jnp
import numpy as np

from setup import projection_2d as P
from setup.grid_2d import setup_problem_2d
from setup.operators_2d import build_operators_2d
from setup.pipeline_2d import LadmmConfig, discretize_then_optimize_2d
from setup.problems_2d import prob_gaussian_2d
from setup.projection_2d import precomp_banded_proj_2d, proj_fokker_planck_banded_2d
from setup.projection_expsemi_2d import precomp_expsemi_proj_2d, proj_fokker_planck_expsemi_2d

if args.grid != "uniform":
    from setup.grid_nu_2d import setup_problem_nu_2d
    from setup.operators_nu_2d import build_operators_nu_2d
    from setup.pipeline_nu_2d import LadmmConfig as LadmmConfigNU
    from setup.pipeline_nu_2d import discretize_then_optimize_nu_2d
    from setup.projection_nu_2d import precomp_banded_proj_nu_2d, proj_fokker_planck_banded_nu_2d
    from setup.time_grid import clustered_time_grid

print(f"JAX {jax.__version__} backend={jax.default_backend()} devices={jax.devices()} x64={jax.config.jax_enable_x64}")
PD = prob_gaussian_2d(mu0=(0.4, 0.4), mu1=(0.6, 0.6), sigma=0.08)


def peak_mem_mb():
    try:
        st = jax.devices()[0].memory_stats()
        return st.get("peak_bytes_in_use", 0) / 2**20
    except Exception:
        return float("nan")


def run(p, scheme, loop, modes, dct, diag, iters):
    P.set_dct_method(dct)
    nu = args.grid != "uniform"
    if nu:
        ps, proj, dto, Cfg = precomp_banded_proj_nu_2d(p, args.eps), proj_fokker_planck_banded_nu_2d, discretize_then_optimize_nu_2d, LadmmConfigNU
    elif scheme == "cn":
        ps, proj, dto, Cfg = precomp_banded_proj_2d(p, args.eps, modes), proj_fokker_planck_banded_2d, discretize_then_optimize_2d, LadmmConfig
    else:
        ps, proj, dto, Cfg = precomp_expsemi_proj_2d(p, args.eps, modes), proj_fokker_planck_expsemi_2d, discretize_then_optimize_2d, LadmmConfig
    jax.block_until_ready(ps)
    jit = loop != "eager"
    ce = int(loop[3:]) if loop.startswith("jit") and len(loop) > 3 else 1
    n_it = ce * max(1, iters // ce) + ce      # whole chunks; the first chunk is the compile
    cfg = Cfg(gamma=1.0, tau=1.01, max_iter=n_it, eps_abs=0.0, eps_rel=0.0)
    if jit:
        cfg.jit, cfg.check_every, cfg.diag_level = True, ce, diag
    t0 = time.perf_counter()
    r = dto(p, proj, ps, args.eps, cfg)
    jax.block_until_ready(r.rho_stag)
    t = time.perf_counter() - t0
    info = r.info
    first = info.walltime_first_chunk if jit else t / n_it
    n_first = ce if jit else 1
    ms = 1e3 * (info.walltime - first) / (n_it - n_first)
    return ms, first, np.asarray(info.D), r


for n in args.sizes:
    nt = args.nt or n
    if args.grid == "uniform":
        p = setup_problem_2d(PD, nt=nt, nx=n, ny=n)
        p.ops = build_operators_2d(p)
    else:
        p = setup_problem_nu_2d(PD, clustered_time_grid(nt, 1.0), nx=n, ny=n)
        p.ops = build_operators_nu_2d(p)
    print(f"\n=== nt={nt} nx=ny={n}  ({(nt * n * n * 8) / 2**20:.1f} MB per (nt,nx,ny) float64 array), "
          f"eps={args.eps:g}, {args.iters} timed iters ===")
    print(f"{'scheme':>6s} {'loop':>6s} {'modes':>7s} {'dct':>7s} {'diag':>8s} {'ms/iter':>9s} {'compile s':>10s} {'peak MB':>9s}  D-hist vs baseline")
    schemes = ["cn"] if args.grid != "uniform" else args.schemes
    modes_l = ["thomas"] if args.grid != "uniform" else args.modes
    for scheme in schemes:
        base = None
        for loop in args.loops:
            if loop == "eager" and n > args.eager_max:
                continue
            for modes in modes_l:
                for dct in args.dcts:
                    for diag in (["full"] if loop == "eager" else args.diag):
                        if loop == "eager" and (modes != "thomas" or dct != "fft2n"):
                            continue  # eager only as the baseline
                        try:
                            ms, tc, D, _ = run(p, scheme, loop, modes, dct, diag, args.iters)
                        except Exception as ex:  # noqa: BLE001
                            print(f"{scheme:>6s} {loop:>6s} {modes:>7s} {dct:>7s} {diag:>8s}   FAILED: {type(ex).__name__}: {str(ex)[:80]}")
                            continue
                        if base is None:
                            base, dev = D, 0.0
                        else:
                            m = min(len(base), len(D))
                            dev = float(np.abs(base[:m] - D[:m]).max() / np.abs(base[:m]).max())
                        print(f"{scheme:>6s} {loop:>6s} {modes:>7s} {dct:>7s} {diag:>8s} {ms:9.3f} {tc:10.2f} {peak_mem_mb():9.0f}  {dev:.1e}", flush=True)
