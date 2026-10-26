"""Per-iteration cost of the 3D solver's knobs -- 3D port of scripts/bench_2d.py.

Times LADMM iterations at steady state (the first compiled chunk, i.e. the
compile, is excluded: see the "recompile per solve" pitfall in
docs/gpu_mixed_precision_2d.tex) for each combination of

  scheme     cn | etd
  loop       eager | jit | jit<k>   (check_every = k)
  modes      thomas | dct            (per-mode time solver)
  dct        fft2n | fftn | matmul   (DCT kernel)
  diag       full | stopping
  precision  fp64 | fp32 | mixed | mixedm | mixedc   (setup/precision.py)

and reports ms/iter, compile time and the process-wide peak device memory
(read the FIRST occurrence per size: the field never decreases).

Usage (GH200):
    python scripts/bench_3d.py --device gpu --sizes 32 48 64 --iters 100 --loops jit50 --modes dct --dcts matmul
    python scripts/bench_3d.py --device gpu --sizes 64 --loops jit50 --modes dct --dcts matmul --precisions fp64 mixedc
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--device", default="auto")
ap.add_argument("--sizes", type=int, nargs="+", default=[16, 32])
ap.add_argument("--nt", type=int, default=None, help="time steps (default: same as spatial size)")
ap.add_argument("--iters", type=int, default=50)
ap.add_argument("--eps", type=float, default=1e-3)
ap.add_argument("--schemes", nargs="+", default=["cn", "etd"])
ap.add_argument("--loops", nargs="+", default=["eager", "jit", "jit10"], help="eager | jit | jit<k>")
ap.add_argument("--modes", nargs="+", default=["thomas", "dct"])
ap.add_argument("--dcts", nargs="+", default=["fft2n", "fftn", "matmul"])
ap.add_argument("--diag", nargs="+", default=["full"], help="jit diag_level(s): full, stopping")
ap.add_argument("--precisions", nargs="+", default=["fp64"], help="fp64 | fp32 | mixed | mixedm | mixedc")
ap.add_argument("--eager-max", type=int, default=16, help="skip eager above this size")
ap.add_argument("--no-preallocate", action="store_true")
ap.add_argument("--mem-fraction", type=float, default=None)
args = ap.parse_args()

from setup import jax_config  # noqa: E402

jax_config.configure(device=args.device, x64=True, preallocate=False if args.no_preallocate else None,
                     mem_fraction=args.mem_fraction)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from setup import projection_3d as P  # noqa: E402
from setup.grid_3d import setup_problem_3d  # noqa: E402
from setup.operators_3d import build_operators_3d  # noqa: E402
from setup.pipeline_3d import LadmmConfig, discretize_then_optimize_3d  # noqa: E402
from setup.precision import configure_precision  # noqa: E402
from setup.problems_3d import prob_gaussian_3d  # noqa: E402
from setup.projection_3d import precomp_banded_proj_3d, proj_fokker_planck_banded_3d  # noqa: E402
from setup.projection_expsemi_3d import precomp_expsemi_proj_3d, proj_fokker_planck_expsemi_3d  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")   # never TF32 for a float32 DCT
print(f"JAX {jax.__version__} backend={jax.default_backend()} devices={jax.devices()} x64={jax.config.jax_enable_x64}")
PD = prob_gaussian_3d(sigma=0.08)


def peak_mem_mb():
    try:
        return jax.devices()[0].memory_stats().get("peak_bytes_in_use", 0) / 2**20
    except Exception:  # noqa: BLE001
        return float("nan")


def run(p, scheme, loop, modes, dct, diag, precision, iters):
    P.set_dct_method(dct)
    if scheme == "cn":
        ps, proj = precomp_banded_proj_3d(p, args.eps, modes), proj_fokker_planck_banded_3d
    else:
        ps, proj = precomp_expsemi_proj_3d(p, args.eps, modes), proj_fokker_planck_expsemi_3d
    prob, ps, proj, kw = configure_precision(precision, p, ps, proj, build_operators_3d)
    jax.block_until_ready(ps)
    jit = loop != "eager"
    ce = int(loop[3:]) if loop.startswith("jit") and len(loop) > 3 else 1
    n_it = ce * max(1, iters // ce) + ce      # whole chunks; the first chunk is the compile
    cfg = LadmmConfig(gamma=0.3, tau=0.303, max_iter=n_it, eps_abs=0.0, eps_rel=0.0, **kw)
    if jit:
        cfg.jit, cfg.check_every, cfg.diag_level = True, ce, diag
    if precision == "fp32":
        jax.config.update("jax_enable_x64", False)
    try:
        t0 = time.perf_counter()
        r = discretize_then_optimize_3d(prob, proj, ps, args.eps, cfg)
        jax.block_until_ready(r.rho_stag)
        t = time.perf_counter() - t0
    finally:
        jax.config.update("jax_enable_x64", True)
    info = r.info
    first = info.walltime_first_chunk if jit else t / n_it
    n_first = ce if jit else 1
    ms = 1e3 * (info.walltime - first) / (n_it - n_first)
    return ms, first, np.asarray(info.D), r


for n in args.sizes:
    nt = args.nt or n
    p = setup_problem_3d(PD, nt=nt, nx=n, ny=n, nz=n)
    p.ops = build_operators_3d(p)
    print(f"\n=== nt={nt} nx=ny=nz={n}  ({(nt * n**3 * 8) / 2**20:.1f} MB per (nt,nx,ny,nz) float64 array), "
          f"eps={args.eps:g}, {args.iters} timed iters ===")
    print(f"{'scheme':>6s} {'loop':>6s} {'modes':>7s} {'dct':>7s} {'diag':>8s} {'prec':>7s} {'ms/iter':>9s} "
          f"{'compile s':>10s} {'peak MB':>9s}  D-hist vs baseline")
    for scheme in args.schemes:
        base = None
        for loop in args.loops:
            if loop == "eager" and n > args.eager_max:
                continue
            for modes in args.modes:
                for dct in args.dcts:
                    for diag in (["full"] if loop == "eager" else args.diag):
                        for prec in (["fp64"] if loop == "eager" else args.precisions):
                            if loop == "eager" and (modes != "thomas" or dct != "fft2n"):
                                continue  # eager only as the baseline
                            try:
                                ms, tc, D, _ = run(p, scheme, loop, modes, dct, diag, prec, args.iters)
                            except Exception as ex:  # noqa: BLE001
                                print(f"{scheme:>6s} {loop:>6s} {modes:>7s} {dct:>7s} {diag:>8s} {prec:>7s}   "
                                      f"FAILED: {type(ex).__name__}: {str(ex)[:80]}")
                                continue
                            if base is None:
                                base, dev = D, 0.0
                            else:
                                m = min(len(base), len(D))
                                dev = float(np.abs(base[:m] - D[:m]).max() / np.abs(base[:m]).max())
                            print(f"{scheme:>6s} {loop:>6s} {modes:>7s} {dct:>7s} {diag:>8s} {prec:>7s} {ms:9.3f} "
                                  f"{tc:10.2f} {peak_mem_mb():9.0f}  {dev:.1e}", flush=True)
