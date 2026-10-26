"""Time the pieces of one jitted 2D LADMM iteration separately, and count the
kernels XLA emits for each, to see where a per-iteration floor comes from.

    python scripts/bench_parts_2d.py --device gpu --sizes 32 128
"""
import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="auto")
ap.add_argument("--sizes", type=int, nargs="+", default=[32, 128])
ap.add_argument("--reps", type=int, default=50)
ap.add_argument("--eps", type=float, default=1e-3)
args = ap.parse_args()

from setup import jax_config

jax_config.configure(device=args.device, x64=True)

import jax
import jax.numpy as jnp
import numpy as np

from setup import projection_2d as P
from setup.grid_2d import setup_problem_2d
from setup.operators_2d import build_operators_2d
from setup.pipeline_2d import LadmmConfig, discretize_then_optimize_2d
from setup.problems_2d import prob_gaussian_2d
from setup.projection_2d import (dct2_xy, idct2_xy, precomp_banded_proj_2d,
                                 proj_fokker_planck_banded_2d, solve_time_modes)
from setup.prox_2d import prox_ke_cc_2d
from setup.state_2d import State2D

print(f"JAX {jax.__version__} backend={jax.default_backend()} x64={jax.config.jax_enable_x64}")
PD = prob_gaussian_2d(mu0=(0.4, 0.4), mu1=(0.6, 0.6), sigma=0.08)


def kernels(fn, *a):
    txt = jax.jit(fn).lower(*a).compile().as_text()
    n_fusion = len(re.findall(r"^\s*(?:ROOT )?%?\S+ = \S+ fusion\(", txt, re.M))
    n_cc = len(re.findall(r"custom-call\(", txt))
    n_while = len(re.findall(r"^\s*(?:ROOT )?%?\S+ = \S+ while\(", txt, re.M))
    n_copy = len(re.findall(r"^\s*(?:ROOT )?%?\S+ = \S+ copy\(", txt, re.M))
    return f"fusions={n_fusion:4d} custom-calls={n_cc:3d} whiles={n_while} copies={n_copy:3d}"


def bench(name, fn, *a):
    f = jax.jit(fn)
    out = f(*a)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(args.reps):
        out = f(*a)
    jax.block_until_ready(out)
    ms = 1e3 * (time.perf_counter() - t0) / args.reps
    print(f"  {name:38s} {ms:8.3f} ms   {kernels(fn, *a)}", flush=True)
    return ms


for n in args.sizes:
    p = setup_problem_2d(PD, nt=n, nx=n, ny=n)
    p.ops = build_operators_2d(p)
    nt, nx, ny = p.nt, p.nx, p.ny
    rng = np.random.default_rng(0)
    x = State2D(jnp.asarray(rng.standard_normal((nt - 1, nx, ny))), jnp.asarray(rng.standard_normal((nt, nx - 1, ny))),
                jnp.asarray(rng.standard_normal((nt, nx, ny - 1))))
    y = State2D(jnp.asarray(rng.standard_normal((nt, nx, ny))), jnp.asarray(rng.standard_normal((nt, nx, ny))),
                jnp.asarray(rng.standard_normal((nt, nx, ny))))
    f = jnp.asarray(rng.standard_normal((nt, nx, ny)))
    zx, zy = jnp.zeros((nt, ny)), jnp.zeros((nt, nx))
    ops = p.ops
    print(f"\n=== n={n} ===")
    for m in ("fft2n", "fftn", "matmul"):
        P.set_dct_method(m)
        bench(f"dct along t [{m}]", lambda a: P._dct_axis(a, 0), f)
        bench(f"dct along x [{m}]", lambda a: P._dct_axis(a, 1), f)
        bench(f"dct along y [{m}]", lambda a: P._dct_axis(a, 2), f)
        bench(f"dct2_xy + idct2_xy [{m}]", lambda a: idct2_xy(dct2_xy(a)), f)
    P.set_dct_method("fft2n")
    bp_th = precomp_banded_proj_2d(p, args.eps, "thomas")
    bp_dc = precomp_banded_proj_2d(p, args.eps, "dct")
    bench("solve_time_modes [thomas scan]", lambda b, a: solve_time_modes(b, a, p.lambda_t), bp_th, f)
    bench("solve_time_modes [dct]", lambda b, a: solve_time_modes(b, a, p.lambda_t), bp_dc, f)
    bench("projection CN [thomas]", lambda s, b: proj_fokker_planck_banded_2d(s, p, args.eps, b), x, bp_th)
    bench("projection CN [dct]", lambda s, b: proj_fokker_planck_banded_2d(s, p, args.eps, b), x, bp_dc)
    bench("prox_ke_cc_2d", lambda s: prox_ke_cc_2d(s, 1.0), y)
    A = lambda s: State2D(ops.interp_t_at_phi(s.rho, p.rho0, p.rho1), ops.interp_x_at_phi(s.mx, zx, zx), ops.interp_y_at_phi(s.my, zy, zy))
    At = lambda v: State2D(ops.interp_t_at_rho(v.rho), ops.interp_x_at_m(v.mx), ops.interp_y_at_m(v.my))
    bench("A_fn", A, x)
    bench("At_fn", At, y)
    nrm = lambda v: jnp.sqrt(jnp.sum(v.rho**2) + jnp.sum(v.mx**2) + jnp.sum(v.my**2))
    bench("norm of a state", nrm, y)
    bench("7 norms stacked", lambda v: jnp.stack([nrm(v), nrm(v * 2.0), nrm(v * 3.0), nrm(v * 4.0), nrm(v * 5.0), nrm(v * 6.0), nrm(v * 7.0)]), y)
    # whole iteration through the real pipeline (check_every=1 and 10)
    for ce in (1, 10):
        n_it = max(ce, (args.reps // ce) * ce)   # whole chunks only, so nothing retraces inside the timing
        cfg = LadmmConfig(gamma=1.0, tau=1.01, max_iter=ce, eps_abs=0.0, eps_rel=0.0, jit=True, check_every=ce)
        r = discretize_then_optimize_2d(p, proj_fokker_planck_banded_2d, bp_dc, args.eps, cfg)
        jax.block_until_ready(r.rho_stag)
        cfg = LadmmConfig(gamma=1.0, tau=1.01, max_iter=n_it, eps_abs=0.0, eps_rel=0.0, jit=True, check_every=ce)
        t0 = time.perf_counter()
        r = discretize_then_optimize_2d(p, proj_fokker_planck_banded_2d, bp_dc, args.eps, cfg)
        jax.block_until_ready(r.rho_stag)
        t_all = time.perf_counter() - t0
        print(f"  {'full iteration [dct, check_every=%d]' % ce:38s} {1e3 * t_all / n_it:8.3f} ms   "
              f"(solver-reported {1e3 * r.info.walltime / n_it:.3f} ms/iter over {n_it} iters)")
