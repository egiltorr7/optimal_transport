"""Isolate where the per-iteration floor of the jitted LADMM step comes from.

    python scripts/bench_step_2d.py --device gpu --sizes 32 128

Times (ms per iteration), all through jax.jit:
  step            one full iteration, jitted on its own (no scan)
  scan10          the same step inside lax.scan(length=10), per iteration
  step-nodiag     step with every norm/diagnostic removed
  step-noproj     step with the projection replaced by the identity
  step-noprox     step with the KE prox replaced by the identity
  step-nonorm7    step keeping only the two stopping-rule norms
and the HLO op histogram of `step` vs `scan10` (copies, fusions, whiles).
"""
import argparse
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="auto")
ap.add_argument("--sizes", type=int, nargs="+", default=[32, 128])
ap.add_argument("--reps", type=int, default=30)
ap.add_argument("--eps", type=float, default=1e-3)
args = ap.parse_args()

from setup import jax_config

jax_config.configure(device=args.device, x64=True)

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from setup.grid_2d import setup_problem_2d
from setup.ladmm import LadmmOpts
from setup.ladmm_jit import _Carry, _make_step
from setup.operators_2d import build_operators_2d
from setup.problems_2d import prob_gaussian_2d
from setup.projection_2d import precomp_banded_proj_2d, proj_fokker_planck_banded_2d
from setup.prox_2d import prox_ke_cc_2d
from setup.state_2d import State2D

print(f"JAX {jax.__version__} backend={jax.default_backend()} x64={jax.config.jax_enable_x64}")
PD = prob_gaussian_2d(mu0=(0.4, 0.4), mu1=(0.6, 0.6), sigma=0.08)


def hlo_hist(fn, *a):
    txt = jax.jit(fn).lower(*a).compile().as_text()
    ops = Counter()
    for m in re.finditer(r"^\s*(?:ROOT )?%?[\w.\-]+ = \S+ ([a-z\-]+)\(", txt, re.M):
        ops[m.group(1)] += 1
    keep = {k: v for k, v in ops.items() if k in ("fusion", "copy", "while", "fft", "custom-call", "dynamic-update-slice",
                                                   "dynamic-slice", "transpose", "reduce", "dot", "get-tuple-element", "tuple")}
    return " ".join(f"{k}={v}" for k, v in sorted(keep.items()))


def timeit(fn, *a, per=1):
    f = jax.jit(fn)
    out = f(*a)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(args.reps):
        out = f(*a)
    jax.block_until_ready(out)
    return 1e3 * (time.perf_counter() - t0) / args.reps / per


for n in args.sizes:
    p = setup_problem_2d(PD, nt=n, nx=n, ny=n)
    p.ops = build_operators_2d(p)
    ops, rho0, rho1 = p.ops, p.rho0, p.rho1
    nt, nx, ny = p.nt, p.nx, p.ny
    dt, dx, dy = p.dt, p.dx, p.dy
    zx, zy = jnp.zeros((nt, ny)), jnp.zeros((nt, nx))
    bp = precomp_banded_proj_2d(p, args.eps, "dct")

    A_fn = lambda x: State2D(ops.interp_t_at_phi(x.rho, rho0, rho1), ops.interp_x_at_phi(x.mx, zx, zx), ops.interp_y_at_phi(x.my, zy, zy))
    At_fn = lambda v: State2D(ops.interp_t_at_rho(v.rho), ops.interp_x_at_m(v.mx), ops.interp_y_at_m(v.my))
    B_fn = lambda y: y * -1.0
    proj = lambda v, step, aux: proj_fokker_planck_banded_2d(v, p, args.eps, aux)
    noproj = lambda v, step, aux: v
    solve_y = lambda delta, z_hat, gamma: prox_ke_cc_2d(z_hat - delta * (1.0 / gamma), 1.0 / gamma)
    noprox = lambda delta, z_hat, gamma: z_hat - delta * (1.0 / gamma)
    nrm = lambda v: jnp.sqrt(dt * dx * dy * (jnp.sum(v.rho**2) + jnp.sum(v.mx**2) + jnp.sum(v.my**2)))
    zero = lambda v: jnp.asarray(0.0)

    t_stag = (jnp.arange(1, nt) * dt)[:, None, None]
    x0 = State2D((1 - t_stag) * rho0[None] + t_stag * rho1[None], jnp.zeros((nt, nx - 1, ny)), jnp.zeros((nt, nx, ny - 1)))
    t_cc = ((jnp.arange(1, nt + 1) - 0.5) * dt)[:, None, None]
    y0 = State2D((1 - t_cc) * rho0[None] + t_cc * rho1[None], jnp.zeros((nt, nx, ny)), jnp.zeros((nt, nx, ny)))
    b = State2D.zeros_like(y0)
    By0 = B_fn(y0)
    carry = _Carry(x0, y0, By0, State2D.zeros_like(b), A_fn(x0) + By0 - b)
    opts = LadmmOpts(gamma=1.0, tau=1.01, max_iter=1, eps_abs=0.0, eps_rel=0.0)

    def make(prox=proj, sy=solve_y, norm=nrm, dual=None):
        return _make_step(prox, sy, A_fn, At_fn, B_fn, b, opts, norm, dual or norm)

    step = make()
    print(f"\n=== n={n} ===")
    print(f"  {'step':16s} {timeit(step, carry, bp):8.3f} ms   {hlo_hist(step, carry, bp)}")
    scan10 = lambda c, aux: lax.scan(lambda c, _: step(c, aux), c, None, length=10)
    print(f"  {'scan10':16s} {timeit(scan10, carry, bp, per=10):8.3f} ms   {hlo_hist(scan10, carry, bp)}")
    scan10u = lambda c, aux: lax.scan(lambda c, _: step(c, aux), c, None, length=10, unroll=10)
    print(f"  {'scan10 unrolled':16s} {timeit(scan10u, carry, bp, per=10):8.3f} ms")
    fori10 = lambda c, aux: lax.fori_loop(0, 10, lambda i, c: step(c, aux)[0], c)
    print(f"  {'fori10':16s} {timeit(fori10, carry, bp, per=10):8.3f} ms")
    print(f"  {'step-nodiag':16s} {timeit(make(norm=zero), carry, bp):8.3f} ms")
    print(f"  {'step-noproj':16s} {timeit(make(prox=noproj), carry, bp):8.3f} ms")
    print(f"  {'step-noprox':16s} {timeit(make(sy=noprox), carry, bp):8.3f} ms")
    print(f"  {'step-noproj-noprox':16s} {timeit(make(prox=noproj, sy=noprox), carry, bp):8.3f} ms")
    print(f"  {'step-noproj-nodiag':16s} {timeit(make(prox=noproj, norm=zero), carry, bp):8.3f} ms", flush=True)
    # python-level chain of the pieces, each its own jit (eager-ish composition of jitted parts)
    jA, jAt, jproj, jprox = jax.jit(A_fn), jax.jit(At_fn), jax.jit(lambda v, aux: proj(v, 0.0, aux)), jax.jit(lambda d, z: solve_y(d, z, 1.0))
    def chain(c, aux):
        g = jAt(c.D_prev - c.delta)
        x = jproj(c.x - g, aux)
        Ax = jA(x)
        y = jprox(c.delta, Ax)
        return _Carry(x, y, B_fn(y), c.delta - (Ax - y), Ax - y)
    out = chain(carry, bp); jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(args.reps):
        out = chain(carry, bp)
    jax.block_until_ready(out)
    print(f"  {'chain of jits':16s} {1e3 * (time.perf_counter() - t0) / args.reps:8.3f} ms (no diagnostics)")
