"""Correctness checks for the GPU path of the 3D solver: compiled loop,
parallel time solve, ETD projection, precision modes. Companion to
validate_3d.py (which covers the operators, the eager solver and the
reduction to 2D); everything here is small enough to run on a laptop CPU.

  1. DCT kernels     -- fft2n / fftn / matmul all match scipy along every axis
                        of a 4-D array and round-trip.
  2. mode solvers    -- method="dct" (DCT-in-t + Woodbury) equals method="thomas"
                        for CN and ETD, on a random right-hand side, to 1e-12.
  3. ETD == CN       -- at vareps = 0 the two projections coincide term for term.
  4. ETD projection  -- lands on its own constraint (residual assembled in DCT
                        space -> 0) and is idempotent.
  5. jit == eager    -- ladmm_jit with check_every=1 reproduces the eager loop
                        to rounding; check_every=k gives the same iterate at
                        chunk boundaries.
  6. precision       -- fp32, mixed, mixedm, mixedc solves at n=8 stay within
                        the expected distance of the fp64 solve, the LADMM
                        state dtypes are what each mode promises, and mixedc's
                        mass conservation is float64-level.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from setup import jax_config

_ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
_ap.add_argument("--device", default="cpu", choices=["auto", "cpu", "gpu", "tpu"])
_ARGS = _ap.parse_args()
jax_config.configure(device=_ARGS.device, x64=True)

import jax
import jax.numpy as jnp
import numpy as np
import scipy.fft

from setup import projection_3d as P3
from setup.grid_3d import setup_problem_3d
from setup.operators_3d import build_operators_3d
from setup.pipeline_3d import LadmmConfig, discretize_then_optimize_3d
from setup.precision import configure_precision
from setup.problems_3d import prob_gaussian_3d
from setup.projection_3d import (precomp_banded_proj_3d, proj_fokker_planck_banded_3d,
                                 solve_time_modes)
from setup.projection_expsemi_3d import precomp_expsemi_proj_3d, proj_fokker_planck_expsemi_3d
from setup.state_3d import State3D

RNG = np.random.default_rng(0)
_results = []


def check(name, ok, detail):
    _results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def rand_state(nt, nx, ny, nz, dtype=jnp.float64):
    return State3D(jnp.asarray(RNG.standard_normal((nt - 1, nx, ny, nz)), dtype),
                   jnp.asarray(RNG.standard_normal((nt, nx - 1, ny, nz)), dtype),
                   jnp.asarray(RNG.standard_normal((nt, nx, ny - 1, nz)), dtype),
                   jnp.asarray(RNG.standard_normal((nt, nx, ny, nz - 1)), dtype))


def sdiff(a: State3D, b: State3D):
    return float(sum(jnp.linalg.norm(getattr(a, f) - getattr(b, f)) for f in a.__dataclass_fields__))


def snorm(a: State3D):
    return float(sum(jnp.linalg.norm(getattr(a, f)) for f in a.__dataclass_fields__))


def make_problem(n, sigma=0.12):
    p = setup_problem_3d(prob_gaussian_3d(sigma=sigma), nt=n, nx=n, ny=n, nz=n)
    p.ops = build_operators_3d(p)
    return p


# 1 ---------------------------------------------------------------------------
def test_dct_kernels():
    a = RNG.standard_normal((5, 6, 7, 8))
    for m in ("fft2n", "fftn", "matmul"):
        P3.set_dct_method(m)
        worst = 0.0
        for axis in range(4):
            got = np.asarray(P3._dct_axis(jnp.asarray(a), axis=axis))
            want = scipy.fft.dct(a, type=2, norm="ortho", axis=axis)
            back = np.asarray(P3._idct_axis(jnp.asarray(got), axis=axis))
            worst = max(worst, np.abs(got - want).max(), np.abs(back - a).max())
        check(f"dct method {m} vs scipy + round-trip", worst < 1e-11, f"max|diff|={worst:.2e}")
    P3.set_dct_method("matmul")


# 2 ---------------------------------------------------------------------------
def test_mode_solvers():
    n = 8
    p = make_problem(n)
    for vareps in (0.0, 1e-3, 0.05):
        for name, pre in (("CN", precomp_banded_proj_3d), ("ETD", precomp_expsemi_proj_3d)):
            th = pre(p, vareps, method="thomas")
            dc = pre(p, vareps, method="dct")
            rhs = jnp.asarray(RNG.standard_normal((n, n, n, n)))
            a = solve_time_modes(th, rhs, p.lambda_t)
            b = solve_time_modes(dc, rhs, p.lambda_t)
            rel = float(jnp.linalg.norm(a - b) / jnp.linalg.norm(a))
            check(f"{name} dct-modes == thomas (eps={vareps:g})", rel < 1e-11, f"rel={rel:.2e}")


# 3 ---------------------------------------------------------------------------
def test_etd_equals_cn_at_zero_eps():
    n = 8
    p = make_problem(n)
    s = rand_state(n, n, n, n)
    for method in ("thomas", "dct"):
        cn = proj_fokker_planck_banded_3d(s, p, 0.0, precomp_banded_proj_3d(p, 0.0, method=method))
        et = proj_fokker_planck_expsemi_3d(s, p, 0.0, precomp_expsemi_proj_3d(p, 0.0, method=method))
        rel = sdiff(cn, et) / snorm(cn)
        check(f"ETD == CN at eps=0 ({method})", rel < 1e-11, f"rel={rel:.2e}")


# 4 ---------------------------------------------------------------------------
def _etd_residual_hat(p, s, vareps, ep):
    ops = p.ops
    nt, nx, ny, nz = p.nt, p.nx, p.ny, p.nz
    zx, zy, zz = jnp.zeros((nt, ny, nz)), jnp.zeros((nt, nx, nz)), jnp.zeros((nt, nx, ny))
    mu_all = P3.dct3_xyz(jnp.concatenate([p.rho0[None], s.rho, p.rho1[None]], axis=0))
    f_rho = (mu_all[1:] - ep.c_vals * mu_all[:-1]) / p.dt
    div_m = (ops.deriv_x_at_phi(s.mx, zx, zx) + ops.deriv_y_at_phi(s.my, zy, zy)
             + ops.deriv_z_at_phi(s.mz, zz, zz))
    return f_rho + ep.phi_vals * P3.dct3_xyz(div_m)


def test_etd_projection():
    n = 8
    vareps = 0.05
    p = make_problem(n)
    for method in ("thomas", "dct"):
        ep = precomp_expsemi_proj_3d(p, vareps, method=method)
        s = rand_state(n, n, n, n)
        r0 = float(jnp.linalg.norm(_etd_residual_hat(p, s, vareps, ep)))
        ps = proj_fokker_planck_expsemi_3d(s, p, vareps, ep)
        r1 = float(jnp.linalg.norm(_etd_residual_hat(p, ps, vareps, ep)))
        check(f"ETD projection lands on its constraint ({method})", r1 / r0 < 1e-10,
              f"||res|| {r0:.3e} -> {r1:.3e}")
        pps = proj_fokker_planck_expsemi_3d(ps, p, vareps, ep)
        rel = sdiff(pps, ps) / snorm(ps)
        check(f"ETD projection idempotent ({method})", rel < 1e-10, f"rel={rel:.2e}")


# 5 ---------------------------------------------------------------------------
def test_jit_equals_eager():
    n = 8
    vareps = 0.02
    p = make_problem(n)
    ep = precomp_expsemi_proj_3d(p, vareps, method="dct")
    base = dict(gamma=0.3, tau=0.303, max_iter=60, eps_abs=0.0, eps_rel=0.0)
    r_eager = discretize_then_optimize_3d(p, proj_fokker_planck_expsemi_3d, ep, vareps, LadmmConfig(**base))
    x_eager = State3D(r_eager.rho_stag, r_eager.mx_stag, r_eager.my_stag, r_eager.mz_stag)
    for ce in (1, 20):
        r_jit = discretize_then_optimize_3d(p, proj_fokker_planck_expsemi_3d, ep, vareps,
                                            LadmmConfig(**base, jit=True, check_every=ce))
        x_jit = State3D(r_jit.rho_stag, r_jit.mx_stag, r_jit.my_stag, r_jit.mz_stag)
        rel = sdiff(x_jit, x_eager) / snorm(x_eager)
        dD = float(np.abs(r_jit.info.D - r_eager.info.D).max() / np.abs(r_eager.info.D).max())
        check(f"jit(check_every={ce}) == eager after 60 iters", rel < 1e-12 and dD < 1e-10,
              f"rel state diff={rel:.2e}, rel D-history diff={dD:.2e}, compile {r_jit.info.walltime_first_chunk:.2f}s")


# 6 ---------------------------------------------------------------------------
def test_precision_modes():
    n = 8
    vareps = 0.02
    iters = 400
    p64 = make_problem(n)
    ps64 = precomp_expsemi_proj_3d(p64, vareps, method="dct")
    base = dict(gamma=0.3, tau=0.303, max_iter=iters, eps_abs=0.0, eps_rel=0.0, jit=True, check_every=50)

    def solve(precision):
        prob, ps, proj, kw = configure_precision(precision, p64, ps64, proj_fokker_planck_expsemi_3d,
                                                 build_operators_3d)
        if precision == "fp32":
            jax.config.update("jax_enable_x64", False)
        try:
            r = discretize_then_optimize_3d(prob, proj, ps, vareps, LadmmConfig(**base, **kw))
        finally:
            jax.config.update("jax_enable_x64", True)
        return r

    ref = solve("fp64")
    rho_ref = np.asarray(ref.rho_stag, np.float64)
    vol = p64.dx * p64.dy * p64.dz
    expect = {"fp32": (1e-4, "float32", "float32"), "mixed": (1e-4, "float32", "float32"),
              "mixedm": (1e-4, "float64", "float32"), "mixedc": (2e-5, "float64", "float32")}
    for prec, (tol, want_x, want_y) in expect.items():
        r = solve(prec)
        rel = float(np.linalg.norm(np.asarray(r.rho_stag, np.float64) - rho_ref) / np.linalg.norm(rho_ref))
        mass = float(np.abs(np.asarray(r.rho_stag, np.float64).sum(axis=(1, 2, 3)) * vol - 1).max())
        dt_ok = str(r.rho_stag.dtype) == want_x and str(r.rho_cc.dtype) == want_y
        check(f"{prec}: dtypes x={r.rho_stag.dtype} y={r.rho_cc.dtype}", dt_ok, f"want x={want_x} y={want_y}")
        check(f"{prec}: |rho - rho_fp64|/|rho_fp64| < {tol:g}", rel < tol, f"rel={rel:.2e}, max|mass-1|={mass:.1e}")
        if prec == "mixedc":
            check("mixedc: mass conservation float64-level", mass < 1e-10, f"max|mass-1|={mass:.1e}")


if __name__ == "__main__":
    for t in (test_dct_kernels, test_mode_solvers, test_etd_equals_cn_at_zero_eps,
              test_etd_projection, test_jit_equals_eager, test_precision_modes):
        print(f"\n--- {t.__name__} ---")
        t()
    n_fail = _results.count(False)
    print(f"\n{len(_results) - n_fail}/{len(_results)} checks passed")
    sys.exit(1 if n_fail else 0)
