"""Correctness checks for the 3D port (setup/*_3d.py).

Each check is self-contained and prints PASS/FAIL with the measured
quantity, so a regression says which invariant broke rather than just that
some number moved.

  1. DCT           -- _dct_axis matches scipy's orthonormal DCT-II on 4-D
                      arrays along every axis; dct3_xyz round-trips.
  2. adjoints      -- <A u, v> == <u, A* v> for every interp/deriv pair in
                      operators_3d, on random arrays.
  3. Thomas        -- the batched constant-off-diagonal solve reproduces a
                      dense np.linalg.solve of the same tridiagonal system.
  4. projection    -- proj_fokker_planck_banded_3d lands ON the FP
                      constraint set (residual -> 0) and is idempotent.
  5. 2D reduction  -- THE key structural check. Take a problem whose
                      marginals are uniform in z; then the exact 3D solution
                      is the 2D one, constant along z, with mz == 0. So the
                      3D solver must reproduce the (already validated) 2D
                      solver on such a problem. This is what actually tests
                      that lambda_xyz = lx+ly+lz, the z-axis operators, the
                      extra divergence term and mz's prox coupling are all
                      wired up right -- an isotropic problem would pass even
                      with x/y/z mixed up.
  6. reference     -- analytical_sb_gaussian_3d satisfies the discrete FP
                      equation to O(h^2), checked under refinement and
                      without the solver -- so a disagreement in 7 can be
                      attributed to the solver rather than the reference.
  7. analytical    -- relative error of the solve against that reference.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from setup import jax_config

jax_config.configure(device="cpu", x64=True)

import importlib.util

import jax.numpy as jnp
import numpy as np
import scipy.fft

from setup.grid_3d import setup_problem_3d
from setup.operators_3d import build_operators_3d
from setup.pipeline_3d import LadmmConfig, discretize_then_optimize_3d
from setup.problems_3d import ProblemDef3D, analytical_sb_gaussian_3d, prob_gaussian_3d
from setup.projection_3d import (
    _dct_axis,
    _idct_axis,
    dct3_xyz,
    idct3_xyz,
    precomp_banded_proj_3d,
    proj_fokker_planck_banded_3d,
    thomas_batch_precomp,
    thomas_batch_solve,
)
from setup.state_3d import State3D

RNG = np.random.default_rng(0)
_results = []


def check(name, ok, detail):
    _results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def _load_2d_package():
    """Import python/2d/setup as `setup2d`.

    2d/setup and 3d/setup are both top-level packages literally named
    `setup`, so a plain `sys.path` insert would collide (3d's is already
    imported above). Loading the 2D one under an explicit alias, with
    submodule_search_locations set, keeps its internal relative imports
    (`from .state_2d import ...`) resolving inside the alias.
    """
    pkg_dir = ROOT.parent / "2d" / "setup"
    spec = importlib.util.spec_from_file_location(
        "setup2d", pkg_dir / "__init__.py", submodule_search_locations=[str(pkg_dir)]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["setup2d"] = mod
    spec.loader.exec_module(mod)
    import setup2d.grid_2d  # noqa: F401
    import setup2d.operators_2d  # noqa: F401
    import setup2d.pipeline_2d  # noqa: F401
    import setup2d.problems_2d  # noqa: F401
    import setup2d.projection_2d  # noqa: F401
    return mod


# --------------------------------------------------------------------------
# 1. DCT
# --------------------------------------------------------------------------
def test_dct():
    a = RNG.standard_normal((5, 6, 7, 8))
    for axis in range(4):
        got = np.asarray(_dct_axis(jnp.asarray(a), axis=axis))
        want = scipy.fft.dct(a, type=2, norm="ortho", axis=axis)
        err = np.abs(got - want).max()
        check(f"dct axis={axis} vs scipy", err < 1e-11, f"max|diff|={err:.2e}")

        back = np.asarray(_idct_axis(jnp.asarray(got), axis=axis))
        err = np.abs(back - a).max()
        check(f"idct axis={axis} round-trip", err < 1e-11, f"max|diff|={err:.2e}")

    rt = np.asarray(idct3_xyz(dct3_xyz(jnp.asarray(a))))
    err = np.abs(rt - a).max()
    check("dct3_xyz round-trip", err < 1e-11, f"max|diff|={err:.2e}")


# --------------------------------------------------------------------------
# 2. operator adjoints
# --------------------------------------------------------------------------
def test_adjoints():
    nt, nx, ny, nz = 5, 6, 7, 8
    p = setup_problem_3d(prob_gaussian_3d(), nt=nt, nx=nx, ny=ny, nz=nz)
    ops = build_operators_3d(p)
    ntm, nxm, nym, nzm = nt - 1, nx - 1, ny - 1, nz - 1

    zt = jnp.zeros((nx, ny, nz))
    zx = jnp.zeros((nt, ny, nz))
    zy = jnp.zeros((nt, nx, nz))
    zz = jnp.zeros((nt, nx, ny))

    def r(*shape):
        return jnp.asarray(RNG.standard_normal(shape))

    # interp: <interp_a_at_phi(u, 0, 0), v> == <u, interp_a_at_m(v)>
    cases = [
        ("interp_t", r(ntm, nx, ny, nz), r(nt, nx, ny, nz),
         lambda u: ops.interp_t_at_phi(u, zt, zt), ops.interp_t_at_rho),
        ("interp_x", r(nt, nxm, ny, nz), r(nt, nx, ny, nz),
         lambda u: ops.interp_x_at_phi(u, zx, zx), ops.interp_x_at_m),
        ("interp_y", r(nt, nx, nym, nz), r(nt, nx, ny, nz),
         lambda u: ops.interp_y_at_phi(u, zy, zy), ops.interp_y_at_m),
        ("interp_z", r(nt, nx, ny, nzm), r(nt, nx, ny, nz),
         lambda u: ops.interp_z_at_phi(u, zz, zz), ops.interp_z_at_m),
    ]
    # deriv: the *_at_m maps are the NEGATIVE transposes of the *_at_phi maps
    # (a difference operator picks up the sign its adjoint's does not).
    dcases = [
        ("deriv_t", r(ntm, nx, ny, nz), r(nt, nx, ny, nz),
         lambda u: ops.deriv_t_at_phi(u, zt, zt), ops.deriv_t_at_rho),
        ("deriv_x", r(nt, nxm, ny, nz), r(nt, nx, ny, nz),
         lambda u: ops.deriv_x_at_phi(u, zx, zx), ops.deriv_x_at_m),
        ("deriv_y", r(nt, nx, nym, nz), r(nt, nx, ny, nz),
         lambda u: ops.deriv_y_at_phi(u, zy, zy), ops.deriv_y_at_m),
        ("deriv_z", r(nt, nx, ny, nzm), r(nt, nx, ny, nz),
         lambda u: ops.deriv_z_at_phi(u, zz, zz), ops.deriv_z_at_m),
    ]

    for name, u, v, fwd, adj in cases:
        lhs = float(jnp.sum(fwd(u) * v))
        rhs = float(jnp.sum(u * adj(v)))
        rel = abs(lhs - rhs) / max(abs(lhs), 1e-300)
        check(f"adjoint {name}", rel < 1e-12, f"<Au,v>={lhs:.6e} <u,A*v>={rhs:.6e} rel={rel:.2e}")

    for name, u, v, fwd, adj in dcases:
        lhs = float(jnp.sum(fwd(u) * v))
        rhs = float(jnp.sum(u * adj(v)))
        rel = abs(lhs + rhs) / max(abs(lhs), 1e-300)
        check(f"adjoint {name} (anti)", rel < 1e-12, f"<Au,v>={lhs:.6e} -<u,A*v>={-rhs:.6e} rel={rel:.2e}")


# --------------------------------------------------------------------------
# 3. batched Thomas
# --------------------------------------------------------------------------
def test_thomas():
    nt, nx, ny, nz = 9, 3, 4, 2
    D = jnp.asarray(RNG.standard_normal((nt, nx, ny, nz)) + 8.0)  # diag-dominant
    e = jnp.asarray(RNG.standard_normal((nx, ny, nz)))
    F = jnp.asarray(RNG.standard_normal((nt, nx, ny, nz)))

    X = np.asarray(thomas_batch_solve(thomas_batch_precomp(D, e), e, F))

    Dn, en, Fn = np.asarray(D), np.asarray(e), np.asarray(F)
    worst = 0.0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                T = np.diag(Dn[:, i, j, k]) + np.diag(np.full(nt - 1, en[i, j, k]), 1) \
                    + np.diag(np.full(nt - 1, en[i, j, k]), -1)
                want = np.linalg.solve(T, Fn[:, i, j, k])
                worst = max(worst, np.abs(X[:, i, j, k] - want).max())
    check("thomas vs dense solve", worst < 1e-10, f"max|diff|={worst:.2e}")


# --------------------------------------------------------------------------
# 4. projection is a projection ONTO the FP set
# --------------------------------------------------------------------------
def _fp_residual(p, s, vareps):
    ops = p.ops
    nt, nx, ny, nz = p.nt, p.nx, p.ny, p.nz
    zx, zy, zz = jnp.zeros((nt, ny, nz)), jnp.zeros((nt, nx, nz)), jnp.zeros((nt, nx, ny))
    mu_phi = ops.interp_t_at_phi(s.rho, p.rho0, p.rho1)
    lap = (ops.deriv_x_at_phi(ops.deriv_x_at_m(mu_phi), zx, zx)
           + ops.deriv_y_at_phi(ops.deriv_y_at_m(mu_phi), zy, zy)
           + ops.deriv_z_at_phi(ops.deriv_z_at_m(mu_phi), zz, zz))
    return (ops.deriv_t_at_phi(s.rho, p.rho0, p.rho1)
            + ops.deriv_x_at_phi(s.mx, zx, zx)
            + ops.deriv_y_at_phi(s.my, zy, zy)
            + ops.deriv_z_at_phi(s.mz, zz, zz)
            - vareps * lap)


def test_projection():
    nt, nx, ny, nz = 8, 8, 8, 8
    vareps = 0.05
    p = setup_problem_3d(prob_gaussian_3d(sigma=0.12), nt=nt, nx=nx, ny=ny, nz=nz)
    p.ops = build_operators_3d(p)
    bp = precomp_banded_proj_3d(p, vareps)

    s = State3D(
        rho=jnp.asarray(RNG.standard_normal((nt - 1, nx, ny, nz))),
        mx=jnp.asarray(RNG.standard_normal((nt, nx - 1, ny, nz))),
        my=jnp.asarray(RNG.standard_normal((nt, nx, ny - 1, nz))),
        mz=jnp.asarray(RNG.standard_normal((nt, nx, ny, nz - 1))),
    )
    r0 = float(jnp.linalg.norm(_fp_residual(p, s, vareps)))
    ps = proj_fokker_planck_banded_3d(s, p, vareps, bp)
    r1 = float(jnp.linalg.norm(_fp_residual(p, ps, vareps)))
    check("projection lands on FP set", r1 / r0 < 1e-10, f"||res|| {r0:.3e} -> {r1:.3e}")

    pps = proj_fokker_planck_banded_3d(ps, p, vareps, bp)
    d = float(jnp.linalg.norm(pps.rho - ps.rho) + jnp.linalg.norm(pps.mx - ps.mx)
              + jnp.linalg.norm(pps.my - ps.my) + jnp.linalg.norm(pps.mz - ps.mz))
    scale = float(jnp.linalg.norm(ps.rho))
    check("projection idempotent", d / scale < 1e-10, f"||P(Px)-Px||/||Px||={d / scale:.2e}")


# --------------------------------------------------------------------------
# 5. reduction to 2D on a z-uniform problem
# --------------------------------------------------------------------------
def test_reduces_to_2d():
    s2 = _load_2d_package()
    nt, nx, ny, nz = 12, 12, 12, 6
    vareps = 0.1
    sigma = 0.12
    mu0xy, mu1xy = (0.35, 0.4), (0.6, 0.65)  # deliberately NOT isotropic in x/y
    cfg = dict(gamma=1.0, tau=1.01, max_iter=4000, eps_abs=1e-12, eps_rel=1e-10)

    # --- 2D reference ---
    d2 = s2.problems_2d.ProblemDef2D(
        name="aniso",
        rho0_func=lambda xx, yy: s2.problems_2d.normal_pdf_2d(xx, yy, mu0xy, sigma),
        rho1_func=lambda xx, yy: s2.problems_2d.normal_pdf_2d(xx, yy, mu1xy, sigma),
        mu0=mu0xy, mu1=mu1xy, sigma=sigma,
    )
    p2 = s2.grid_2d.setup_problem_2d(d2, nt=nt, nx=nx, ny=ny)
    p2.ops = s2.operators_2d.build_operators_2d(p2)
    bp2 = s2.projection_2d.precomp_banded_proj_2d(p2, vareps)
    r2 = s2.pipeline_2d.discretize_then_optimize_2d(
        p2, s2.projection_2d.proj_fokker_planck_banded_2d, bp2, vareps,
        s2.pipeline_2d.LadmmConfig(**cfg),
    )

    # --- 3D, same marginals extruded uniformly along z ---
    d3 = ProblemDef3D(
        name="aniso_extruded",
        rho0_func=lambda xx, yy, zz: s2.problems_2d.normal_pdf_2d(xx, yy, mu0xy, sigma) + 0 * zz,
        rho1_func=lambda xx, yy, zz: s2.problems_2d.normal_pdf_2d(xx, yy, mu1xy, sigma) + 0 * zz,
        mu0=(*mu0xy, 0.5), mu1=(*mu1xy, 0.5), sigma=sigma,
    )
    p3 = setup_problem_3d(d3, nt=nt, nx=nx, ny=ny, nz=nz)
    p3.ops = build_operators_3d(p3)
    bp3 = precomp_banded_proj_3d(p3, vareps)
    r3 = discretize_then_optimize_3d(
        p3, proj_fokker_planck_banded_3d, bp3, vareps, LadmmConfig(**cfg),
    )

    # mz must be identically zero: nothing drives transport along z.
    mz = float(jnp.abs(r3.mz_stag).max())
    check("extruded problem has mz == 0", mz < 1e-12, f"max|mz|={mz:.2e}")

    # rho must be constant along z.
    rho3 = np.asarray(r3.rho_stag)
    zvar = float(np.abs(rho3 - rho3.mean(axis=3, keepdims=True)).max() / np.abs(rho3).max())
    check("extruded rho constant in z", zvar < 1e-10, f"max rel z-variation={zvar:.2e}")

    # And the z-slice must equal the 2D solution. rho0 is normalized to
    # integrate to 1 over [0,1]^3 vs [0,1]^2 -- with L=1 and a z-uniform
    # extrusion those coincide, so no rescaling is needed.
    rho2 = np.asarray(r2.rho_stag)
    err = np.linalg.norm(rho3[:, :, :, 0] - rho2) / np.linalg.norm(rho2)
    check("3D z-slice == 2D solution (rho)", err < 1e-8, f"rel err={err:.3e}")

    mx3 = np.asarray(r3.mx_stag)[:, :, :, 0]
    mx2 = np.asarray(r2.mx_stag)
    errm = np.linalg.norm(mx3 - mx2) / np.linalg.norm(mx2)
    check("3D z-slice == 2D solution (mx)", errm < 1e-8, f"rel err={errm:.3e}")

    my3 = np.asarray(r3.my_stag)[:, :, :, 0]
    my2 = np.asarray(r2.my_stag)
    errmy = np.linalg.norm(my3 - my2) / np.linalg.norm(my2)
    check("3D z-slice == 2D solution (my)", errmy < 1e-8, f"rel err={errmy:.3e}")


# --------------------------------------------------------------------------
# 6. the analytical reference itself satisfies the FP equation
# --------------------------------------------------------------------------
def test_analytical_is_fp_feasible():
    """Check analytical_sb_gaussian_3d WITHOUT involving the solver.

    The exact SB solution satisfies d_t rho + div m = eps*Laplacian rho
    exactly in the continuum, so on the staggered grid its residual must be
    a pure truncation error: O(h^2), i.e. it must drop by ~4x per halving of
    h. This pins down alpha, sig2_t and the velocity field independently of
    the solver -- test_analytical below can only ever tell us that solver
    and reference agree, not that either is right.
    """
    vareps, sigma = 0.05, 0.08
    prev = None
    for n in (8, 16, 32):
        p = setup_problem_3d(prob_gaussian_3d(sigma=sigma), nt=n, nx=n, ny=n, nz=n)
        p.ops = build_operators_3d(p)
        rho_a, mx_a, my_a, mz_a = analytical_sb_gaussian_3d(p, vareps)
        res = _fp_residual(p, State3D(rho_a, mx_a, my_a, mz_a), vareps)
        # Relative to d_t rho, the leading term the residual must cancel.
        scale = float(jnp.linalg.norm(p.ops.deriv_t_at_phi(rho_a, p.rho0, p.rho1)))
        rel = float(jnp.linalg.norm(res)) / scale
        if prev is None:
            check(f"analytical FP residual n={n}", rel < 0.1, f"rel={rel:.3e}")
        else:
            rate = prev / rel
            check(f"analytical FP residual n={n} (2nd order)", rate > 2.5,
                  f"rel={rel:.3e}, dropped {rate:.2f}x (expect ~4x)")
        prev = rel


# --------------------------------------------------------------------------
# 7. vs the closed-form 3D Schrodinger bridge
# --------------------------------------------------------------------------
def test_analytical():
    nt, nx, ny, nz = 12, 12, 12, 12
    vareps = 0.1
    p = setup_problem_3d(prob_gaussian_3d(sigma=0.12), nt=nt, nx=nx, ny=ny, nz=nz)
    p.ops = build_operators_3d(p)
    bp = precomp_banded_proj_3d(p, vareps)
    res = discretize_then_optimize_3d(
        p, proj_fokker_planck_banded_3d, bp, vareps,
        LadmmConfig(gamma=1.0, tau=1.01, max_iter=8000, eps_abs=1e-12, eps_rel=1e-10),
    )
    rho_a, mx_a, my_a, mz_a = analytical_sb_gaussian_3d(p, vareps)
    for nm, got, want in [("rho", res.rho_stag, rho_a), ("mx", res.mx_stag, mx_a),
                          ("my", res.my_stag, my_a), ("mz", res.mz_stag, mz_a)]:
        e = float(np.linalg.norm(np.asarray(got) - np.asarray(want)) / np.linalg.norm(np.asarray(want)))
        # Coarse grid + clipped Gaussian tails -- this is a sanity bound, not
        # a convergence test (see scripts/sb_gaussian_3d.py for refinement).
        check(f"analytical SB {nm}", e < 0.1, f"rel err={e:.3e}  (iters={res.info.iters})")


if __name__ == "__main__":
    for t in (test_dct, test_adjoints, test_thomas, test_projection,
              test_reduces_to_2d, test_analytical_is_fp_feasible, test_analytical):
        print(f"\n--- {t.__name__} ---")
        t()
    n_fail = _results.count(False)
    print(f"\n{len(_results) - n_fail}/{len(_results)} checks passed")
    sys.exit(1 if n_fail else 0)
