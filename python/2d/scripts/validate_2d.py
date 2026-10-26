"""Correctness suite for the UNIFORM 2D CN/banded FP projection
(projection_2d.py) and the pipeline around it.

This is the suite the uniform 2D tree never had: 3d/ has validate_3d.py and
the non-uniform 2D track has validate_nu_2d.py, but projection_2d.py itself
was only ever exercised indirectly (by sb_gaussian_2d.py's demo, and by
validate_3d.py's reduces-to-2D check). Structured to match
validate_expsemi_2d.py so the CN and ETD results are directly comparable,
and run across the full vareps range rather than one value, since that is
where the two schemes are supposed to differ.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from setup import jax_config

jax_config.configure(device="cpu", x64=True)

import jax.numpy as jnp
import numpy as np
from scipy.fft import dct as sdct

from setup.grid_2d import setup_problem_2d
from setup.operators_2d import build_operators_2d
from setup.pipeline_2d import LadmmConfig, discretize_then_optimize_2d
from setup.problems_2d import (ProblemDef2D, analytical_sb_gaussian_2d,
                               prob_gaussian_2d)
from setup.projection_2d import (_dct_axis, dct2_xy, idct2_xy,
                                 precomp_banded_proj_2d,
                                 proj_fokker_planck_banded_2d,
                                 thomas_batch_precomp, thomas_batch_solve)
from setup.projection_expsemi_2d import (precomp_expsemi_proj_2d,
                                         proj_fokker_planck_expsemi_2d)
from setup.state_2d import State2D

EPSS = [0.0, 1e-8, 1e-4, 1e-3, 1e-2, 1e-1]
PD_BF = prob_gaussian_2d(mu0=(0.4, 0.4), mu1=(0.6, 0.6), sigma=0.08)
RNG = np.random.default_rng(0)
N_FAIL = 0


def check(name, ok, detail=""):
    global N_FAIL
    if not ok:
        N_FAIL += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def _problem(nt, nx, ny, pd=None, L=1.0):
    p = setup_problem_2d(pd or PD_BF, nt=nt, nx=nx, ny=ny, L=L)
    p.ops = build_operators_2d(p)
    return p


def _rand(p):
    return State2D(jnp.asarray(RNG.standard_normal((p.nt - 1, p.nx, p.ny))),
                   jnp.asarray(RNG.standard_normal((p.nt, p.nx - 1, p.ny))),
                   jnp.asarray(RNG.standard_normal((p.nt, p.nx, p.ny - 1))))


def cn_fp_residual(p, s, vareps):
    """d_t rho + d_x mx + d_y my - eps*Laplacian(rho), built independently of
    proj_fokker_planck_banded_2d's own internals."""
    ops = p.ops
    zx, zy = jnp.zeros((p.nt, p.ny)), jnp.zeros((p.nt, p.nx))
    mu_phi = ops.interp_t_at_phi(s.rho, p.rho0, p.rho1)
    lap = (ops.deriv_x_at_phi(ops.deriv_x_at_m(mu_phi), zx, zx)
           + ops.deriv_y_at_phi(ops.deriv_y_at_m(mu_phi), zy, zy))
    return (ops.deriv_t_at_phi(s.rho, p.rho0, p.rho1)
            + ops.deriv_x_at_phi(s.mx, zx, zx)
            + ops.deriv_y_at_phi(s.my, zy, zy) - vareps * lap)


# ==========================================================================
print("1. DCT-II vs scipy, and the dct2_xy round trip")
# ==========================================================================
f = RNG.standard_normal((5, 7, 6))
for ax in (0, 1, 2):
    e = float(np.abs(np.asarray(_dct_axis(jnp.asarray(f), ax))
                     - sdct(f, type=2, norm="ortho", axis=ax)).max())
    check(f"_dct_axis == scipy.fft.dct axis={ax}", e < 1e-12, f"{e:.2e}")
e = float(np.abs(np.asarray(idct2_xy(dct2_xy(jnp.asarray(f)))) - f).max())
check("idct2_xy(dct2_xy(f)) == f", e < 1e-12, f"{e:.2e}")

# ==========================================================================
print("\n2. operator adjoint identities")
# ==========================================================================
p = _problem(7, 6, 5)
ops = p.ops
nt, nx, ny = p.nt, p.nx, p.ny
ntm, nxm, nym = nt - 1, nx - 1, ny - 1
zt, zx, zy = jnp.zeros((nx, ny)), jnp.zeros((nt, ny)), jnp.zeros((nt, nx))
for name, F, Ft, su, sv in [
    ("interp_t_at_phi^T == interp_t_at_rho",
     lambda u: ops.interp_t_at_phi(u, zt, zt), ops.interp_t_at_rho, (ntm, nx, ny), (nt, nx, ny)),
    ("interp_x_at_phi^T == interp_x_at_m",
     lambda u: ops.interp_x_at_phi(u, zx, zx), ops.interp_x_at_m, (nt, nxm, ny), (nt, nx, ny)),
    ("interp_y_at_phi^T == interp_y_at_m",
     lambda u: ops.interp_y_at_phi(u, zy, zy), ops.interp_y_at_m, (nt, nx, nym), (nt, nx, ny)),
    ("deriv_t_at_phi^T == -deriv_t_at_rho",
     lambda u: ops.deriv_t_at_phi(u, zt, zt), lambda v: -ops.deriv_t_at_rho(v), (ntm, nx, ny), (nt, nx, ny)),
    ("deriv_x_at_phi^T == -deriv_x_at_m",
     lambda u: ops.deriv_x_at_phi(u, zx, zx), lambda v: -ops.deriv_x_at_m(v), (nt, nxm, ny), (nt, nx, ny)),
    ("deriv_y_at_phi^T == -deriv_y_at_m",
     lambda u: ops.deriv_y_at_phi(u, zy, zy), lambda v: -ops.deriv_y_at_m(v), (nt, nx, nym), (nt, nx, ny)),
]:
    u, v = jnp.asarray(RNG.standard_normal(su)), jnp.asarray(RNG.standard_normal(sv))
    lhs, rhs = float(jnp.sum(F(u) * v)), float(jnp.sum(u * Ft(v)))
    check(name, abs(lhs - rhs) <= 1e-11 * max(1.0, abs(lhs)), f"{lhs:.10e} vs {rhs:.10e}")

# ==========================================================================
print("\n3. batched Thomas vs a dense solve")
# ==========================================================================
K = 5
D = jnp.asarray(RNG.standard_normal((nt, K)) * 0.1 + 5.0)
ee = jnp.asarray(RNG.standard_normal(K) * 0.1 - 1.0)
F = jnp.asarray(RNG.standard_normal((nt, K)))
X = thomas_batch_solve(thomas_batch_precomp(D, ee), ee, F)
worst = max(float(np.abs(np.linalg.solve(
    np.diag(np.asarray(D)[:, k])
    + np.diag(np.full(nt - 1, float(ee[k])), 1)
    + np.diag(np.full(nt - 1, float(ee[k])), -1), np.asarray(F)[:, k]) - np.asarray(X)[:, k]).max())
    for k in range(K))
check("thomas_batch_solve == np.linalg.solve", worst < 1e-10, f"{worst:.2e}")

# ==========================================================================
print("\n4. the projection is an orthogonal projection onto the CN FP set")
# ==========================================================================
for vareps in EPSS:
    p = _problem(13, 12, 11)
    bp = precomp_banded_proj_2d(p, vareps)
    proj = lambda st: proj_fokker_planck_banded_2d(st, p, vareps, bp)
    s = _rand(p)
    s1, s2 = proj(s), proj(proj(s))
    r0 = float(jnp.linalg.norm(cn_fp_residual(p, s, vareps)))
    r1 = float(jnp.linalg.norm(cn_fp_residual(p, s1, vareps)))
    check(f"lands on CN FP set (eps={vareps:g})", r1 / r0 < 1e-10,
          f"{r0:.2e} -> {r1:.2e} (ratio {r1 / r0:.1e})")
    idem = max(float(jnp.abs(s2.rho - s1.rho).max()), float(jnp.abs(s2.mx - s1.mx).max()),
               float(jnp.abs(s2.my - s1.my).max()))
    check(f"idempotent (eps={vareps:g})", idem < 1e-9, f"{idem:.2e}")
    zs = State2D(jnp.zeros_like(s.rho), jnp.zeros_like(s.mx), jnp.zeros_like(s.my))
    off = proj(zs)
    lin = lambda st: proj(st) - off
    u, v = _rand(p), _rand(p)
    ip = lambda A, B: float(jnp.sum(A.rho * B.rho) + jnp.sum(A.mx * B.mx) + jnp.sum(A.my * B.my))
    lhs, rhs = ip(lin(u), v), ip(u, lin(v))
    check(f"self-adjoint (eps={vareps:g})", abs(lhs - rhs) <= 1e-9 * max(1.0, abs(lhs)),
          f"{lhs:.10e} vs {rhs:.10e}")

# ==========================================================================
print("\n5. x <-> y transpose symmetry")
# ==========================================================================
def _swap(st):
    return State2D(st.rho.transpose(0, 2, 1), st.my.transpose(0, 2, 1),
                   st.mx.transpose(0, 2, 1))


n = 13
p_sym = _problem(11, n, n)
for vareps in EPSS:
    bp = precomp_banded_proj_2d(p_sym, vareps)
    s_in = _rand(p_sym)
    a = proj_fokker_planck_banded_2d(_swap(s_in), p_sym, vareps, bp)
    b = _swap(proj_fokker_planck_banded_2d(s_in, p_sym, vareps, bp))
    d = max(float(jnp.abs(a.rho - b.rho).max()), float(jnp.abs(a.mx - b.mx).max()),
            float(jnp.abs(a.my - b.my).max()))
    check(f"P(T s) == T P(s) (eps={vareps:g})", d / float(jnp.abs(b.rho).max()) < 1e-11,
          f"{d:.2e}")

# ==========================================================================
print("\n6. reduction to the 1D CN solver on a y-uniform problem")
# ==========================================================================
from optimal_transport.grid import setup_problem as setup_1d
from optimal_transport.operators import build_operators as build_ops_1d
from optimal_transport.pipeline import LadmmConfig as Cfg1D
from optimal_transport.pipeline import discretize_then_optimize as dto_1d
from optimal_transport.problems import prob_gaussian
from optimal_transport.projection import (precomp_banded_proj,
                                          proj_fokker_planck_banded)
from optimal_transport.state import State as State1D

MU0, MU1, SIG = 1 / 3, 2 / 3, 0.05
pd_flat = ProblemDef2D(
    name="gaussian-y-uniform",
    rho0_func=lambda xx, yy: jnp.exp(-0.5 * ((xx - MU0) / SIG) ** 2) + 0.0 * yy,
    rho1_func=lambda xx, yy: jnp.exp(-0.5 * ((xx - MU1) / SIG) ** 2) + 0.0 * yy,
    mu0=(MU0, MU0), mu1=(MU1, MU1), sigma=SIG,
)
NT1, NX1, NY1 = 14, 16, 5
p2 = _problem(NT1, NX1, NY1, pd_flat)
p1 = setup_1d(prob_gaussian(mu0=MU0, mu1=MU1, sigma=SIG), nt=NT1, nx=NX1)
p1.ops = build_ops_1d(p1)
m0 = float(np.abs(np.asarray(p2.rho0) - np.asarray(p1.rho0)[:, None]).max())
check("2D y-uniform marginals == 1D", m0 < 1e-12, f"{m0:.2e}")
for vareps in EPSS:
    bp2, bp1 = precomp_banded_proj_2d(p2, vareps), precomp_banded_proj(p1, vareps)
    r1d = RNG.standard_normal((NT1 - 1, NX1))
    m1d = RNG.standard_normal((NT1, NX1 - 1))
    s2 = State2D(jnp.asarray(np.repeat(r1d[:, :, None], NY1, axis=2)),
                 jnp.asarray(np.repeat(m1d[:, :, None], NY1, axis=2)),
                 jnp.zeros((NT1, NX1, NY1 - 1)))
    o2 = proj_fokker_planck_banded_2d(s2, p2, vareps, bp2)
    o1 = proj_fokker_planck_banded(State1D(r1d, m1d), p1, vareps, bp1)
    e_rho = float(np.abs(np.asarray(o2.rho) - np.asarray(o1.rho)[:, :, None]).max())
    e_mx = float(np.abs(np.asarray(o2.mx) - np.asarray(o1.mx)[:, :, None]).max())
    e_my = float(np.abs(np.asarray(o2.my)).max())
    check(f"projection == 1D CN (eps={vareps:g})",
          max(e_rho, e_mx, e_my) / float(np.abs(np.asarray(o1.rho)).max()) < 1e-10,
          f"rho {e_rho:.1e} mx {e_mx:.1e} my {e_my:.1e}")
    ITERS = 150
    ru = discretize_then_optimize_2d(p2, proj_fokker_planck_banded_2d, bp2, vareps,
                                     LadmmConfig(gamma=1.0, tau=1.01, max_iter=ITERS,
                                                 eps_abs=0.0, eps_rel=0.0))
    r1 = dto_1d(p1, proj_fokker_planck_banded, bp1, vareps,
                Cfg1D(gamma=1.0, tau=1.01, max_iter=ITERS, eps_abs=0.0, eps_rel=0.0))
    dp = max(float(np.abs(np.asarray(ru.rho_stag) - np.asarray(r1.rho_stag)[:, :, None]).max()),
             float(np.abs(np.asarray(ru.mx_stag) - np.asarray(r1.mx_stag)[:, :, None]).max()),
             float(np.abs(np.asarray(ru.my_stag)).max()))
    check(f"pipeline == 1D CN (eps={vareps:g}, {ITERS} iters)",
          ru.info.iters == r1.info.iters
          and dp / float(np.abs(np.asarray(r1.rho_stag)).max()) < 1e-10,
          f"max diff {dp:.2e}, iters {ru.info.iters}/{r1.info.iters}")

# ==========================================================================
print("\n7. the closed form satisfies the discrete CN equation to O(h^2)")
# ==========================================================================
# The free-space closed form does not satisfy the no-flux walls the
# discretization imposes, so it is only a valid O(h^2) target while its tails
# die before the wall. That is a property of the TEST PROBLEM, not of either
# scheme, and it is checked here explicitly rather than left to look like a
# solver failure: at sigma=0.08 the mid-time width is
# sqrt(sigma^2 + 2*alpha*t(1-t)) with alpha = sqrt(sigma^4+eps^2)-sigma^2,
# which at eps=1e-1 puts the wall only 2.17 sd out (9.6% of peak density) and
# the observed order collapses to ~1.2. Widening the domain is the fix --
# grid.py/grid_nu.py now take an x_min offset for exactly this, and grid_2d.py
# has no 2D equivalent yet.
WALL_TOL = 1e-4   # rho_wall/rho_peak above which the reference is not a target


def _wall_fraction(vareps, sigma=0.08):
    alpha = np.sqrt(sigma**4 + vareps**2) - sigma**2
    sig_t = np.sqrt(sigma**2 + 2 * alpha * 0.25)      # widest point, t=1/2
    return float(np.exp(-0.5 * (0.5 / sig_t) ** 2))


for vareps in EPSS:
    prev = prev_n = None
    worst_dev, row = 0.0, []
    for nn in (16, 24, 32, 48):
        pp = _problem(nn, nn, nn)
        rho_a, mx_a, my_a = analytical_sb_gaussian_2d(pp, vareps)
        R = cn_fp_residual(pp, State2D(rho_a, mx_a, my_a), vareps)
        scale = float(jnp.linalg.norm(pp.ops.deriv_t_at_phi(rho_a, pp.rho0, pp.rho1)))
        rel = float(jnp.linalg.norm(R)) / scale
        order = None
        if prev is not None:
            order = np.log(prev / rel) / np.log(nn / prev_n)
            worst_dev = max(worst_dev, abs(order - 2.0))
        row.append((nn, rel, order))
        prev, prev_n = rel, nn
    wf = _wall_fraction(vareps)
    print("        " + "  ".join(f"n={a}: {b:.3e}" + (f" [p={c:.2f}]" if c else "")
                                 for a, b, c in row))
    if wf > WALL_TOL:
        # Expected: assert the reference is KNOWN-BAD rather than assert O(h^2)
        # on a target that cannot be met. Flipping to a pass here would hide it.
        check(f"reference correctly detected as boundary-invalid (eps={vareps:g})",
              worst_dev > 0.3,
              f"wall/peak={wf:.2e} > {WALL_TOL:g}, order in [{2 - worst_dev:.2f}, "
              f"{2 + worst_dev:.2f}] -- widen the domain to test this eps")
    else:
        check(f"reference is O(h^2)-feasible under CN (eps={vareps:g})", worst_dev < 0.15,
              f"wall/peak={wf:.2e}, order in [{2 - worst_dev:.2f}, {2 + worst_dev:.2f}]")

# ==========================================================================
print("\n8. positivity: CN vs ETD, min rho after one projection of the closed form")
# ==========================================================================
# The known failure mode of the CN/banded scheme is not accuracy but
# positivity (it is a plain projection, with nothing enforcing rho >= 0).
# Solver-free probe: project the analytical state and look at min rho.
print(f"        {'eps':>8s} {'CN min rho':>13s} {'ETD min rho':>13s}   {'peak rho':>10s}")
for vareps in EPSS:
    pp = _problem(24, 24, 24)
    rho_a, mx_a, my_a = analytical_sb_gaussian_2d(pp, vareps)
    st = State2D(rho_a, mx_a, my_a)
    cn = proj_fokker_planck_banded_2d(st, pp, vareps, precomp_banded_proj_2d(pp, vareps))
    etd = proj_fokker_planck_expsemi_2d(st, pp, vareps, precomp_expsemi_proj_2d(pp, vareps))
    print(f"        {vareps:>8g} {float(cn.rho.min()):>13.3e} {float(etd.rho.min()):>13.3e}"
          f"   {float(rho_a.max()):>10.3e}")

print(f"\n{'ALL PASS' if N_FAIL == 0 else f'{N_FAIL} FAILURE(S)'}")
