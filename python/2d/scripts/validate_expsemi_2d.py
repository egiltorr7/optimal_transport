"""Correctness suite for the 2D ETD (exponential-time-differencing) FP
projection, projection_expsemi_2d.py, on a UNIFORM grid.

Structured after 3d/scripts/validate_3d.py: cheap structural checks first,
each one chosen so it can only pass if a specific piece of the port is right,
then the comparison against the closed-form Schrodinger bridge last.

The decisive check is #4, reduction to the already-validated 1D ETD solver.
On a problem whose marginals are constant in y, only the ky=0 DCT modes are
excited, so lxy = lambda_x[kx] and every per-mode system is literally the 1D
one; the exact 2D solution is the 1D solution replicated along y with my == 0.
An isotropic problem would pass even with the axes mixed up -- this cannot.
It also lines up bit-for-bit rather than to O(h^2), because with L=1 the 2D
norm (dt*dx*dy, summed over ny identical y-columns) equals the 1D norm
(dt*dx) exactly, so both solvers see the same stopping thresholds.
"""
import sys
from decimal import Decimal, getcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))          # 2d/  -> `setup`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))   # python/ -> `optimal_transport`

from setup import jax_config

jax_config.configure(device="cpu", x64=True)

getcontext().prec = 40

import jax
import jax.numpy as jnp
import numpy as np

from setup.grid_2d import setup_problem_2d
from setup.operators_2d import build_operators_2d
from setup.pipeline_2d import LadmmConfig, discretize_then_optimize_2d
from setup.problems_2d import ProblemDef2D, prob_gaussian_2d
from setup.projection_2d import (dct2_xy, idct2_xy, precomp_banded_proj_2d,
                                 proj_fokker_planck_banded_2d)
from setup.projection_expsemi_2d import (_etd_coeffs, precomp_expsemi_proj_2d,
                                         proj_fokker_planck_expsemi_2d)
from setup.state_2d import State2D

EPSS = [0.0, 1e-8, 1e-4]          # the "eps < 1e-3" stage
N_FAIL = 0


def check(name, ok, detail=""):
    global N_FAIL
    if not ok:
        N_FAIL += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def _problem(nt, nx, ny, prob_def=None, L=1.0):
    p = setup_problem_2d(prob_def or prob_gaussian_2d(), nt=nt, nx=nx, ny=ny, L=L)
    p.ops = build_operators_2d(p)
    return p


def _rand_state(rng, nt, nx, ny):
    return State2D(jnp.asarray(rng.standard_normal((nt - 1, nx, ny))),
                   jnp.asarray(rng.standard_normal((nt, nx - 1, ny))),
                   jnp.asarray(rng.standard_normal((nt, nx, ny - 1))))


def etd_fp_residual(p, s, vareps, ep):
    """The constraint proj_fokker_planck_expsemi_2d claims to project onto:
    (rho_{k+1} - S rho_k)/dt + Phi*(d_x mx + d_y my), built independently of
    the projection's own internals (physical-space semigroup, not the fused
    DCT-space form the module uses)."""
    ops = p.ops
    nt, nx, ny = p.nt, p.nx, p.ny
    zx, zy = jnp.zeros((nt, ny)), jnp.zeros((nt, nx))
    mu_prev = jnp.concatenate([p.rho0[None], s.rho], axis=0)
    mu_curr = jnp.concatenate([s.rho, p.rho1[None]], axis=0)
    S_prev = idct2_xy(ep.c_vals * dct2_xy(mu_prev))          # S rho_k
    div_m = (ops.deriv_x_at_phi(s.mx, zx, zx) + ops.deriv_y_at_phi(s.my, zy, zy))
    Phi_div = idct2_xy(ep.phi_vals * dct2_xy(div_m))
    return (mu_curr - S_prev) / p.dt + Phi_div


# ==========================================================================
print("1. ETD coefficients c = exp(-alpha), phi = (1-c)/alpha")
# ==========================================================================
# Reference in 40-digit `decimal`, NOT np.longdouble: on arm64 longdouble IS
# float64, so a "float128" reference would just be the naive (1-exp(-a))/a the
# module exists to avoid -- it would flag the accurate value as the wrong one.
def _phi_exact(a: float) -> Decimal:
    if a == 0.0:
        return Decimal(1)
    d = Decimal(a)
    return (1 - (-d).exp()) / d


for vareps in [1e-8, 1e-4, 1e-2, 1e-1]:
    p = _problem(16, 16, 16)
    lxy = p.lambda_x[:, None] + p.lambda_y[None, :]
    c, phi = _etd_coeffs(lxy, vareps, p.dt)
    alpha = np.asarray(vareps * lxy * p.dt)
    flat = alpha.ravel()
    ref = [_phi_exact(float(a)) for a in flat]
    got = np.asarray(phi).ravel()
    naive = np.where(flat == 0, 1.0, (1 - np.asarray(c).ravel()) / np.where(flat == 0, 1.0, flat))
    rel_got = max(abs(Decimal(float(g)) - r) / r for g, r in zip(got, ref))
    rel_naive = max(abs(Decimal(float(g)) - r) / r for g, r in zip(naive, ref))
    amin = alpha[alpha > 0].min()
    check(f"phi accurate to float64 eps (vareps={vareps:g})", float(rel_got) < 5e-16,
          f"max rel dev {float(rel_got):.2e},  alpha in [{amin:.2e}, {alpha.max():.2e}]")
    check(f"phi(alpha=0) == 1, c(alpha=0) == 1 (vareps={vareps:g})",
          float(phi[0, 0]) == 1.0 and float(c[0, 0]) == 1.0)
    print(f"        naive (1-c)/alpha would give {float(rel_naive):.1e} "
          f"-- expm1 is {float(rel_naive) / max(float(rel_got), 1e-17):.0f}x better here")

# ==========================================================================
print("\n2. vareps=0: ETD must collapse onto CN term for term")
# ==========================================================================
p = _problem(12, 12, 10)
bp0 = precomp_banded_proj_2d(p, 0.0)
ep0 = precomp_expsemi_proj_2d(p, 0.0)
check("c == 1 and phi == 1 everywhere", bool(jnp.all(ep0.c_vals == 1.0) and jnp.all(ep0.phi_vals == 1.0)))
check("D_mod identical", float(jnp.abs(bp0.D_mod - ep0.D_mod).max()) == 0.0)
check("off-diagonal e identical", float(jnp.abs(bp0.e - ep0.e).max()) == 0.0)
rng = np.random.default_rng(0)
s = _rand_state(rng, p.nt, p.nx, p.ny)
a = proj_fokker_planck_banded_2d(s, p, 0.0, bp0)
b = proj_fokker_planck_expsemi_2d(s, p, 0.0, ep0)
d = max(float(jnp.abs(a.rho - b.rho).max()), float(jnp.abs(a.mx - b.mx).max()),
        float(jnp.abs(a.my - b.my).max()))
check("projection output identical", d < 1e-13, f"max abs diff {d:.2e}")

# ==========================================================================
print("\n3. the projection is an orthogonal projection onto the ETD FP set")
# ==========================================================================
for vareps in EPSS:
    p = _problem(13, 12, 11)
    ep = precomp_expsemi_proj_2d(p, vareps)
    proj = lambda st: proj_fokker_planck_expsemi_2d(st, p, vareps, ep)
    rng = np.random.default_rng(1)
    s = _rand_state(rng, p.nt, p.nx, p.ny)
    s1, s2 = proj(s), proj(proj(s))
    r0 = float(jnp.linalg.norm(etd_fp_residual(p, s, vareps, ep)))
    r1 = float(jnp.linalg.norm(etd_fp_residual(p, s1, vareps, ep)))
    check(f"lands on ETD FP set (eps={vareps:g})", r1 / r0 < 1e-11,
          f"||R x|| {r0:.2e} -> {r1:.2e}  (ratio {r1 / r0:.1e})")
    idem = max(float(jnp.abs(s2.rho - s1.rho).max()), float(jnp.abs(s2.mx - s1.mx).max()),
               float(jnp.abs(s2.my - s1.my).max()))
    check(f"idempotent (eps={vareps:g})", idem < 1e-10, f"||P^2 x - P x||_inf = {idem:.2e}")
    # self-adjointness of the LINEAR part (the projection is affine via the BCs)
    zero = State2D(jnp.zeros_like(s.rho), jnp.zeros_like(s.mx), jnp.zeros_like(s.my))
    off = proj(zero)
    lin = lambda st: proj(st) - off
    u, v = _rand_state(rng, p.nt, p.nx, p.ny), _rand_state(rng, p.nt, p.nx, p.ny)
    ip = lambda A, B: float(jnp.sum(A.rho * B.rho) + jnp.sum(A.mx * B.mx) + jnp.sum(A.my * B.my))
    lhs, rhs = ip(lin(u), v), ip(u, lin(v))
    check(f"self-adjoint (eps={vareps:g})", abs(lhs - rhs) <= 1e-10 * max(1.0, abs(lhs)),
          f"{lhs:.12e} vs {rhs:.12e}")

print(f"\n{'ALL PASS' if N_FAIL == 0 else f'{N_FAIL} FAILURE(S)'} in checks 1-3")

# ==========================================================================
print("\n4. reduction to the 1D ETD solver on a y-uniform problem")
# ==========================================================================
from optimal_transport.grid import setup_problem as setup_problem_1d
from optimal_transport.operators import build_operators as build_operators_1d
from optimal_transport.pipeline import LadmmConfig as LadmmConfig1D
from optimal_transport.pipeline import discretize_then_optimize as dto_1d
from optimal_transport.problems import normal_pdf, prob_gaussian
from optimal_transport.projection_expsemi import (precomp_expsemi_proj,
                                                  proj_fokker_planck_expsemi)
from optimal_transport.state import State as State1D

MU0, MU1, SIG = 1 / 3, 2 / 3, 0.05
nt, nx, ny = 14, 16, 5

# 2D problem whose marginals do not depend on y at all.
pd_flat = ProblemDef2D(
    name="gaussian-y-uniform",
    rho0_func=lambda xx, yy: jnp.exp(-0.5 * ((xx - MU0) / SIG) ** 2) + 0.0 * yy,
    rho1_func=lambda xx, yy: jnp.exp(-0.5 * ((xx - MU1) / SIG) ** 2) + 0.0 * yy,
    mu0=(MU0, MU0), mu1=(MU1, MU1), sigma=SIG,
)
p2 = _problem(nt, nx, ny, pd_flat)
p1 = setup_problem_1d(prob_gaussian(mu0=MU0, mu1=MU1, sigma=SIG), nt=nt, nx=nx)
p1.ops = build_operators_1d(p1)

# With L=1 the 2D per-slice normalization (sum*dx*dy over ny identical
# columns) equals the 1D one (sum*dx), so the marginals must match exactly.
m0 = float(np.abs(np.asarray(p2.rho0) - np.asarray(p1.rho0)[:, None]).max())
check("2D y-uniform marginals == 1D marginals", m0 < 1e-12, f"max abs diff {m0:.2e}")

for vareps in EPSS:
    ep2 = precomp_expsemi_proj_2d(p2, vareps)
    ep1 = precomp_expsemi_proj(p1, vareps)
    rng = np.random.default_rng(7)

    # --- projection level: a y-constant input with my == 0 must stay y-constant ---
    rho1d = rng.standard_normal((nt - 1, nx))
    mx1d = rng.standard_normal((nt, nx - 1))
    s2 = State2D(jnp.asarray(np.repeat(rho1d[:, :, None], ny, axis=2)),
                 jnp.asarray(np.repeat(mx1d[:, :, None], ny, axis=2)),
                 jnp.zeros((nt, nx, ny - 1)))
    o2 = proj_fokker_planck_expsemi_2d(s2, p2, vareps, ep2)
    o1 = proj_fokker_planck_expsemi(State1D(rho1d, mx1d), p1, vareps, ep1)
    e_rho = float(np.abs(np.asarray(o2.rho) - np.asarray(o1.rho)[:, :, None]).max())
    e_mx = float(np.abs(np.asarray(o2.mx) - np.asarray(o1.mx)[:, :, None]).max())
    e_my = float(np.abs(np.asarray(o2.my)).max())
    check(f"projection == 1D ETD (eps={vareps:g})", max(e_rho, e_mx, e_my) < 1e-12,
          f"rho {e_rho:.1e}  mx {e_mx:.1e}  my {e_my:.1e} (must be 0)")

    # --- full pipeline: eps_abs=eps_rel=0 so both run exactly max_iter steps ---
    ITERS = 200
    r2 = discretize_then_optimize_2d(p2, proj_fokker_planck_expsemi_2d, ep2, vareps,
                                     LadmmConfig(gamma=1.0, tau=1.01, max_iter=ITERS,
                                                 eps_abs=0.0, eps_rel=0.0))
    r1 = dto_1d(p1, proj_fokker_planck_expsemi, ep1, vareps,
                LadmmConfig1D(gamma=1.0, tau=1.01, max_iter=ITERS,
                              eps_abs=0.0, eps_rel=0.0))
    check(f"same iteration count (eps={vareps:g})", r2.info.iters == r1.info.iters,
          f"2D {r2.info.iters} vs 1D {r1.info.iters}")
    p_rho = float(np.abs(np.asarray(r2.rho_stag) - np.asarray(r1.rho_stag)[:, :, None]).max())
    p_mx = float(np.abs(np.asarray(r2.mx_stag) - np.asarray(r1.mx_stag)[:, :, None]).max())
    p_my = float(np.abs(np.asarray(r2.my_stag)).max())
    scale = float(np.abs(np.asarray(r1.rho_stag)).max())
    check(f"pipeline == 1D ETD (eps={vareps:g})", max(p_rho, p_mx, p_my) / scale < 1e-11,
          f"rho {p_rho:.1e}  mx {p_mx:.1e}  my {p_my:.1e}  (rho scale {scale:.2e})")
    dd = float(np.abs(r2.info.D - r1.info.D).max() / r1.info.D.max())
    check(f"identical D-residual history (eps={vareps:g})", dd < 1e-11, f"max rel dev {dd:.1e}")

print(f"\n{'ALL PASS' if N_FAIL == 0 else f'{N_FAIL} FAILURE(S)'} in checks 1-4")

# ==========================================================================
print("\n5. the closed-form SB satisfies the DISCRETE ETD equation to O(h^2)")
# ==========================================================================
# Solver-independent: no LADMM runs here. This is the check that pins the
# ETD *discretization*, and it also fixes the error floor that check 6's
# solve-vs-reference comparison can never beat (3d/README.md's caveat).
# Boundary-free marginals (means 0.4/0.6, sigma=0.08): with prob_gaussian_2d's
# own defaults the free-space reference is not a valid target and this
# residual STALLS near 3.5e-2 and then rises -- measured, see the report.
from setup.problems_2d import analytical_sb_gaussian_2d

PD_BF = prob_gaussian_2d(mu0=(0.4, 0.4), mu1=(0.6, 0.6), sigma=0.08)

for vareps in EPSS:
    prev, prev_n, worst, row = None, None, 0.0, []
    for n in (16, 24, 32, 48, 64):
        p = _problem(n, n, n, PD_BF)
        ep = precomp_expsemi_proj_2d(p, vareps)
        rho_a, mx_a, my_a = analytical_sb_gaussian_2d(p, vareps)
        R = etd_fp_residual(p, State2D(rho_a, mx_a, my_a), vareps, ep)
        mu_prev = jnp.concatenate([p.rho0[None], rho_a], axis=0)
        mu_curr = jnp.concatenate([rho_a, p.rho1[None]], axis=0)
        S_prev = idct2_xy(ep.c_vals * dct2_xy(mu_prev))
        scale = float(jnp.linalg.norm((mu_curr - S_prev) / p.dt))
        rel = float(jnp.linalg.norm(R)) / scale
        # The grid sequence is NOT a doubling one (ratios 1.5, 1.33, 1.5, 1.33),
        # so the O(h^2) prediction is ratio^2 per step, not a flat 4x.
        order = None
        if prev is not None:
            order = np.log(prev / rel) / np.log(n / prev_n)
            worst = max(worst, abs(order - 2.0))
        row.append((n, rel, order))
        prev, prev_n = rel, n
    print("        " + "  ".join(
        f"n={n}: {rel:.3e}" + (f" [p={o:.2f}]" if o else "") for n, rel, o in row))
    check(f"reference is O(h^2)-feasible under ETD (eps={vareps:g})", worst < 0.1,
          f"observed order in [{2 - worst:.2f}, {2 + worst:.2f}], finest {row[-1][1]:.2e}")

# ==========================================================================
print("\n6. x <-> y transpose symmetry")
# ==========================================================================
# On a problem invariant under swapping the axes (nx == ny, dx == dy, marginals
# centred on the diagonal), the projection must commute with the swap. This is
# what pins `my` specifically: check 4's y-uniform reduction has my == 0
# throughout, so it can only show that my stays zero, never that a NONZERO my
# is corrected on the right axis with the same shared phi as mx. A sign flip,
# a transposed lambda, or my using deriv_x_at_m would all survive checks 1-5
# and die here.
def _swap(st):
    return State2D(st.rho.transpose(0, 2, 1),
                   st.my.transpose(0, 2, 1),      # my (nt,nx,nym) -> mx (nt,nxm,ny)
                   st.mx.transpose(0, 2, 1))


n = 13
p_sym = _problem(11, n, n, prob_gaussian_2d(mu0=(0.4, 0.4), mu1=(0.6, 0.6), sigma=0.08))
check("the test problem really is x<->y symmetric",
      float(jnp.abs(p_sym.rho0 - p_sym.rho0.T).max()) < 1e-15
      and float(jnp.abs(p_sym.lambda_x - p_sym.lambda_y).max()) < 1e-15)

for vareps in EPSS:
    ep = precomp_expsemi_proj_2d(p_sym, vareps)
    rng = np.random.default_rng(11)
    s_in = _rand_state(rng, p_sym.nt, n, n)
    lhs = proj_fokker_planck_expsemi_2d(_swap(s_in), p_sym, vareps, ep)   # P(T s)
    rhs = _swap(proj_fokker_planck_expsemi_2d(s_in, p_sym, vareps, ep))   # T P(s)
    d = max(float(jnp.abs(lhs.rho - rhs.rho).max()), float(jnp.abs(lhs.mx - rhs.mx).max()),
            float(jnp.abs(lhs.my - rhs.my).max()))
    sc = float(jnp.abs(rhs.rho).max())
    check(f"P(T s) == T P(s) (eps={vareps:g})", d / sc < 1e-12,
          f"max abs diff {d:.2e} (state scale {sc:.2e})")
    # and the momentum correction must not be axis-degenerate: a state with
    # my != 0, mx == 0 must produce a nonzero my correction.
    s_my = State2D(jnp.zeros_like(s_in.rho), jnp.zeros_like(s_in.mx), s_in.my)
    out = proj_fokker_planck_expsemi_2d(s_my, p_sym, vareps, ep)
    zero = proj_fokker_planck_expsemi_2d(
        State2D(jnp.zeros_like(s_in.rho), jnp.zeros_like(s_in.mx), jnp.zeros_like(s_in.my)),
        p_sym, vareps, ep)
    moved = float(jnp.abs(out.my - zero.my).max())
    check(f"my alone drives a my correction (eps={vareps:g})", moved > 1e-3,
          f"||delta my||_inf = {moved:.2e}")

print(f"\n{'ALL PASS' if N_FAIL == 0 else f'{N_FAIL} FAILURE(S)'} in checks 1-6")
