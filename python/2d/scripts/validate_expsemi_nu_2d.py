"""Correctness suite for the 2D ETD FP projection on a NON-UNIFORM time grid
(projection_expsemi_nu_2d.py).

Same shape as validate_expsemi_2d.py, with the checks that only exist once the
grid stops being uniform: the N-weighted adjoint identity, reduction to the
uniform ETD-2D module when the grid IS uniform, and reduction to the 1D
non-uniform ETD on a y-uniform problem. Plus the one check specific to this
module's implementation: the 3-probe tridiagonal extraction against a dense
1D-style build, together with the tridiagonality and symmetry it assumes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from setup import jax_config

jax_config.configure(device="cpu", x64=True)

import jax.numpy as jnp
import numpy as np

from setup.grid_nu_2d import setup_problem_nu_2d
from setup.grid_2d import setup_problem_2d
from setup.operators_nu_2d import build_operators_nu_2d
from setup.operators_2d import build_operators_2d
from setup.pipeline_nu_2d import LadmmConfig as CfgNU
from setup.pipeline_nu_2d import discretize_then_optimize_nu_2d
from setup.pipeline_2d import LadmmConfig as Cfg2D
from setup.pipeline_2d import discretize_then_optimize_2d
from setup.problems_2d import ProblemDef2D, prob_gaussian_2d
from setup.projection_2d import dct2_xy, idct2_xy
from setup.projection_expsemi_2d import (precomp_expsemi_proj_2d,
                                         proj_fokker_planck_expsemi_2d)
from setup.projection_expsemi_nu_2d import (_etd_coeffs_nu, _probe_T_dense,
                                            _probe_T_diagonals,
                                            precomp_expsemi_proj_nu_2d,
                                            proj_fokker_planck_expsemi_nu_2d)
from setup.state_2d import State2D
from setup.time_grid import (clustered_time_grid, graded_ends_time_grid,
                             refine_ends_time_grid, uniform_time_grid)

EPSS = [0.0, 1e-8, 1e-4]
NT, NX, NY = 12, 11, 10
PD = prob_gaussian_2d(mu0=(0.4, 0.4), mu1=(0.6, 0.6), sigma=0.08)
RNG = np.random.default_rng(3)
N_FAIL = 0


def check(name, ok, detail=""):
    global N_FAIL
    if not ok:
        N_FAIL += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def _nu_problem(tg, nx=NX, ny=NY, pd=PD):
    p = setup_problem_nu_2d(pd, tg, nx=nx, ny=ny)
    p.ops = build_operators_nu_2d(p)
    return p


GRIDS = [(clustered_time_grid(NT, strength=0.6), "clustered"),
         (refine_ends_time_grid(NT), "refine_ends"),
         (graded_ends_time_grid(NT, n_boundary=3), "graded_ends(3)")]


def _rand(p):
    return State2D(jnp.asarray(RNG.standard_normal((p.nt - 1, p.nx, p.ny))),
                   jnp.asarray(RNG.standard_normal((p.nt, p.nx - 1, p.ny))),
                   jnp.asarray(RNG.standard_normal((p.nt, p.nx, p.ny - 1))))


def etd_nu_residual(p, s, ep):
    ops = p.ops
    zx, zy = jnp.zeros((p.nt, p.ny)), jnp.zeros((p.nt, p.nx))
    dtv = p.dt_vec[:, None, None]
    mu_prev = jnp.concatenate([p.rho0[None], s.rho], axis=0)
    mu_curr = jnp.concatenate([s.rho, p.rho1[None]], axis=0)
    f = (dct2_xy(mu_curr) - ep.c_vals * dct2_xy(mu_prev)) / dtv
    div_m = ops.deriv_x_at_phi(s.mx, zx, zx) + ops.deriv_y_at_phi(s.my, zy, zy)
    return f + ep.phi_vals * dct2_xy(div_m)


# ==========================================================================
print("1. 3-probe tridiagonal extraction vs a dense 1D-style build")
# ==========================================================================
for tg, label in GRIDS:
    p = _nu_problem(tg)
    lxy = p.lambda_x[:, None] + p.lambda_y[None, :]
    for vareps in EPSS:
        c, phi = _etd_coeffs_nu(lxy, vareps, p.dt_vec)
        T = _probe_T_dense(c, phi, lxy, p.dt_vec, p.dt_dual, p.cell_area)  # (nt,nt,nx,ny)
        main, sub = _probe_T_diagonals(c, phi, lxy, p.dt_vec, p.dt_dual, p.cell_area)
        Tn = np.asarray(T)
        nt = p.nt
        band = np.abs(np.arange(nt)[:, None] - np.arange(nt)[None, :]) <= 1
        offband = float(np.abs(Tn[~band]).max())
        scale = float(np.abs(Tn).max())
        sym = float(np.abs(Tn - Tn.transpose(1, 0, 2, 3)).max())
        d_main = float(np.abs(np.asarray(main) - np.einsum("iijk->ijk", Tn)).max())
        d_sub = float(np.abs(np.asarray(sub) - Tn[np.arange(nt - 1) + 1, np.arange(nt - 1)]).max())
        check(f"T is tridiagonal [{label}, eps={vareps:g}]", offband / scale < 1e-14,
              f"max off-band {offband:.2e} vs scale {scale:.2e}")
        check(f"T is symmetric [{label}, eps={vareps:g}]", sym / scale < 1e-13,
              f"max asym {sym:.2e}")
        check(f"3-probe == dense diagonals [{label}, eps={vareps:g}]",
              max(d_main, d_sub) / scale < 1e-14, f"main {d_main:.2e} sub {d_sub:.2e}")

# ==========================================================================
print("\n2. R* is the N-weighted adjoint of R   (-<Rx,Phi> == <x, R*Phi>_N)")
# ==========================================================================
for tg, label in GRIDS:
    p = _nu_problem(tg)
    lxy = p.lambda_x[:, None] + p.lambda_y[None, :]
    ops = p.ops
    nt, nx, ny = p.nt, p.nx, p.ny
    zx, zy = jnp.zeros((nt, ny)), jnp.zeros((nt, nx))
    dtv = p.dt_vec[:, None, None]
    for vareps in EPSS:
        c, phi = _etd_coeffs_nu(lxy, vareps, p.dt_vec)
        q = jnp.asarray(RNG.standard_normal((nt - 1, nx, ny)))
        bx = jnp.asarray(RNG.standard_normal((nt, nx - 1, ny)))
        by = jnp.asarray(RNG.standard_normal((nt, nx, ny - 1)))
        Phi = jnp.asarray(RNG.standard_normal((nt, nx, ny)))   # dual, in DCT space

        z = jnp.zeros((1, nx, ny))
        q_prev = jnp.concatenate([z, q], axis=0)
        q_curr = jnp.concatenate([q, z], axis=0)
        f = (dct2_xy(q_curr) - c * dct2_xy(q_prev)) / dtv
        div = ops.deriv_x_at_phi(bx, zx, zx) + ops.deriv_y_at_phi(by, zy, zy)
        Rx = f + phi * dct2_xy(div)
        lhs = float(jnp.sum(Rx * Phi))                          # Parseval

        adj = c[1:] * Phi[1:] / dtv[1:] - Phi[:-1] / dtv[:-1]
        q_corr = idct2_xy(adj) / (p.dt_dual * p.cell_area)[:, None, None]
        pw = idct2_xy(phi * Phi)
        w_b = dtv * p.cell_area
        bx_corr = ops.deriv_x_at_m(pw) / w_b
        by_corr = ops.deriv_y_at_m(pw) / w_b
        rhs = float(
            jnp.sum((p.dt_dual * p.cell_area)[:, None, None] * q * q_corr)
            + jnp.sum(w_b * bx * bx_corr) + jnp.sum(w_b * by * by_corr))
        check(f"N-weighted adjoint [{label}, eps={vareps:g}]",
              np.isclose(-lhs, rhs, rtol=1e-10), f"-lhs={-lhs:.8g} rhs={rhs:.8g}")

# ==========================================================================
print("\n3. projection lands on the ETD FP set, is idempotent, stays bounded")
# ==========================================================================
for tg, label in GRIDS:
    p = _nu_problem(tg)
    for vareps in EPSS:
        ep = precomp_expsemi_proj_nu_2d(p, vareps)
        s = _rand(p)
        s1 = proj_fokker_planck_expsemi_nu_2d(s, p, vareps, ep)
        s2 = proj_fokker_planck_expsemi_nu_2d(s1, p, vareps, ep)
        r0 = float(jnp.linalg.norm(etd_nu_residual(p, s, ep)))
        r1 = float(jnp.linalg.norm(etd_nu_residual(p, s1, ep)))
        check(f"lands on FP set [{label}, eps={vareps:g}]", r1 / r0 < 1e-10,
              f"{r0:.2e} -> {r1:.2e}  (ratio {r1 / r0:.1e})")
        idem = max(float(jnp.abs(s2.rho - s1.rho).max()),
                   float(jnp.abs(s2.mx - s1.mx).max()),
                   float(jnp.abs(s2.my - s1.my).max()))
        check(f"idempotent [{label}, eps={vareps:g}]", idem < 1e-9, f"{idem:.2e}")
        norms, cur = [], s
        for _ in range(6):
            cur = proj_fokker_planck_expsemi_nu_2d(cur, p, vareps, ep)
            norms.append(float(jnp.linalg.norm(cur.rho)))
        check(f"bounded under repetition [{label}, eps={vareps:g}]",
              max(norms) / min(norms) < 1.0001, f"{norms[0]:.4e} -> {norms[-1]:.4e}")

# ==========================================================================
print("\n4. reduces to the uniform ETD-2D module on a uniform time grid")
# ==========================================================================
tg_u = uniform_time_grid(NT)
p_nu = _nu_problem(tg_u)
p_u = setup_problem_2d(PD, nt=NT, nx=NX, ny=NY)
p_u.ops = build_operators_2d(p_u)
for vareps in EPSS:
    ep_nu = precomp_expsemi_proj_nu_2d(p_nu, vareps)
    ep_u = precomp_expsemi_proj_2d(p_u, vareps)
    s = _rand(p_u)
    a = proj_fokker_planck_expsemi_nu_2d(s, p_nu, vareps, ep_nu)
    b = proj_fokker_planck_expsemi_2d(s, p_u, vareps, ep_u)
    d = max(float(jnp.abs(a.rho - b.rho).max()), float(jnp.abs(a.mx - b.mx).max()),
            float(jnp.abs(a.my - b.my).max()))
    sc = float(jnp.abs(b.rho).max())
    check(f"projection == uniform ETD-2D (eps={vareps:g})", d / sc < 1e-11,
          f"max diff {d:.2e} (scale {sc:.2e})")
    ITERS = 150
    rn = discretize_then_optimize_nu_2d(p_nu, proj_fokker_planck_expsemi_nu_2d, ep_nu,
                                        vareps, CfgNU(gamma=1.0, tau=1.01, max_iter=ITERS,
                                                      eps_abs=0.0, eps_rel=0.0))
    ru = discretize_then_optimize_2d(p_u, proj_fokker_planck_expsemi_2d, ep_u, vareps,
                                     Cfg2D(gamma=1.0, tau=1.01, max_iter=ITERS,
                                           eps_abs=0.0, eps_rel=0.0))
    dp = max(float(jnp.abs(rn.rho_stag - ru.rho_stag).max()),
             float(jnp.abs(rn.mx_stag - ru.mx_stag).max()),
             float(jnp.abs(rn.my_stag - ru.my_stag).max()))
    check(f"pipeline == uniform ETD-2D (eps={vareps:g})",
          dp / float(jnp.abs(ru.rho_stag).max()) < 1e-10, f"max diff {dp:.2e}")

# ==========================================================================
print("\n5. reduces to the 1D non-uniform ETD on a y-uniform problem")
# ==========================================================================
from optimal_transport.grid_nu import setup_problem_nu as setup_nu_1d
from optimal_transport.operators_nu import build_operators_nu as build_ops_nu_1d
from optimal_transport.problems import prob_gaussian
from optimal_transport.projection_expsemi_nu import (precomp_expsemi_proj_nu,
                                                     proj_fokker_planck_expsemi_nu)
from optimal_transport.state import State as State1D

MU0, MU1, SIG = 1 / 3, 2 / 3, 0.05
pd_flat = ProblemDef2D(
    name="gaussian-y-uniform",
    rho0_func=lambda xx, yy: jnp.exp(-0.5 * ((xx - MU0) / SIG) ** 2) + 0.0 * yy,
    rho1_func=lambda xx, yy: jnp.exp(-0.5 * ((xx - MU1) / SIG) ** 2) + 0.0 * yy,
    mu0=(MU0, MU0), mu1=(MU1, MU1), sigma=SIG,
)
for tg, label in GRIDS:
    nyf = 5
    p2 = _nu_problem(tg, nx=16, ny=nyf, pd=pd_flat)
    p1 = setup_nu_1d(prob_gaussian(mu0=MU0, mu1=MU1, sigma=SIG), tg, nx=16)
    p1.ops = build_ops_nu_1d(p1)
    m0 = float(np.abs(np.asarray(p2.rho0) - np.asarray(p1.rho0)[:, None]).max())
    check(f"y-uniform marginals == 1D [{label}]", m0 < 1e-12, f"{m0:.2e}")
    for vareps in EPSS:
        ep2 = precomp_expsemi_proj_nu_2d(p2, vareps)
        ep1 = precomp_expsemi_proj_nu(p1, vareps)
        nt = p2.nt
        r1d = RNG.standard_normal((nt - 1, 16))
        m1d = RNG.standard_normal((nt, 15))
        s2 = State2D(jnp.asarray(np.repeat(r1d[:, :, None], nyf, axis=2)),
                     jnp.asarray(np.repeat(m1d[:, :, None], nyf, axis=2)),
                     jnp.zeros((nt, 16, nyf - 1)))
        o2 = proj_fokker_planck_expsemi_nu_2d(s2, p2, vareps, ep2)
        o1 = proj_fokker_planck_expsemi_nu(State1D(r1d, m1d), p1, vareps, ep1)
        e_rho = float(np.abs(np.asarray(o2.rho) - np.asarray(o1.rho)[:, :, None]).max())
        e_mx = float(np.abs(np.asarray(o2.mx) - np.asarray(o1.mx)[:, :, None]).max())
        e_my = float(np.abs(np.asarray(o2.my)).max())
        sc = float(np.abs(np.asarray(o1.rho)).max())
        check(f"projection == 1D nu-ETD [{label}, eps={vareps:g}]",
              max(e_rho, e_mx, e_my) / sc < 1e-10,
              f"rho {e_rho:.1e} mx {e_mx:.1e} my {e_my:.1e} (scale {sc:.1e})")

print(f"\n{'ALL PASS' if N_FAIL == 0 else f'{N_FAIL} FAILURE(S)'}")
